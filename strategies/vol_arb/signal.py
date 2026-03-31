"""
signal.py — VRP 信号生成器
核心逻辑已在 portfolio_analysis.py 中验证，这里封装成可复用的类。
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VRPSignalGenerator:
    """
    Volatility Risk Premium (VRP) signal generator.

    Computes the spread between ATM implied volatility and GARCH-forecast
    realized volatility, and generates SELL_VOL / BUY_VOL / NEUTRAL signals.

    Parameters
    ----------
    risk_free_rate : float
        Annualized risk-free rate (e.g. 0.02 for 2%).
    garch_lookback : int
        Number of trading days of history to use when fitting GARCH.
    forecast_horizon : int
        Number of days to forecast (average vol over this horizon vs ATM IV).
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        garch_lookback: int = 504,
        forecast_horizon: int = 5,
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.garch_lookback = garch_lookback
        self.forecast_horizon = forecast_horizon

        self._garch_model = None
        self._last_fit_date: str = ""
        self._garch_level: str = ""  # "GJR-GARCH", "GARCH", "EWMA", or ""

    # ------------------------------------------------------------------
    # GARCH fitting
    # ------------------------------------------------------------------

    def fit_garch(self, futures_daily: pd.DataFrame) -> bool:
        """
        Fit volatility model with cascade fallback:
          1. GJR-GARCH(1,1) skewt  — best quality, needs ≥120 rows
          2. GARCH(1,1) t-dist      — simpler, more robust to short series
          3. EWMA (λ=0.94)          — always succeeds, never non-stationary

        Each level is accepted only if persistence < 1 and the
        5-day forecast is in (1%, 150%) annualised vol.

        Returns True if any level succeeded, False if all failed.
        """
        from models.volatility.garch_model import GJRGARCHModel, GARCHModel

        close = futures_daily["close"].dropna()
        if len(close) < 60:
            logger.warning("fit_garch: insufficient data (%d rows)", len(close))
            return False

        log_returns = np.log(close / close.shift(1)).dropna()
        if len(log_returns) > self.garch_lookback:
            log_returns = log_returns.iloc[-self.garch_lookback:]

        def _valid(fc: float) -> bool:
            return not np.isnan(fc) and 0.01 < fc < 1.5

        # --- Level 1: GJR-GARCH(1,1) skewt ---
        if len(log_returns) >= 120:
            try:
                model = GJRGARCHModel(dist="skewt")
                fit_result = model.fit(log_returns)
                if fit_result.persistence < 1.0:
                    fc = model.forecast_period_avg(horizon=self.forecast_horizon)
                    if _valid(fc):
                        self._garch_model = model
                        self._garch_level = "GJR-GARCH"
                        return True
                logger.debug("GJR-GARCH skewt: persistence=%.4f or bad forecast, trying fallback",
                             fit_result.persistence)
            except Exception as exc:
                logger.debug("GJR-GARCH skewt failed: %s", exc)

        # --- Level 2: Standard GARCH(1,1) t-dist (no asymmetry) ---
        try:
            inner = GARCHModel(p=1, q=1, o=0, dist="t")
            fit_dict = inner.fit(log_returns)
            if fit_dict["persistence"] < 1.0:
                fc = inner.predict(horizon=self.forecast_horizon)["mean_vol"]
                if _valid(fc):
                    # Wrap GARCHModel to expose forecast_period_avg interface
                    class _GARCHAdapter:
                        def __init__(self, m: GARCHModel, h: int) -> None:
                            self._m = m
                            self._h = h

                        def forecast_period_avg(self, horizon: int = 5) -> float:
                            return float(self._m.predict(horizon=horizon)["mean_vol"])

                    self._garch_model = _GARCHAdapter(inner, self.forecast_horizon)
                    self._garch_level = "GARCH"
                    logger.debug("GARCH(1,1) t-dist fallback: forecast=%.3f", fc)
                    return True
            logger.debug("GARCH(1,1) t-dist: persistence=%.4f or bad forecast, trying EWMA",
                         fit_dict["persistence"])
        except Exception as exc:
            logger.debug("GARCH(1,1) t-dist failed: %s", exc)

        # --- Level 3: EWMA (always succeeds) ---
        try:
            # RiskMetrics-style EWMA: λ≈0.94 ↔ span≈33
            ewma_var = float(log_returns.ewm(span=33, adjust=False).var().iloc[-1])
            ewma_vol = float(np.sqrt(ewma_var * 252))
            if _valid(ewma_vol):
                _vol = ewma_vol

                class _EWMAModel:
                    def forecast_period_avg(self, horizon: int = 5) -> float:  # noqa: ARG002
                        return _vol

                self._garch_model = _EWMAModel()
                self._garch_level = "EWMA"
                logger.debug("GARCH cascade exhausted; EWMA fallback vol=%.3f", ewma_vol)
                return True
        except Exception as exc:
            logger.debug("EWMA fallback failed: %s", exc)

        self._garch_model = None
        self._garch_level = ""
        return False

    # ------------------------------------------------------------------
    # VRP computation
    # ------------------------------------------------------------------

    def compute_vrp(
        self,
        futures_daily: pd.DataFrame,
        options_chain: pd.DataFrame,
        trade_date: str,
        futures_price: float | None = None,
    ) -> Dict:
        """
        Compute VRP signal for the given trade date.

        Parameters
        ----------
        futures_daily : pd.DataFrame
            Futures OHLCV history up to (and including) trade_date.
            Must have 'close' column.
        options_chain : pd.DataFrame
            Current day option chain. Should have: exercise_price (or
            strike_price), call_put, close, expire_date. Volume optional.
        trade_date : str
            Current date (YYYYMMDD).
        futures_price : float, optional
            Futures close price for the day. If None, uses last close from
            futures_daily. Used for market ATM IV (VRP calculation).

        Returns
        -------
        dict with keys:
            vrp, atm_iv, atm_iv_market, atm_iv_structural,
            garch_forecast, rv_20d, signal,
            atm_strike, atm_expiry, implied_forward, days_to_expiry
        Returns empty dict on failure.
        """
        try:
            return self._compute_vrp_impl(futures_daily, options_chain, trade_date, futures_price)
        except Exception as exc:
            logger.warning("compute_vrp failed on %s: %s", trade_date, exc)
            return {}

    def _compute_vrp_impl(
        self,
        futures_daily: pd.DataFrame,
        options_chain: pd.DataFrame,
        trade_date: str,
        futures_price: float | None = None,
    ) -> Dict:
        close_series = futures_daily["close"].dropna()
        if len(close_series) < 60:
            return {}

        spot = float(close_series.iloc[-1])
        if futures_price is None:
            futures_price = spot

        log_returns = np.log(close_series / close_series.shift(1)).dropna()

        # Step 1: Fit/update GARCH (cascade: GJR-GARCH → GARCH → EWMA)
        if not self.fit_garch(futures_daily.iloc[-self.garch_lookback:] if len(futures_daily) > self.garch_lookback else futures_daily):
            return {}

        # Step 2: GARCH forecast (horizon-day average vol)
        try:
            garch_forecast = self._garch_model.forecast_period_avg(horizon=self.forecast_horizon)
        except Exception as exc:
            logger.warning("GARCH forecast failed: %s", exc)
            return {}

        # Step 3: 20-day realized vol
        rv_20d = float(log_returns.tail(20).std() * np.sqrt(252)) if len(log_returns) >= 20 else float("nan")

        # Step 4 & 5: Process options chain for ATM IV
        if options_chain is None or options_chain.empty:
            # No options data: compute partial VRP using RV as IV proxy
            vrp = rv_20d - garch_forecast if not np.isnan(rv_20d) else None
            return {
                "vrp": vrp,
                "atm_iv": rv_20d,
                "atm_iv_market": rv_20d,
                "atm_iv_structural": rv_20d,
                "garch_forecast": garch_forecast,
                "garch_level": self._garch_level,
                "rv_20d": rv_20d,
                "signal": "NEUTRAL",
                "atm_strike": None,
                "atm_expiry": None,
                "implied_forward": spot,
                "days_to_expiry": None,
            }

        # Structural ATM IV (Forward-based) — used for Greeks / contract selection
        structural_iv, atm_strike, atm_expiry, implied_fwd, dte = self._extract_atm_iv(
            options_chain, trade_date, spot
        )

        if structural_iv is None or np.isnan(structural_iv):
            return {}

        # Market ATM IV (futures-price-based) — used for VRP signal
        market_iv = self._extract_market_atm_iv(
            options_chain, trade_date, futures_price
        )
        # Fallback to structural IV if market IV fails
        if market_iv is None or np.isnan(market_iv):
            market_iv = structural_iv

        # Step 6: VRP uses market IV (avoids Forward-based circular reasoning)
        vrp = market_iv - garch_forecast

        # Step 7: Signal
        if vrp > 0.02:
            signal = "SELL_VOL"
        elif vrp < -0.01:
            signal = "BUY_VOL"
        else:
            signal = "NEUTRAL"

        return {
            "vrp": vrp,
            "atm_iv": market_iv,                # VRP-facing IV (backward compat)
            "atm_iv_market": market_iv,          # explicit: futures-price-based
            "atm_iv_structural": structural_iv,  # explicit: Forward-based
            "garch_forecast": garch_forecast,
            "garch_level": self._garch_level,
            "rv_20d": rv_20d,
            "signal": signal,
            "atm_strike": atm_strike,
            "atm_expiry": atm_expiry,
            "implied_forward": implied_fwd,
            "days_to_expiry": dte,
        }

    def _extract_atm_iv(
        self,
        options_chain: pd.DataFrame,
        trade_date: str,
        spot_price: float,
    ) -> Tuple[Optional[float], Optional[float], Optional[str], float, Optional[int]]:
        """
        Extract ATM IV from the options chain, preferring contracts with
        20–60 DTE. Returns (atm_iv, atm_strike, atm_expiry, implied_forward, dte).
        """
        from models.pricing.implied_vol import calc_implied_vol

        # Normalize column names
        chain = options_chain.copy()
        if "strike_price" in chain.columns and "exercise_price" not in chain.columns:
            chain = chain.rename(columns={"strike_price": "exercise_price"})
        if "call_put" not in chain.columns and "option_type" in chain.columns:
            chain = chain.rename(columns={"option_type": "call_put"})

        required = {"exercise_price", "call_put", "close", "expire_date"}
        if not required.issubset(chain.columns):
            missing = required - set(chain.columns)
            logger.debug("_extract_atm_iv: missing columns %s", missing)
            return None, None, None, spot_price, None

        ref_dt = pd.Timestamp(trade_date)
        best_iv: Optional[float] = None
        best_strike: Optional[float] = None
        best_expiry: Optional[str] = None
        best_fwd = spot_price
        best_dte: Optional[int] = None
        best_score = float("inf")  # lower = better (prefer 30-45 DTE)

        # Group by expiry
        try:
            chain["expire_date"] = chain["expire_date"].astype(str).str.replace("-", "")
        except Exception:
            pass

        expiries = chain["expire_date"].unique()
        for expiry in sorted(expiries):
            try:
                exp_dt = pd.Timestamp(str(expiry))
                dte = (exp_dt - ref_dt).days
            except Exception:
                continue

            if dte < 14:
                continue

            exp_chain = chain[chain["expire_date"] == expiry].copy()
            if exp_chain.empty:
                continue

            T = dte / 365.0

            # Compute implied forward via PCP
            implied_fwd = self._calc_implied_forward(exp_chain, T, spot_price)

            # Find ATM strike (closest to implied forward)
            strikes = exp_chain["exercise_price"].astype(float).unique()
            if len(strikes) == 0:
                continue

            atm_strike = float(min(strikes, key=lambda k: abs(k - implied_fwd)))

            # Get ATM call and put prices
            atm_rows = exp_chain[exp_chain["exercise_price"].astype(float) == atm_strike]
            if atm_rows.empty:
                continue

            # Filter to liquid options (volume > 0 if available)
            if "volume" in atm_rows.columns:
                liquid = atm_rows[atm_rows["volume"].fillna(0) > 0]
                if not liquid.empty:
                    atm_rows = liquid

            # Compute IV for each ATM option and average
            ivs = []
            for _, row in atm_rows.iterrows():
                try:
                    cp = str(row["call_put"]).upper()
                    if cp not in ("C", "P", "CALL", "PUT"):
                        continue
                    opt_type = "C" if cp in ("C", "CALL") else "P"
                    price = float(row["close"])
                    if price <= 0 or np.isnan(price):
                        continue
                    iv = calc_implied_vol(
                        market_price=price,
                        S=implied_fwd,
                        K=atm_strike,
                        T=T,
                        r=self.risk_free_rate,
                        option_type=opt_type,
                    )
                    if iv is not None and 0.01 <= iv <= 5.0:
                        ivs.append(iv)
                except Exception:
                    continue

            if not ivs:
                continue

            avg_iv = float(np.mean(ivs))

            # Prefer 20-60 DTE contracts; score = distance from target of 35 DTE
            if 20 <= dte <= 60:
                score = abs(dte - 35)
                if score < best_score:
                    best_score = score
                    best_iv = avg_iv
                    best_strike = atm_strike
                    best_expiry = str(expiry)
                    best_fwd = implied_fwd
                    best_dte = dte
            elif best_iv is None:
                # Use as fallback even outside preferred range
                best_iv = avg_iv
                best_strike = atm_strike
                best_expiry = str(expiry)
                best_fwd = implied_fwd
                best_dte = dte

        return best_iv, best_strike, best_expiry, best_fwd, best_dte

    def _extract_market_atm_iv(
        self,
        options_chain: pd.DataFrame,
        trade_date: str,
        futures_price: float,
    ) -> Optional[float]:
        """
        Extract market ATM IV using futures price directly (not Forward).

        Avoids Forward-based circular reasoning: Forward derived from option
        prices via PCP → IV naturally "reasonable" → VRP understated.

        Uses same DTE selection as _extract_atm_iv but:
        - ATM strike = closest to futures_price (not implied Forward)
        - S = futures_price in BS inversion (not Forward)
        """
        from models.pricing.implied_vol import calc_implied_vol

        chain = options_chain.copy()
        if "strike_price" in chain.columns and "exercise_price" not in chain.columns:
            chain = chain.rename(columns={"strike_price": "exercise_price"})
        if "call_put" not in chain.columns and "option_type" in chain.columns:
            chain = chain.rename(columns={"option_type": "call_put"})

        required = {"exercise_price", "call_put", "close", "expire_date"}
        if not required.issubset(chain.columns):
            return None

        ref_dt = pd.Timestamp(trade_date)
        try:
            chain["expire_date"] = chain["expire_date"].astype(str).str.replace("-", "")
        except Exception:
            pass

        best_iv: Optional[float] = None
        best_score = float("inf")

        for expiry in sorted(chain["expire_date"].unique()):
            try:
                dte = (pd.Timestamp(str(expiry)) - ref_dt).days
            except Exception:
                continue
            if dte < 14:
                continue

            exp_chain = chain[chain["expire_date"] == expiry]
            if exp_chain.empty:
                continue

            T = dte / 365.0

            # ATM strike closest to futures price (NOT Forward)
            strikes = exp_chain["exercise_price"].astype(float).unique()
            if len(strikes) == 0:
                continue
            atm_strike = float(min(strikes, key=lambda k: abs(k - futures_price)))

            atm_rows = exp_chain[exp_chain["exercise_price"].astype(float) == atm_strike]
            if atm_rows.empty:
                continue

            # Compute IV with futures price as underlying
            ivs = []
            for _, row in atm_rows.iterrows():
                try:
                    cp = str(row["call_put"]).upper()
                    opt_type = "C" if cp in ("C", "CALL") else "P" if cp in ("P", "PUT") else None
                    if opt_type is None:
                        continue
                    price = float(row["close"])
                    if price <= 0 or np.isnan(price):
                        continue
                    iv = calc_implied_vol(
                        market_price=price,
                        S=futures_price,   # key difference: futures price, not Forward
                        K=atm_strike,
                        T=T,
                        r=self.risk_free_rate,
                        option_type=opt_type,
                    )
                    if iv is not None and 0.01 <= iv <= 5.0:
                        ivs.append(iv)
                except Exception:
                    continue

            if not ivs:
                continue

            avg_iv = float(np.mean(ivs))

            # Same DTE preference as _extract_atm_iv: prefer 20-60 DTE, target 35
            if 20 <= dte <= 60:
                score = abs(dte - 35)
                if score < best_score:
                    best_score = score
                    best_iv = avg_iv
            elif best_iv is None:
                best_iv = avg_iv

        return best_iv

    def _calc_implied_forward(
        self,
        exp_chain: pd.DataFrame,
        T: float,
        fallback: float,
    ) -> float:
        """
        Compute PCP-implied forward price for a single expiry.
        F = K + (C - P) * e^(rT)
        Returns weighted median across valid strike pairs.
        """
        try:
            discount = np.exp(self.risk_free_rate * T)
            cp_map: Dict[float, Dict[str, float]] = {}

            for _, row in exp_chain.iterrows():
                k = float(row["exercise_price"])
                cp = str(row["call_put"]).upper()
                px = float(row["close"] or 0)
                if px > 0:
                    cp_map.setdefault(k, {})[cp] = px

            forwards = []
            for k, sides in cp_map.items():
                c_key = next((x for x in sides if x in ("C", "CALL")), None)
                p_key = next((x for x in sides if x in ("P", "PUT")), None)
                if c_key and p_key:
                    forwards.append(k + (sides[c_key] - sides[p_key]) * discount)

            if forwards:
                return float(np.median(forwards))
        except Exception as exc:
            logger.debug("_calc_implied_forward failed: %s", exc)

        return fallback

    # ------------------------------------------------------------------
    # Strangle leg selection
    # ------------------------------------------------------------------

    def select_strangle_legs(
        self,
        options_chain: pd.DataFrame,
        implied_forward: float,
        expire_date: str,
        put_delta_target: float = -0.15,
        call_delta_target: float = 0.15,
        r: float = 0.02,
    ) -> Dict:
        """
        Select OTM put and call strikes for a strangle.

        For each candidate put (strike < forward): compute BS delta using IV.
        For each candidate call (strike > forward): compute BS delta using IV.
        Select the put closest to put_delta_target and call closest to call_delta_target.

        Parameters
        ----------
        options_chain : pd.DataFrame
            Full options chain for the day. Must have exercise_price, call_put,
            close, expire_date (or ts_code).
        implied_forward : float
            Implied forward price for the expiry.
        expire_date : str
            Target expiry date (YYYYMMDD).
        put_delta_target : float
            Target delta for put leg (negative, e.g. -0.15).
        call_delta_target : float
            Target delta for call leg (positive, e.g. 0.15).
        r : float
            Risk-free rate.

        Returns
        -------
        dict with keys "put" and "call", each containing:
            ts_code, exercise_price, delta, close, iv, expire_date
        Returns empty dict if insufficient data.
        """
        try:
            return self._select_strangle_impl(
                options_chain, implied_forward, expire_date,
                put_delta_target, call_delta_target, r
            )
        except Exception as exc:
            logger.warning("select_strangle_legs failed: %s", exc)
            return {}

    def _select_strangle_impl(
        self,
        options_chain: pd.DataFrame,
        implied_forward: float,
        expire_date: str,
        put_delta_target: float,
        call_delta_target: float,
        r: float,
    ) -> Dict:
        from models.pricing.implied_vol import calc_implied_vol, bs_d1_d2
        from scipy.stats import norm

        chain = options_chain.copy()

        # Normalize column names
        if "strike_price" in chain.columns and "exercise_price" not in chain.columns:
            chain = chain.rename(columns={"strike_price": "exercise_price"})
        if "call_put" not in chain.columns and "option_type" in chain.columns:
            chain = chain.rename(columns={"option_type": "call_put"})

        required = {"exercise_price", "call_put", "close", "expire_date"}
        if not required.issubset(chain.columns):
            return {}

        # Normalize expire_date format
        expire_norm = str(expire_date).replace("-", "")
        chain["expire_date"] = chain["expire_date"].astype(str).str.replace("-", "")
        exp_chain = chain[chain["expire_date"] == expire_norm]

        if exp_chain.empty:
            return {}

        ref_dt = pd.Timestamp(expire_norm)
        # Use current date embedded in expire_date context - use 30 DTE as estimate
        # We compute T from the expire_date relative to now but we don't have trade_date here
        # Estimate T from the chain: for backtest purposes assume the DTE was pre-computed
        dte_est = 30  # fallback
        if "days_to_expiry" in options_chain.columns:
            dte_est = int(options_chain["days_to_expiry"].iloc[0])
        T = dte_est / 365.0
        if T <= 0:
            return {}

        best_put: Optional[Dict] = None
        best_put_dist = float("inf")
        best_call: Optional[Dict] = None
        best_call_dist = float("inf")

        for _, row in exp_chain.iterrows():
            try:
                k = float(row["exercise_price"])
                cp = str(row["call_put"]).upper()
                price = float(row.get("close", 0) or 0)
                ts_code = str(row.get("ts_code", f"{cp}-{k}"))

                if price <= 0:
                    continue

                # Compute IV
                opt_type = "C" if cp in ("C", "CALL") else "P" if cp in ("P", "PUT") else None
                if opt_type is None:
                    continue

                iv = calc_implied_vol(
                    market_price=price,
                    S=implied_forward,
                    K=k,
                    T=T,
                    r=r,
                    option_type=opt_type,
                )
                if iv is None or iv <= 0:
                    continue

                # BS delta
                d1, _ = bs_d1_d2(implied_forward, k, T, r, iv)
                if opt_type == "C":
                    delta = float(norm.cdf(d1))
                else:
                    delta = float(norm.cdf(d1) - 1)

                leg_info = {
                    "ts_code": ts_code,
                    "exercise_price": k,
                    "delta": delta,
                    "close": price,
                    "iv": iv,
                    "expire_date": expire_norm,
                }

                if opt_type == "P" and k < implied_forward:
                    dist = abs(delta - put_delta_target)
                    if dist < best_put_dist:
                        best_put_dist = dist
                        best_put = leg_info

                elif opt_type == "C" and k > implied_forward:
                    dist = abs(delta - call_delta_target)
                    if dist < best_call_dist:
                        best_call_dist = dist
                        best_call = leg_info

            except Exception:
                continue

        result = {}
        if best_put:
            result["put"] = best_put
        if best_call:
            result["call"] = best_call
        return result

    # ------------------------------------------------------------------
    # Iron Condor leg selection
    # ------------------------------------------------------------------

    def select_iron_condor_legs(
        self,
        options_chain: pd.DataFrame,
        implied_forward: float,
        expire_date: str,
        put_delta_target: float = -0.15,
        call_delta_target: float = 0.15,
        wing_width: int = 400,
        r: float = 0.02,
    ) -> Dict:
        """
        Select 4 legs for an Iron Condor:
          - Sell OTM Put  (K1, delta ~ put_delta_target)
          - Buy deeper OTM Put  (K2 = nearest strike to K1 - wing_width)
          - Sell OTM Call (K3, delta ~ call_delta_target)
          - Buy deeper OTM Call (K4 = nearest strike to K3 + wing_width)

        Returns dict with keys: sell_put, buy_put, sell_call, buy_call.
        Each value has: ts_code, exercise_price, delta, close, iv, expire_date.
        Returns empty dict if insufficient data.
        """
        # First get the sell legs via strangle selection
        strangle = self.select_strangle_legs(
            options_chain, implied_forward, expire_date,
            put_delta_target, call_delta_target, r,
        )
        if not (strangle.get("put") and strangle.get("call")):
            return {}

        sell_put = strangle["put"]
        sell_call = strangle["call"]

        # Now find buy legs at K ± wing_width (nearest available strike)
        buy_put = self._find_wing_leg(
            options_chain, expire_date, implied_forward,
            target_strike=sell_put["exercise_price"] - wing_width,
            option_type="P", r=r,
        )
        buy_call = self._find_wing_leg(
            options_chain, expire_date, implied_forward,
            target_strike=sell_call["exercise_price"] + wing_width,
            option_type="C", r=r,
        )

        if buy_put is None or buy_call is None:
            return {}

        return {
            "sell_put": sell_put,
            "buy_put": buy_put,
            "sell_call": sell_call,
            "buy_call": buy_call,
        }

    def _find_wing_leg(
        self,
        options_chain: pd.DataFrame,
        expire_date: str,
        implied_forward: float,
        target_strike: float,
        option_type: str,
        r: float = 0.02,
    ) -> Optional[Dict]:
        """Find the option closest to target_strike for the given type and expiry."""
        from models.pricing.implied_vol import calc_implied_vol, bs_d1_d2
        from scipy.stats import norm

        chain = options_chain.copy()
        if "strike_price" in chain.columns and "exercise_price" not in chain.columns:
            chain = chain.rename(columns={"strike_price": "exercise_price"})
        if "call_put" not in chain.columns and "option_type" in chain.columns:
            chain = chain.rename(columns={"option_type": "call_put"})

        required = {"exercise_price", "call_put", "close", "expire_date"}
        if not required.issubset(chain.columns):
            return None

        expire_norm = str(expire_date).replace("-", "")
        chain["expire_date"] = chain["expire_date"].astype(str).str.replace("-", "")
        exp_chain = chain[chain["expire_date"] == expire_norm]
        if exp_chain.empty:
            return None

        cp_filter = ("C", "CALL") if option_type == "C" else ("P", "PUT")
        candidates = exp_chain[exp_chain["call_put"].str.upper().isin(cp_filter)]
        if candidates.empty:
            return None

        # Estimate T
        dte_est = 30
        if "days_to_expiry" in options_chain.columns:
            dte_est = int(options_chain["days_to_expiry"].iloc[0])
        T = max(dte_est, 1) / 365.0

        best: Optional[Dict] = None
        best_dist = float("inf")

        for _, row in candidates.iterrows():
            try:
                k = float(row["exercise_price"])
                price = float(row.get("close", 0) or 0)
                if price <= 0:
                    continue

                dist = abs(k - target_strike)
                if dist < best_dist:
                    ts_code = str(row.get("ts_code", f"{option_type}-{k}"))
                    iv = calc_implied_vol(
                        market_price=price, S=implied_forward, K=k,
                        T=T, r=r, option_type=option_type,
                    )
                    if iv is None or iv <= 0:
                        iv = 0.0
                    delta = 0.0
                    if iv > 0:
                        d1, _ = bs_d1_d2(implied_forward, k, T, r, iv)
                        if option_type == "C":
                            delta = float(norm.cdf(d1))
                        else:
                            delta = float(norm.cdf(d1) - 1)

                    best_dist = dist
                    best = {
                        "ts_code": ts_code,
                        "exercise_price": k,
                        "delta": delta,
                        "close": price,
                        "iv": iv,
                        "expire_date": expire_norm,
                    }
            except Exception:
                continue

        return best
