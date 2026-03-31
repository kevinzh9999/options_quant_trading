"""
data_feed.py
------------
回测数据馈送器。从数据库预加载历史数据，按日期逐步推送给策略，严格防止前视偏差。
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Pattern for index codes like 000852.SH, 000300.SH
_INDEX_CODE_RE = re.compile(r"^\d{6}\.\w+$")

# Option underlying prefixes that need options data
_OPTION_UNDERLYINGS = ("MO", "IO")


class DataFeed:
    """
    Backtest data feed. Preloads all required data from DB, then serves it
    date-by-date to strategies with no look-ahead bias.

    Parameters
    ----------
    db_manager : DBManager
        Database manager instance.
    start_date : str
        Backtest start date (YYYYMMDD).
    end_date : str
        Backtest end date (YYYYMMDD).
    symbols : list[str]
        Futures/index symbols, e.g. ["IM.CFX", "000852.SH"].
    """

    def __init__(
        self,
        db_manager,
        start_date: str,
        end_date: str,
        symbols: List[str],
    ) -> None:
        self.db = db_manager
        self.start_date = start_date
        self.end_date = end_date
        self.symbols = symbols

        self._futures_data: Dict[str, pd.DataFrame] = {}   # symbol -> full sorted df
        self._index_data: Dict[str, pd.DataFrame] = {}     # index code -> full sorted df
        self._options_daily: pd.DataFrame = pd.DataFrame()
        self._options_contracts: pd.DataFrame = pd.DataFrame()
        self._trading_dates: List[str] = []
        self._loaded = False

    # ------------------------------------------------------------------
    # Preload
    # ------------------------------------------------------------------

    def preload(self) -> None:
        """Preload all data needed for the backtest period into memory."""
        if self._loaded:
            return
        # Pad start by 300 calendar days for lookback
        start_dt = datetime.strptime(self.start_date, "%Y%m%d")
        padded_start = (start_dt - timedelta(days=300)).strftime("%Y%m%d")

        logger.info(
            "DataFeed.preload: %s ~ %s  (padded from %s)",
            padded_start, self.end_date, self.start_date
        )

        # Separate futures/spot, index, and option-underlying symbols
        futures_syms = [
            s for s in self.symbols
            if not _INDEX_CODE_RE.match(s)
            and not any(s.startswith(prefix) for prefix in _OPTION_UNDERLYINGS)
        ]
        option_syms = [
            s for s in self.symbols
            if any(s.startswith(prefix) for prefix in _OPTION_UNDERLYINGS)
        ]
        index_syms = [s for s in self.symbols if _INDEX_CODE_RE.match(s)]

        # Load futures daily (skip option underlying symbols like MO.CFX — not in futures_daily)
        for sym in futures_syms:
            try:
                df = self.db.query_df(
                    "SELECT * FROM futures_daily WHERE ts_code = ? "
                    "AND trade_date >= ? AND trade_date <= ? ORDER BY trade_date ASC",
                    (sym, padded_start, self.end_date),
                )
                if df.empty:
                    logger.warning("DataFeed: no futures data for %s", sym)
                else:
                    df = df.sort_values("trade_date").reset_index(drop=True)
                    self._futures_data[sym] = df
                    logger.info("Loaded %d rows for %s", len(df), sym)
            except Exception as exc:
                logger.error("DataFeed: failed loading %s: %s", sym, exc)

        # Load index daily
        for sym in index_syms:
            try:
                df = self.db.query_df(
                    "SELECT * FROM index_daily WHERE ts_code = ? "
                    "AND trade_date >= ? AND trade_date <= ? ORDER BY trade_date ASC",
                    (sym, padded_start, self.end_date),
                )
                if df.empty:
                    logger.warning("DataFeed: no index data for %s", sym)
                else:
                    df = df.sort_values("trade_date").reset_index(drop=True)
                    self._index_data[sym] = df
                    logger.info("Loaded %d index rows for %s", len(df), sym)
            except Exception as exc:
                logger.error("DataFeed: failed loading index %s: %s", sym, exc)

        # Build trading dates from the union of all futures data
        all_dates: set = set()
        for df in self._futures_data.values():
            if "trade_date" in df.columns:
                all_dates.update(df["trade_date"].tolist())

        self._trading_dates = sorted(
            d for d in all_dates
            if self.start_date <= d <= self.end_date
        )
        logger.info("DataFeed: %d trading dates in [%s, %s]",
                    len(self._trading_dates), self.start_date, self.end_date)

        # Load options data if any symbol has option underlying prefix
        needs_options = len(option_syms) > 0
        if needs_options:
            self._load_options(padded_start)

        self._loaded = True

    def _load_options(self, padded_start: str) -> None:
        """Load options_daily and options_contracts tables."""
        try:
            self._options_daily = self.db.query_df(
                "SELECT * FROM options_daily WHERE trade_date >= ? AND trade_date <= ?",
                (padded_start, self.end_date),
            )
            logger.info("Loaded %d options_daily rows", len(self._options_daily))
        except Exception as exc:
            logger.warning("DataFeed: options_daily load failed: %s", exc)
            self._options_daily = pd.DataFrame()

        try:
            self._options_contracts = self.db.query_df(
                "SELECT * FROM options_contracts",
                None,
            )
            logger.info("Loaded %d options_contracts rows", len(self._options_contracts))
        except Exception as exc:
            logger.warning("DataFeed: options_contracts load failed: %s", exc)
            # Try alternate table name
            try:
                self._options_contracts = self.db.query_df(
                    "SELECT * FROM options_basic",
                    None,
                )
            except Exception:
                self._options_contracts = pd.DataFrame()

    # ------------------------------------------------------------------
    # Trading dates
    # ------------------------------------------------------------------

    def get_trading_dates(self) -> List[str]:
        """Return sorted list of trading dates in [start_date, end_date]."""
        if not self._loaded:
            self.preload()
        return list(self._trading_dates)

    # ------------------------------------------------------------------
    # Single bar lookup
    # ------------------------------------------------------------------

    def get_daily_bar(self, symbol: str, trade_date: str) -> Optional[pd.Series]:
        """
        Get OHLCV bar for symbol on trade_date.

        Returns None if not found.
        """
        df = self._futures_data.get(symbol)
        if df is None:
            df = self._index_data.get(symbol)
        if df is None or df.empty:
            return None

        rows = df[df["trade_date"] == trade_date]
        if rows.empty:
            return None
        return rows.iloc[0]

    # ------------------------------------------------------------------
    # History slice (no look-ahead)
    # ------------------------------------------------------------------

    def get_history(
        self,
        symbol: str,
        trade_date: str,
        lookback: int = 60,
    ) -> pd.DataFrame:
        """
        Return up to `lookback` rows of daily bars UP TO AND INCLUDING trade_date.

        No data after trade_date is included (no look-ahead bias).

        Parameters
        ----------
        symbol : str
            Symbol to query.
        trade_date : str
            Current trading date (YYYYMMDD).
        lookback : int
            Maximum number of bars to return.

        Returns
        -------
        pd.DataFrame
            Rows sorted ascending by trade_date, at most `lookback` rows.
        """
        df = self._futures_data.get(symbol)
        if df is None:
            df = self._index_data.get(symbol)
        if df is None or df.empty:
            return pd.DataFrame()

        mask = df["trade_date"] <= trade_date
        sliced = df[mask]
        if sliced.empty:
            return pd.DataFrame()

        return sliced.tail(lookback).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Options chain
    # ------------------------------------------------------------------

    def get_options_chain_on_date(
        self,
        underlying: str,
        trade_date: str,
    ) -> Optional[pd.DataFrame]:
        """
        Return merged options daily + contracts for the given underlying on trade_date.

        Parameters
        ----------
        underlying : str
            Option underlying prefix, e.g. "MO" or "IO".
        trade_date : str
            Trading date (YYYYMMDD).

        Returns
        -------
        pd.DataFrame or None
            Merged DataFrame with options data, or None if unavailable.
        """
        if self._options_daily.empty:
            return None

        daily = self._options_daily[
            self._options_daily["trade_date"] == trade_date
        ].copy()

        if daily.empty:
            return None

        # Filter by underlying: ts_code starts with the underlying prefix
        if "ts_code" in daily.columns:
            daily = daily[daily["ts_code"].str.startswith(underlying)]

        if daily.empty:
            return None

        # Merge with contracts metadata if available
        if not self._options_contracts.empty and "ts_code" in self._options_contracts.columns:
            try:
                merged = daily.merge(
                    self._options_contracts,
                    on="ts_code",
                    how="left",
                    suffixes=("", "_contract"),
                )
                # options_daily often stores NULL for exercise_price / call_put / expire_date.
                # The actual values come from options_contracts and land in the _contract columns.
                # Promote them back to the canonical column names so downstream code can use them.
                for col in ("exercise_price", "call_put", "expire_date"):
                    contract_col = col + "_contract"
                    if contract_col in merged.columns:
                        null_mask = merged[col].isna()
                        if null_mask.any():
                            merged.loc[null_mask, col] = merged.loc[null_mask, contract_col]
                return merged
            except Exception as exc:
                logger.warning("Options merge failed: %s", exc)
                return daily

        return daily

    # ------------------------------------------------------------------
    # Index price
    # ------------------------------------------------------------------

    def get_index_close(self, index_code: str, trade_date: str) -> Optional[float]:
        """
        Return close price for an index on trade_date.

        Parameters
        ----------
        index_code : str
            Index code, e.g. "000852.SH".
        trade_date : str
            Trading date (YYYYMMDD).

        Returns
        -------
        float or None
        """
        df = self._index_data.get(index_code)
        if df is None or df.empty:
            return None

        rows = df[df["trade_date"] == trade_date]
        if rows.empty:
            return None

        row = rows.iloc[0]
        close_val = row.get("close")
        if close_val is None or (isinstance(close_val, float) and pd.isna(close_val)):
            return None
        return float(close_val)
