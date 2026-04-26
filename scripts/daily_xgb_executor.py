#!/usr/bin/env python3
"""Daily XGB executor — independent TQ-based executor for daily strategy.

Workflow:
  1. 启动后从 daily_xgb_trades 恢复 OPEN positions
  2. Poll tmp/daily_xgb_pending_*.json 每 5s
  3. 收到 PENDING 信号 → 操作员 review (60s timeout opt-in) → 限价单 entry
  4. 每日 14:55 BJ 跑一次 EOD check: TIME 到期 / SL 触发 → 次日开盘平仓
  5. 全独立于 intraday executor — 不读 signal_pending_*.json，自己 TqApi 实例

用法:
    python scripts/daily_xgb_executor.py            # live
    python scripts/daily_xgb_executor.py --dry-run  # 模拟，不下单
"""
from __future__ import annotations

import argparse
import json
import os
import select
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(Path(ROOT) / ".env")
except ImportError:
    pass

from strategies.daily_xgb.config import DailyXGBConfig
from strategies.daily_xgb import persist
from strategies.daily_xgb import position_manager as pm
from strategies.daily_xgb import risk_guard


W = 70


def _bj_now() -> datetime:
    return datetime.utcnow() + timedelta(hours=8)


def _confirm(prompt: str, timeout_s: float = 60.0) -> str:
    """Read line from stdin with timeout. Returns 'y'/'n'/'' (timeout)."""
    print(prompt, end="", flush=True)
    rlist, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if rlist:
        return sys.stdin.readline().strip().lower()
    return ""


# ─────────────────────────────────────────────────────────────────────
# TQ client wrapper
# ─────────────────────────────────────────────────────────────────────

def _connect_tq():
    """Establish TQ connection; returns (tq_client, tq_api). dry-run safe."""
    from data.sources.tq_client import TqClient
    creds = {
        "auth_account": os.getenv("TQ_ACCOUNT", ""),
        "auth_password": os.getenv("TQ_PASSWORD", ""),
        "broker_id": os.getenv("TQ_BROKER", ""),
        "account_id": os.getenv("TQ_ACCOUNT_ID", ""),
        "broker_password": os.getenv("TQ_BROKER_PASSWORD", ""),
    }
    cli = TqClient(**creds)
    cli.connect()
    return cli, cli._api


def _resolve_contract(api, underlying: str) -> str:
    """Get current main contract via TQ open_interest, fallback to calendar."""
    from utils.cffex_calendar import get_main_contract
    return get_main_contract(underlying, api=api)


def _get_quote_prices(api, contract: str):
    """Get bid1/ask1/last from TQ; returns (bid, ask, last) or None on error."""
    try:
        q = api.get_quote(contract)
        api.wait_update(deadline=time.time() + 3)
        return float(q.bid_price1), float(q.ask_price1), float(q.last_price)
    except Exception as e:
        print(f"  [WARN] get_quote failed for {contract}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────
# Order placement
# ─────────────────────────────────────────────────────────────────────

def _place_open_order(cfg: DailyXGBConfig, api, signal: Dict[str, Any],
                        contract: str, dry_run: bool = False) -> Optional[Dict[str, Any]]:
    """Place limit order to open a new position. Returns dict with fill info."""
    direction = signal["direction"]   # LONG / SHORT
    lots = int(signal.get("lots", cfg.lots_per_signal))
    sd = signal["signal_date"]

    # Risk gate
    ok, reason = risk_guard.check_can_open(cfg, lots)
    if not ok:
        print(f"  [BLOCKED] Risk gate: {reason}")
        persist.insert_executor_event(cfg, "RISK_BLOCK", details=reason)
        persist.update_signal_status(cfg, sd, "SKIPPED_RISK", reason)
        return None

    if dry_run:
        prices = _get_quote_prices(api, contract) if api else None
        last = prices[2] if prices else float(signal.get("entry_intended_open", 0))
        print(f"  [DRY] would open {direction} {contract} {lots}手 @ ~{last:.1f}")
        return {"status": "DRY_RUN", "filled_lots": 0, "filled_price": 0.0}

    # Get prices
    prices = _get_quote_prices(api, contract)
    if prices is None:
        print(f"  [ERROR] cannot get quote for {contract}")
        return None
    bid, ask, last = prices

    # Limit price (passive at our side)
    if direction == "LONG":
        tq_dir = "BUY"
        limit_price = ask  # take ask to ensure fill at open
    else:
        tq_dir = "SELL"
        limit_price = bid
    tq_offset = "OPEN"

    order_id_prefix = cfg.tq_order_prefix + "OPEN_"
    print(f"  [LIVE] Submitting: {direction} {contract} {lots}@{limit_price:.1f}")

    # Place
    order = api.insert_order(
        contract, direction=tq_dir, offset=tq_offset,
        volume=lots, limit_price=limit_price,
    )
    order_id = str(getattr(order, "order_id", ""))
    full_order_id = order_id_prefix + order_id

    # Audit
    persist.insert_order(cfg, {
        "order_id": full_order_id,
        "strategy_id": cfg.strategy_id,
        "submit_time": _bj_now().strftime("%Y-%m-%d %H:%M:%S"),
        "update_time": _bj_now().strftime("%Y-%m-%d %H:%M:%S"),
        "contract_code": contract,
        "direction": tq_dir,
        "offset": tq_offset,
        "lots": lots,
        "limit_price": limit_price,
        "filled_lots": 0,
        "filled_price": None,
        "status": "SUBMITTED",
        "cancel_reason": None,
        "trade_id": None,
        "signal_date": sd,
    })
    persist.insert_executor_event(cfg, "OPEN_SUBMITTED",
                                     order_id=full_order_id,
                                     details=f"{direction} {contract} {lots}@{limit_price:.1f}")

    # Wait fill
    deadline = time.time() + cfg.tq_open_fill_timeout_s
    while not order.is_dead and time.time() < deadline:
        api.wait_update()

    filled = order.volume_orign - order.volume_left
    trade_price = float(getattr(order, "trade_price", limit_price) or limit_price)

    # Cancel if not filled
    if order.volume_left > 0 and not order.is_dead:
        api.cancel_order(order)
        cancel_dl = time.time() + 5
        while not order.is_dead and time.time() < cancel_dl:
            api.wait_update()
        filled = order.volume_orign - order.volume_left

    # Determine status
    if filled == 0:
        # Try aggressive price
        prices2 = _get_quote_prices(api, contract)
        if prices2 is not None:
            bid2, ask2, _ = prices2
            aggr = (ask2 + cfg.order_aggressive_offset_pts
                     if direction == "LONG"
                     else bid2 - cfg.order_aggressive_offset_pts)
            print(f"  Limit not filled, retry aggressive @ {aggr:.1f}")
            order2 = api.insert_order(contract, direction=tq_dir, offset=tq_offset,
                                        volume=lots, limit_price=aggr)
            dl2 = time.time() + 15
            while not order2.is_dead and time.time() < dl2:
                api.wait_update()
            if not order2.is_dead:
                api.cancel_order(order2)
            filled = order2.volume_orign - order2.volume_left
            trade_price = float(getattr(order2, "trade_price", aggr) or aggr)

    if filled == lots:
        status = "FILLED"
    elif filled > 0:
        status = "PARTIAL"
    else:
        status = "TIMEOUT"

    persist.update_order(cfg, full_order_id, status=status,
                            filled_lots=filled, filled_price=trade_price,
                            cancel_reason=None if status == "FILLED" else "TIMEOUT")
    persist.insert_executor_event(cfg, "OPEN_FILLED" if filled > 0 else "OPEN_FAILED",
                                     order_id=full_order_id,
                                     details=f"filled {filled}/{lots} @ {trade_price:.1f}")
    return {
        "status": status, "filled_lots": filled,
        "filled_price": trade_price, "order_id": full_order_id,
    }


def _place_close_order(cfg: DailyXGBConfig, api,
                         pos: pm.OpenPosition, dry_run: bool = False) -> Optional[Dict[str, Any]]:
    """Close out an existing position. Returns fill info."""
    contract = pos.contract_code
    lots = pos.entry_lots
    direction = pos.direction

    if dry_run:
        prices = _get_quote_prices(api, contract) if api else None
        last = prices[2] if prices else pos.entry_price
        print(f"  [DRY] would close {direction} {contract} {lots}手 @ ~{last:.1f}")
        return {"status": "DRY_RUN", "filled_lots": 0, "filled_price": 0.0}

    prices = _get_quote_prices(api, contract)
    if prices is None:
        return None
    bid, ask, last = prices

    # CFFEX 平今手续费 10x → 用锁仓 (反向 OPEN) 代替 CLOSE
    # 但是 daily strategy 持仓多日，平昨手续费正常 → 用 CLOSE
    # entry_date < today: 平昨 → 用 CLOSE
    today = _bj_now().strftime("%Y%m%d")
    if pos.entry_date == today:
        # 同日：用锁仓（虽然不太可能，因为 daily 是次日开仓）
        if direction == "LONG":
            tq_dir, tq_offset = "SELL", "OPEN"  # 反向 OPEN
        else:
            tq_dir, tq_offset = "BUY", "OPEN"
        limit_price = (bid - cfg.order_aggressive_offset_pts
                        if direction == "LONG" else ask + cfg.order_aggressive_offset_pts)
    else:
        # 平昨
        if direction == "LONG":
            tq_dir, tq_offset = "SELL", "CLOSE"
            limit_price = bid
        else:
            tq_dir, tq_offset = "BUY", "CLOSE"
            limit_price = ask

    order_id_prefix = cfg.tq_order_prefix + "CLOSE_"
    print(f"  [LIVE] Closing: {direction} {contract} {lots}@{limit_price:.1f} ({tq_offset})")

    order = api.insert_order(
        contract, direction=tq_dir, offset=tq_offset,
        volume=lots, limit_price=limit_price,
    )
    order_id = str(getattr(order, "order_id", ""))
    full_order_id = order_id_prefix + order_id

    persist.insert_order(cfg, {
        "order_id": full_order_id,
        "strategy_id": cfg.strategy_id,
        "submit_time": _bj_now().strftime("%Y-%m-%d %H:%M:%S"),
        "update_time": _bj_now().strftime("%Y-%m-%d %H:%M:%S"),
        "contract_code": contract,
        "direction": tq_dir,
        "offset": tq_offset,
        "lots": lots,
        "limit_price": limit_price,
        "filled_lots": 0,
        "filled_price": None,
        "status": "SUBMITTED",
        "cancel_reason": None,
        "trade_id": pos.trade_id,
        "signal_date": pos.signal_date,
    })
    persist.insert_executor_event(cfg, "CLOSE_SUBMITTED",
                                     order_id=full_order_id,
                                     details=f"trade_id={pos.trade_id}")

    deadline = time.time() + cfg.tq_close_fill_timeout_s
    while not order.is_dead and time.time() < deadline:
        api.wait_update()
    filled = order.volume_orign - order.volume_left
    trade_price = float(getattr(order, "trade_price", limit_price) or limit_price)

    if order.volume_left > 0 and not order.is_dead:
        api.cancel_order(order)
        cdl = time.time() + 5
        while not order.is_dead and time.time() < cdl:
            api.wait_update()
        filled = order.volume_orign - order.volume_left

    if filled == 0:
        # Aggressive retry — close is urgent
        prices2 = _get_quote_prices(api, contract)
        if prices2 is not None:
            bid2, ask2, _ = prices2
            aggr = (bid2 - cfg.order_aggressive_offset_pts
                     if direction == "LONG"
                     else ask2 + cfg.order_aggressive_offset_pts)
            print(f"  Close retry aggressive @ {aggr:.1f}")
            order2 = api.insert_order(contract, direction=tq_dir, offset=tq_offset,
                                        volume=lots, limit_price=aggr)
            dl2 = time.time() + 20
            while not order2.is_dead and time.time() < dl2:
                api.wait_update()
            if not order2.is_dead:
                api.cancel_order(order2)
            filled = order2.volume_orign - order2.volume_left
            trade_price = float(getattr(order2, "trade_price", aggr) or aggr)

    status = "FILLED" if filled == lots else ("PARTIAL" if filled > 0 else "TIMEOUT")
    persist.update_order(cfg, full_order_id, status=status,
                            filled_lots=filled, filled_price=trade_price,
                            cancel_reason=None if status == "FILLED" else "TIMEOUT")
    persist.insert_executor_event(cfg, "CLOSE_FILLED" if filled > 0 else "CLOSE_FAILED",
                                     order_id=full_order_id,
                                     details=f"filled {filled}/{lots} @ {trade_price:.1f}")

    return {"status": status, "filled_lots": filled, "filled_price": trade_price}


# ─────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────

def _process_pending_signal(cfg: DailyXGBConfig, api, signal: Dict[str, Any],
                              dry_run: bool = False):
    """Review + place open order."""
    sd = signal["signal_date"]
    direction = signal["direction"]
    print(f"\n{'═' * W}")
    print(f" Daily XGB Pending Signal | {_bj_now().strftime('%Y-%m-%d %H:%M:%S')} BJ")
    print(f"{'═' * W}")
    print(f"  Signal date:    {sd}")
    print(f"  Direction:      {direction}")
    print(f"  Underlying:     {signal['underlying']}")
    print(f"  Lots:           {signal['lots']}")
    print(f"  Hold:           {signal['hold_days']} 个交易日 ({signal['enhancement_type']})")
    print(f"  SL:             {signal['sl_pct']*100:.2f}% (ATR×{signal['atr_k']})")
    print(f"  Pred:           {signal['pred']:+.4f}")
    print(f"  Entry intended: T+1 open (next session)")
    if "regime" in signal:
        r = signal["regime"]
        print(f"  Regime: c/sma60={r.get('close_sma60'):.4f}  "
              f"c/sma200={r.get('close_sma200'):.4f}  "
              f"vol_regime={r.get('vol_regime'):.3f}")

    risk_status = risk_guard.status_report(cfg)
    print(f"\n{risk_status}")

    resp = _confirm(f"\n  Confirm open? [y/n] (60s timeout=skip): ", 60.0)
    if resp != "y":
        reason = "operator declined" if resp == "n" else "timeout"
        print(f"  → SKIPPED ({reason})")
        persist.update_signal_status(cfg, sd, "SKIPPED_OPERATOR", reason)
        persist.remove_pending_signal(cfg, sd)
        return

    # Resolve contract
    contract = _resolve_contract(api, signal["underlying"]) if api else f"CFFEX.{signal['underlying']}2606"
    print(f"  Contract: {contract}")

    # Place order
    fill = _place_open_order(cfg, api, signal, contract, dry_run=dry_run)
    if fill is None or fill["filled_lots"] == 0:
        print(f"  → Open failed, signal kept PENDING")
        return

    # Register entry in DB
    today = _bj_now().strftime("%Y%m%d")
    atr_pct = signal.get("regime", {}).get("atr20_pct", 0.015)
    trade_id = pm.register_entry(
        cfg,
        signal_date=sd,
        entry_date=today,
        direction=direction,
        contract_code=contract,
        entry_price=fill["filled_price"],
        lots=fill["filled_lots"],
        sl_pct=signal["sl_pct"],
        atr_k=signal["atr_k"],
        hold_days=signal["hold_days"],
        enhancement_type=signal["enhancement_type"],
        atr20=atr_pct or 0.015,
    )
    print(f"  ✓ Entered trade_id={trade_id}: {fill['filled_lots']} 手 @ {fill['filled_price']:.1f}")
    persist.update_signal_status(cfg, sd, "EXECUTED", f"trade_id={trade_id}")
    persist.remove_pending_signal(cfg, sd)
    pm.export_to_json(cfg)


def _process_eod_exits(cfg: DailyXGBConfig, api, dry_run: bool = False):
    """Run EOD: check for time/SL exits and close them."""
    today = _bj_now().strftime("%Y%m%d")
    # Get today's IM close (use last_price as proxy if intraday)
    if api:
        contract_main = _resolve_contract(api, cfg.underlying)
        prices = _get_quote_prices(api, contract_main)
        if prices is None:
            print(f"[EOD] cannot get close price for {contract_main}")
            return
        today_close = prices[2]
    else:
        today_close = 8000.0  # dry-run fallback

    exits = pm.check_exits(cfg, today, today_close)
    if not exits:
        print(f"[EOD] {today} close={today_close:.1f}: no positions to close")
        return

    print(f"\n[EOD] {today} close={today_close:.1f}: {len(exits)} positions to close")
    for pos, reason in exits:
        print(f"  - trade_id={pos.trade_id}  {pos.direction} {pos.contract_code} "
              f"{pos.entry_lots}@{pos.entry_price:.1f}  reason={reason}")
        if not dry_run and api:
            fill = _place_close_order(cfg, api, pos, dry_run=False)
            if fill and fill["filled_lots"] > 0:
                pm.apply_exit(cfg, pos.trade_id, today, fill["filled_price"], reason)
        else:
            # Dry-run: simulate close at today_close
            pm.apply_exit(cfg, pos.trade_id, today, today_close, reason)
    pm.export_to_json(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--eod-only", action="store_true",
                     help="Only run EOD exit check, then exit")
    ap.add_argument("--signal-only", action="store_true",
                     help="Only process pending signals once, then exit")
    args = ap.parse_args()

    cfg = DailyXGBConfig()
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)

    print(f"{'═' * W}")
    print(f" Daily XGB Executor | {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f" Mode: 保守 (1 lot/signal, cap=10)")
    print(f" Account: {cfg.account_equity:,.0f}")
    print(f"{'═' * W}")

    persist.insert_executor_event(cfg, "START",
                                     details=f"dry_run={args.dry_run}, "
                                             f"eod_only={args.eod_only}, "
                                             f"signal_only={args.signal_only}")

    # Risk status report
    print(risk_guard.status_report(cfg))
    print()

    # Existing positions
    open_pos = pm.get_open_positions(cfg)
    print(f"Existing open positions: {len(open_pos)}")
    for p in open_pos:
        print(f"  - id={p.trade_id} {p.direction} {p.contract_code} {p.entry_lots}@{p.entry_price:.1f} "
              f"entry={p.entry_date} planned_exit={p.planned_exit_date} sl={p.sl_price:.1f}")

    # Connect TQ
    tq_client, api = None, None
    if not args.dry_run:
        try:
            tq_client, api = _connect_tq()
            print(f" TQ connected.")
        except Exception as e:
            print(f" [ERROR] TQ connection failed: {e}")
            persist.insert_executor_event(cfg, "ERROR", details=f"TQ connect: {e}")
            sys.exit(1)

    # EOD only mode
    if args.eod_only:
        _process_eod_exits(cfg, api, dry_run=args.dry_run)
        sys.exit(0)

    # Signal-only mode: process pending once
    if args.signal_only:
        signals = persist.read_pending_signals(cfg)
        print(f"\n{len(signals)} pending signals")
        for sig in signals:
            _process_pending_signal(cfg, api, sig, dry_run=args.dry_run)
        sys.exit(0)

    # Continuous mode: poll signals + run EOD at 14:55
    print(f"\nPolling tmp/daily_xgb_pending_*.json every 5s. Ctrl+C to stop.")
    last_eod_run = None
    try:
        while True:
            signals = persist.read_pending_signals(cfg)
            for sig in signals:
                _process_pending_signal(cfg, api, sig, dry_run=args.dry_run)

            # EOD at 14:55 BJ
            now = _bj_now()
            today_eod_dt = now.replace(hour=14, minute=55, second=0, microsecond=0)
            if (now >= today_eod_dt
                    and (last_eod_run is None or last_eod_run.date() != now.date())):
                print(f"\n[{now.strftime('%H:%M:%S')}] Running EOD check...")
                _process_eod_exits(cfg, api, dry_run=args.dry_run)
                last_eod_run = now

            time.sleep(5)
    except KeyboardInterrupt:
        print(f"\nShutting down.")
        persist.insert_executor_event(cfg, "STOP")


if __name__ == "__main__":
    main()
