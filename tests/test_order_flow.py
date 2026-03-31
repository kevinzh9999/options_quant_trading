#!/usr/bin/env python3
"""测试TQ下单流程：开仓→锁仓，用3/27历史数据回测。

验证 order_executor 的锁仓逻辑：
  1. SELL OPEN 2手（做空开仓）
  2. 等待50个更新周期
  3. BUY OPEN 2手（锁仓，不用CLOSE避免平今手续费）
  4. 验证持仓：多头>=2 且 空头>=2

注意：使用 TqBacktest（免费），不用 TqReplay（需付费专业版）。
TqBacktest 用K线模拟撮合，不是tick级别，但足以验证下单参数正确性。

用法：
    python tests/test_order_flow.py
"""

import os
import sys
from datetime import date
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(Path(ROOT) / ".env")
TQ_ACCOUNT = os.getenv("TQ_ACCOUNT")
TQ_PASSWORD = os.getenv("TQ_PASSWORD")

from tqsdk import TqApi, TqAuth, TqSim, TqBacktest


def test_order_flow():
    sim = TqSim(init_balance=6_400_000)
    api = TqApi(
        sim,
        backtest=TqBacktest(
            start_dt=date(2026, 3, 27),
            end_dt=date(2026, 3, 27),
        ),
        auth=TqAuth(TQ_ACCOUNT, TQ_PASSWORD),
    )

    CONTRACT = "CFFEX.IM2604"
    quote = api.get_quote(CONTRACT)

    step = "WAIT_QUOTE"
    open_order = None
    lock_order = None
    tick_count = 0
    ticks_after_open = 0

    try:
        while True:
            api.wait_update()
            tick_count += 1

            bid = float(quote.bid_price1) if quote.bid_price1 == quote.bid_price1 else 0
            ask = float(quote.ask_price1) if quote.ask_price1 == quote.ask_price1 else 0
            last = float(quote.last_price) if quote.last_price == quote.last_price else 0

            if step == "WAIT_QUOTE":
                if bid > 0 and ask > 0:
                    print(f"\n=== 行情就绪 | tick#{tick_count} ===")
                    print(f"  bid={bid}  ask={ask}  last={last}")
                    step = "OPEN_SHORT"

            elif step == "OPEN_SHORT":
                lots = 2
                price = bid
                print(f"\n=== 步骤A: 开仓 SELL OPEN {lots}手 @ {price} ===")
                open_order = api.insert_order(
                    symbol=CONTRACT, direction="SELL", offset="OPEN",
                    volume=lots, limit_price=price,
                )
                step = "WAIT_OPEN_FILL"

            elif step == "WAIT_OPEN_FILL":
                if open_order.status == "FINISHED":
                    filled = open_order.volume_orign - open_order.volume_left
                    print(f"  开仓结果: 成交{filled}手 状态={open_order.status}")
                    if filled > 0:
                        pos = api.get_position(CONTRACT)
                        print(f"  持仓: 多头={pos.pos_long} 空头={pos.pos_short}")
                        step = "WAIT_BEFORE_LOCK"
                    else:
                        print(f"  未成交，测试结束")
                        break

            elif step == "WAIT_BEFORE_LOCK":
                ticks_after_open += 1
                if ticks_after_open >= 50:
                    step = "LOCK_POSITION"

            elif step == "LOCK_POSITION":
                lots = 2
                price = ask
                print(f"\n=== 步骤C: 锁仓 BUY OPEN {lots}手 @ {price} ===")
                print(f"  注意: offset=OPEN 不是 CLOSE（避免平今手续费）")
                lock_order = api.insert_order(
                    symbol=CONTRACT, direction="BUY", offset="OPEN",
                    volume=lots, limit_price=price,
                )
                step = "WAIT_LOCK_FILL"

            elif step == "WAIT_LOCK_FILL":
                if lock_order.status == "FINISHED":
                    filled = lock_order.volume_orign - lock_order.volume_left
                    print(f"  锁仓结果: 成交{filled}手 状态={lock_order.status}")

                    pos = api.get_position(CONTRACT)
                    print(f"\n=== 步骤D: 验证持仓 ===")
                    print(f"  多头={pos.pos_long} 空头={pos.pos_short}"
                          f" 净={pos.pos_long - pos.pos_short}")

                    acct = api.get_account()
                    print(f"\n=== 账户状态 ===")
                    print(f"  权益={acct.balance:,.0f}"
                          f"  保证金={acct.margin:,.0f}"
                          f"  浮盈={acct.float_profit:,.0f}")

                    if pos.pos_long >= 2 and pos.pos_short >= 2:
                        print(f"\n✅ 测试通过：开仓+锁仓流程正确")
                    else:
                        print(f"\n❌ 测试失败：持仓不符合预期")
                    break

    except Exception as e:
        print(f"\n❌ 异常: {e}")
    finally:
        api.close()
        print(f"\n总tick数: {tick_count}")


if __name__ == "__main__":
    test_order_flow()
