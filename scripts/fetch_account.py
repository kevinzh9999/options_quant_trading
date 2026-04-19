"""快速抓取TQ实盘账户/持仓快照。"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from data.sources.tq_client import TqClient

client = TqClient(
    auth_account=os.getenv("TQ_ACCOUNT", ""),
    auth_password=os.getenv("TQ_PASSWORD", ""),
    broker_id=os.getenv("TQ_BROKER", ""),
    account_id=os.getenv("TQ_ACCOUNT_ID", ""),
    broker_password=os.getenv("TQ_BROKER_PASSWORD", ""),
)
client.connect()
api = client._api

a = api.get_account()
p = api.get_position()
t = api.get_trade()
api.wait_update(deadline=time.time() + 30)

print(f"权益: {a.balance:,.0f}  可用: {a.available:,.0f}  浮盈: {a.float_profit:,.0f}  保证金: {a.margin:,.0f}")
print()
for k, pos in p.items():
    if pos.pos_long > 0 or pos.pos_short > 0:
        pnl = pos.float_profit_long + pos.float_profit_short
        print(f"  {k}: 多{pos.pos_long}手 空{pos.pos_short}手 浮盈{pnl:,.0f}")
print()
n = 0
for k, tr in t.items():
    d = "买" if tr.direction == "BUY" else "卖"
    o = "开" if tr.offset == "OPEN" else "平"
    print(f"  成交: {tr.exchange_id}.{tr.instrument_id} {d}{o} {tr.volume}手 @{tr.price}")
    n += 1
if n == 0:
    print("  今日无成交")

client.disconnect()
