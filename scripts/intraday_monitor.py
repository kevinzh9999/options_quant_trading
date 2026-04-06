"""
intraday_monitor.py
-------------------
盘中实时告警监控。通过天勤订阅 IM 主力合约行情 + 账户/持仓数据，
在触发告警条件时打印彩色告警、写入 logs/alerts.log，并可扩展通知渠道。

告警条件：
  a. 标的大幅波动  日内涨跌幅 > 1.5% → 黄色  /  > 2.5% → 红色
  b. 保证金占用    > 40% → 黄色  /  > 50% → 红色
  c. 浮动亏损      > 账户权益 2% → 黄色  /  > 3% → 红色
  d. 临近到期      持仓中有合约剩余 < 7 天 → 每日开盘提醒一次

周期任务：
  - 每 60 秒  检查告警
  - 每 5 分钟 写入 tq_snapshots 快照

用法：
  python scripts/intraday_monitor.py
  python scripts/intraday_monitor.py --background   # nohup 后台运行

安全退出：Ctrl+C 或 SIGTERM，确保 TQ 连接正常关闭。
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# 路径设置
# ---------------------------------------------------------------------------

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(Path(ROOT) / ".env")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

DB_PATH = os.path.join(ROOT, "data", "storage", "trading.db")
LOGS_DIR = Path(ROOT) / "logs"
ALERTS_LOG = LOGS_DIR / "alerts.log"

# 天勤 IM 主力连续合约
IM_MAIN = "KQ.m@CFFEX.IM"

# 默认快照合约（盘中每5分钟抓取）
SNAPSHOT_SYMBOLS: list[str] = [
    "KQ.m@CFFEX.IM",
    "CFFEX.IM2604",
    "CFFEX.IM2606",
]

# 检查/快照周期（秒）
CHECK_INTERVAL    = 60   # 每分钟检查一次告警
SNAPSHOT_INTERVAL = 300  # 每5分钟快照一次

# 告警阈值
PRICE_YELLOW  = 0.015   # 日内涨跌幅 ≥ 1.5%
PRICE_RED     = 0.025   # 日内涨跌幅 ≥ 2.5%
MARGIN_YELLOW = 0.40    # 保证金占用率 ≥ 40%
MARGIN_RED    = 0.50    # 保证金占用率 ≥ 50%
LOSS_YELLOW   = 0.02    # 浮亏/权益 ≥ 2%
LOSS_RED      = 0.03    # 浮亏/权益 ≥ 3%
EXPIRY_WARN_DAYS = 7    # 剩余到期天数 < 7 天

# ---------------------------------------------------------------------------
# 日志：主日志 + 告警日志（写文件）
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

LOGS_DIR.mkdir(parents=True, exist_ok=True)

_alert_handler = logging.FileHandler(ALERTS_LOG, encoding="utf-8")
_alert_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
alert_logger = logging.getLogger("alerts")
alert_logger.setLevel(logging.INFO)
alert_logger.addHandler(_alert_handler)
alert_logger.propagate = False   # 不重复输出到 root logger

# ---------------------------------------------------------------------------
# ANSI 颜色
# ---------------------------------------------------------------------------

class _C:
    YELLOW = "\033[33m"
    RED    = "\033[31m"
    GREEN  = "\033[32m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _cprint(msg: str, level: str = "info") -> None:
    """带时间戳和颜色的终端打印。level: 'yellow' / 'red' / 'green' / 'info'"""
    color = {
        "yellow": _C.YELLOW + _C.BOLD,
        "red":    _C.RED    + _C.BOLD,
        "green":  _C.GREEN,
    }.get(level, "")
    print(f"{color}[{_ts()}] {msg}{_C.RESET}", flush=True)


def _sound_alert() -> None:
    """红色告警时在 macOS 发出声音（其他平台静默跳过）。"""
    try:
        subprocess.Popen(
            ["afplay", "/System/Library/Sounds/Ping.aiff"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass  # 非 macOS 平台跳过


# ---------------------------------------------------------------------------
# 通知渠道（可扩展接口）
# ---------------------------------------------------------------------------

class Notifier:
    """
    告警通知基类。子类化并覆盖 send() 以接入微信/邮件等渠道。

    示例（微信企业号）：
        class WechatNotifier(Notifier):
            def send(self, title, body, level="info"):
                # requests.post(WECHAT_WEBHOOK, json={"msgtype": "text", ...})
                ...
    """

    def send(self, title: str, body: str, level: str = "info") -> None:
        """
        Parameters
        ----------
        title : str
            告警标题，如 "红色告警 – IM 涨幅超 2.5%"
        body : str
            告警详情
        level : str
            'yellow' / 'red' / 'info'
        """
        pass  # 默认不做任何事


# ---------------------------------------------------------------------------
# 告警状态（防重复告警）
# ---------------------------------------------------------------------------

@dataclass
class AlertState:
    """记录各告警维度的当前等级，只在等级上升时触发告警。"""

    price_level:          int = 0   # 0=无  1=黄  2=红
    margin_level:         int = 0
    loss_level:           int = 0
    expiry_alerted_date:  str = ""  # 当日已提醒时记录日期（YYYY-MM-DD）

    def reset_daily(self) -> None:
        """交易日结束后重置，允许次日重新触发。"""
        self.price_level  = 0
        self.margin_level = 0
        self.loss_level   = 0
        # expiry_alerted_date 保留，防止同一自然日重复


# ---------------------------------------------------------------------------
# 凭证 & 连接辅助
# ---------------------------------------------------------------------------

def _tq_credentials() -> dict[str, str]:
    return {
        "auth_account":    os.getenv("TQ_ACCOUNT", ""),
        "auth_password":   os.getenv("TQ_PASSWORD", ""),
        "broker_id":       os.getenv("TQ_BROKER", ""),
        "account_id":      os.getenv("TQ_ACCOUNT_ID", ""),
        "broker_password": os.getenv("TQ_BROKER_PASSWORD", ""),
    }


def _has_tq_auth() -> bool:
    c = _tq_credentials()
    return bool(c["auth_account"] and c["auth_password"])


def _has_tq_broker() -> bool:
    c = _tq_credentials()
    return bool(
        c["auth_account"] and c["auth_password"]
        and c["broker_id"] and c["account_id"] and c["broker_password"]
    )


def _open_db():
    from data.storage.db_manager import get_db
    db = get_db()
    db.initialize_tables()
    return db


# ---------------------------------------------------------------------------
# 告警检查逻辑
# ---------------------------------------------------------------------------

def _check_price(
    quote,
    state: AlertState,
    notifier: Notifier,
) -> None:
    """a. 标的大幅波动告警（日内涨跌幅 vs 昨结算价）。"""
    try:
        last = float(quote.last_price or 0)
        pre  = float(quote.pre_settlement or 0)
        if last <= 0 or pre <= 0:
            return

        chg = (last - pre) / pre
        abs_chg = abs(chg)
        direction = "上涨" if chg > 0 else "下跌"
        pct_str = f"{chg*100:+.2f}%"

        if abs_chg >= PRICE_RED and state.price_level < 2:
            msg = f"🔴 [红色] IM 日内{direction} {pct_str}  last={last:.2f}  pre={pre:.2f}"
            _cprint(msg, "red")
            alert_logger.info("[RED-PRICE] " + msg)
            notifier.send(f"红色告警 – IM 日内{direction}", msg, level="red")
            _sound_alert()
            state.price_level = 2

        elif abs_chg >= PRICE_YELLOW and state.price_level < 1:
            msg = f"🟡 [黄色] IM 日内{direction} {pct_str}  last={last:.2f}  pre={pre:.2f}"
            _cprint(msg, "yellow")
            alert_logger.info("[YLW-PRICE] " + msg)
            notifier.send(f"黄色告警 – IM 日内{direction}", msg, level="yellow")
            state.price_level = 1

        # 告警消除：回落到阈值以下时重置，允许再次触发
        elif abs_chg < PRICE_YELLOW and state.price_level > 0:
            _cprint(f"  IM 涨跌幅回落至 {pct_str}，价格告警已清除", "green")
            state.price_level = 0

    except Exception as e:
        logger.debug("价格告警检查异常: %s", e)


def _check_margin(
    account_ref,
    state: AlertState,
    notifier: Notifier,
) -> None:
    """b. 保证金占用告警。"""
    try:
        balance = float(account_ref.balance or 0)
        margin  = float(account_ref.margin  or 0)
        if balance <= 0:
            return

        ratio = margin / balance

        if ratio >= MARGIN_RED and state.margin_level < 2:
            msg = (f"🔴 [红色] 保证金占用 {ratio*100:.1f}%"
                   f"  margin={margin:,.0f}  balance={balance:,.0f}")
            _cprint(msg, "red")
            alert_logger.info("[RED-MARGIN] " + msg)
            notifier.send("红色告警 – 保证金超限", msg, level="red")
            _sound_alert()
            state.margin_level = 2

        elif ratio >= MARGIN_YELLOW and state.margin_level < 1:
            msg = (f"🟡 [黄色] 保证金占用 {ratio*100:.1f}%"
                   f"  margin={margin:,.0f}  balance={balance:,.0f}")
            _cprint(msg, "yellow")
            alert_logger.info("[YLW-MARGIN] " + msg)
            notifier.send("黄色告警 – 保证金偏高", msg, level="yellow")
            state.margin_level = 1

        elif ratio < MARGIN_YELLOW and state.margin_level > 0:
            _cprint(f"  保证金占用回落至 {ratio*100:.1f}%，告警已清除", "green")
            state.margin_level = 0

    except Exception as e:
        logger.debug("保证金告警检查异常: %s", e)


def _check_float_loss(
    account_ref,
    state: AlertState,
    notifier: Notifier,
) -> None:
    """c. 浮动亏损告警（浮亏/账户权益）。"""
    try:
        balance      = float(account_ref.balance      or 0)
        float_profit = float(account_ref.float_profit or 0)
        if balance <= 0 or float_profit >= 0:
            # 浮盈或余额无效时重置告警等级
            if state.loss_level > 0:
                _cprint(f"  浮亏告警已清除（当前浮盈/持平）", "green")
                state.loss_level = 0
            return

        loss_ratio = abs(float_profit) / balance  # 亏损比例

        if loss_ratio >= LOSS_RED and state.loss_level < 2:
            msg = (f"🔴 [红色] 浮亏 {float_profit:,.0f} 元"
                   f"  占权益 {loss_ratio*100:.2f}%  balance={balance:,.0f}")
            _cprint(msg, "red")
            alert_logger.info("[RED-LOSS] " + msg)
            notifier.send("红色告警 – 浮亏超限", msg, level="red")
            _sound_alert()
            state.loss_level = 2

        elif loss_ratio >= LOSS_YELLOW and state.loss_level < 1:
            msg = (f"🟡 [黄色] 浮亏 {float_profit:,.0f} 元"
                   f"  占权益 {loss_ratio*100:.2f}%  balance={balance:,.0f}")
            _cprint(msg, "yellow")
            alert_logger.info("[YLW-LOSS] " + msg)
            notifier.send("黄色告警 – 浮亏偏大", msg, level="yellow")
            state.loss_level = 1

        elif loss_ratio < LOSS_YELLOW and state.loss_level > 0:
            _cprint(f"  浮亏告警已清除（浮亏 {loss_ratio*100:.2f}%）", "green")
            state.loss_level = 0

    except Exception as e:
        logger.debug("浮亏告警检查异常: %s", e)


def _check_expiry(
    position_ref,
    api,
    state: AlertState,
    notifier: Notifier,
) -> None:
    """d. 持仓临近到期告警（< 7 天），每日仅触发一次。"""
    today_str = date.today().isoformat()
    if state.expiry_alerted_date == today_str:
        return  # 今日已提醒，跳过

    try:
        today_ts = pd.Timestamp.today().normalize()
        near_expiry: list[str] = []

        for symbol, pos in position_ref.items():
            # 只检查有持仓的期权合约（MO 开头）
            tq_symbol = f"CFFEX.{symbol}" if not symbol.startswith("CFFEX.") else symbol
            # TqSdk position key 格式不含交易所前缀，需要拼接
            vol_long  = int(getattr(pos, "volume_long",  0) or 0)
            vol_short = int(getattr(pos, "volume_short", 0) or 0)
            if vol_long + vol_short <= 0:
                continue
            if "MO" not in symbol:
                continue

            try:
                q = api.get_quote(tq_symbol)
                expire_dt = getattr(q, "expire_datetime", None)
                if expire_dt:
                    # expire_datetime 为纳秒时间戳
                    expire_ts = pd.Timestamp(expire_dt, unit="ns", tz="UTC").tz_convert(
                        "Asia/Shanghai"
                    ).normalize().tz_localize(None)
                    days_left = (expire_ts - today_ts).days
                    if days_left < EXPIRY_WARN_DAYS:
                        near_expiry.append(f"{symbol}（剩余 {days_left} 天）")
            except Exception:
                pass

        if near_expiry:
            msg = "⏰ [到期提醒] 以下持仓距到期 < 7 天，请注意 Gamma 加速：" + "  ".join(near_expiry)
            _cprint(msg, "yellow")
            alert_logger.info("[EXPIRY] " + msg)
            notifier.send("持仓临近到期提醒", msg, level="yellow")
            state.expiry_alerted_date = today_str

    except Exception as e:
        logger.debug("到期告警检查异常: %s", e)


# ---------------------------------------------------------------------------
# 行情快照写入
# ---------------------------------------------------------------------------

def _save_snapshot(api, symbols: list[str], db) -> int:
    """拉取 symbols 的行情快照并批量写入 tq_snapshots。"""
    FIELDS = (
        "last_price", "bid_price1", "ask_price1",
        "bid_volume1", "ask_volume1", "volume",
        "open_interest", "highest", "lowest", "open",
    )
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    records = []
    for symbol in symbols:
        try:
            q = api.get_quote(symbol)
            row: dict = {"snapshot_time": now_str, "symbol": symbol}
            for f in FIELDS:
                v = getattr(q, f, None)
                row[f] = float(v) if v is not None else None
            records.append(row)
        except Exception as e:
            logger.warning("快照获取失败 %s: %s", symbol, e)

    if not records:
        return 0
    try:
        db.upsert_dataframe("tq_snapshots", pd.DataFrame(records))
        return len(records)
    except Exception as e:
        logger.error("写入 tq_snapshots 失败: %s", e)
        return 0


# ---------------------------------------------------------------------------
# 主监控循环
# ---------------------------------------------------------------------------

def _run_monitor(notifier: Notifier) -> None:
    """连接 TQ 并进入监控主循环。"""
    from data.sources.tq_client import TqClient
    from tqsdk import TqApi, TqAuth, TqAccount  # noqa: F401 — 校验已安装

    creds    = _tq_credentials()
    has_broker = _has_tq_broker()
    client   = TqClient(**creds)
    client.connect()
    api      = client._api
    db       = _open_db()
    state    = AlertState()

    _cprint(f"已连接天勤  IM主力={IM_MAIN}  实盘账户={'是' if has_broker else '否（仅行情）'}", "green")
    alert_logger.info("=== intraday_monitor 启动 ===")

    # 订阅 IM 主力行情（初次 get_quote 即完成订阅）
    quote_im = api.get_quote(IM_MAIN)

    # 实盘账户引用（仅在有 broker 时有效）
    account_ref  = api.get_account()  if has_broker else None
    position_ref = api.get_position() if has_broker else None

    # 完成初次数据加载
    api.wait_update(deadline=time.time() + 30)

    now = time.time()
    next_check    = now + CHECK_INTERVAL
    next_snapshot = now + SNAPSHOT_INTERVAL

    _cprint(
        f"监控已启动  检查间隔={CHECK_INTERVAL}s  快照间隔={SNAPSHOT_INTERVAL}s  "
        f"日志={ALERTS_LOG}",
        "info",
    )

    # ── 主循环 ──────────────────────────────────────────────────────────
    while not _shutdown.is_set():
        deadline = min(next_check, next_snapshot, time.time() + CHECK_INTERVAL)
        api.wait_update(deadline=deadline)

        now = time.time()

        # 每分钟：检查告警
        if now >= next_check:
            _check_price(quote_im, state, notifier)
            if has_broker and account_ref is not None and position_ref is not None:
                _check_margin(account_ref, state, notifier)
                _check_float_loss(account_ref, state, notifier)
                _check_expiry(position_ref, api, state, notifier)
            next_check = now + CHECK_INTERVAL

        # 每5分钟：行情快照
        if now >= next_snapshot:
            n = _save_snapshot(api, SNAPSHOT_SYMBOLS, db)
            logger.info("快照已写入 %d 条  →  tq_snapshots", n)
            next_snapshot = now + SNAPSHOT_INTERVAL

    # ── 清理 ─────────────────────────────────────────────────────────────
    _cprint("收到退出信号，正在断开天勤连接...", "info")
    alert_logger.info("=== intraday_monitor 停止 ===")
    client.disconnect()
    _cprint("已安全退出", "green")


# ---------------------------------------------------------------------------
# 关闭信号
# ---------------------------------------------------------------------------

_shutdown = threading.Event()


def _handle_signal(signum, frame):  # noqa: ARG001
    _cprint(f"\n收到信号 {signum}，准备退出...", "info")
    _shutdown.set()


signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ---------------------------------------------------------------------------
# 后台运行 (--background)
# ---------------------------------------------------------------------------

def _relaunch_background() -> None:
    """以 nohup 在后台重新启动本脚本（去掉 --background 参数）。"""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = LOGS_DIR / "intraday_monitor.out"
    cmd = [sys.executable] + [a for a in sys.argv if a != "--background"]
    with open(out_file, "a") as fh:
        proc = subprocess.Popen(
            cmd,
            stdout=fh,
            stderr=fh,
            start_new_session=True,   # 等价于 nohup（脱离当前终端会话）
        )
    print(f"后台监控已启动  PID={proc.pid}")
    print(f"输出日志: {out_file}")
    print(f"告警日志: {ALERTS_LOG}")
    print(f"停止命令: kill {proc.pid}")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="IM 期权盘中告警监控",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--background", action="store_true",
        help="以后台 nohup 方式运行",
    )
    parser.add_argument(
        "--notifier", choices=["none"], default="none",
        help="通知渠道（预留接口，目前仅支持 none）",
    )
    args = parser.parse_args()

    if args.background:
        _relaunch_background()
        return

    if not _has_tq_auth():
        print("[错误] 未配置 TQ_ACCOUNT / TQ_PASSWORD 环境变量，无法连接天勤")
        sys.exit(1)

    notifier = Notifier()   # 默认空实现；后续可替换为 WechatNotifier() 等
    _run_monitor(notifier)


if __name__ == "__main__":
    main()
