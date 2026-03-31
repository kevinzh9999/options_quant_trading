"""
download_options_daily.py
-------------------------
批量下载股指期权历史日线数据并存入 SQLite 数据库。

用法：
    python data/download_scripts/download_options_daily.py \
        --start 20190101 --end 20241231

    # 指定交易所
    python data/download_scripts/download_options_daily.py \
        --start 20190101 --exchange CFFEX SSE

    # 增量模式
    python data/download_scripts/download_options_daily.py --update

品种上市日期（建议 --start 不早于对应日期）：
    IO  沪深300股指期权（CFFEX）：20191223
    MO  中证1000股指期权（CFFEX）：20220722

流程：
  1. 获取交易日历，确定需要下载的交易日列表（按年分段避免 Tushare 2000 行截断）
  2. 对每个交易日，按交易所调用 opt_daily 接口下载当日全量期权行情
  3. upsert 到 options_daily 表（幂等）
  4. 同步下载期权合约基本信息到 options_contracts 表
  5. 连续 10 个交易日返回空数据时，自动跳至下月首日继续尝试
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import load_config
from data.quality_check import DataQualityChecker
from data.sources.tushare_client import TushareClient
from data.storage.db_manager import DBManager

logger = logging.getLogger(__name__)

DAILY_TABLE = "options_daily"
CONTRACTS_TABLE = "options_contracts"


def _today() -> str:
    return date.today().strftime("%Y%m%d")


def _next_day(yyyymmdd: str) -> str:
    d = date(int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8]))
    return (d + timedelta(days=1)).strftime("%Y%m%d")


def _first_day_of_next_month(yyyymmdd: str) -> str:
    """返回给定日期所在月份的下个月第一天（YYYYMMDD）"""
    year = int(yyyymmdd[:4])
    month = int(yyyymmdd[4:6])
    if month == 12:
        return f"{year + 1}0101"
    return f"{year}{month + 1:02d}01"


def _year_ranges(start: str, end: str) -> list[tuple[str, str]]:
    """将 [start, end] 按年拆分为子区间列表，避免单次请求超 Tushare 2000 行限制"""
    start_year = int(start[:4])
    end_year = int(end[:4])
    ranges = []
    for year in range(start_year, end_year + 1):
        y_start = max(start, f"{year}0101")
        y_end = min(end, f"{year}1231")
        if y_start <= y_end:
            ranges.append((y_start, y_end))
    return ranges


def _get_trade_dates(
    client: TushareClient,
    exchange: str,
    start_date: str,
    end_date: str,
) -> list[str]:
    """获取指定区间内的交易日列表，按年分段请求避免 Tushare 2000 行截断"""
    open_days: list[str] = []
    for y_start, y_end in _year_ranges(start_date, end_date):
        cal_df = client.get_trade_calendar(
            exchange=exchange,
            start_date=y_start,
            end_date=y_end,
        )
        if cal_df.empty:
            continue
        open_days.extend(cal_df[cal_df["is_open"] == 1]["trade_date"].tolist())
        time.sleep(0.5)
    return sorted(open_days)


def download_options_contracts(
    exchanges: list[str],
    db: DBManager,
    client: TushareClient,
) -> None:
    """下载期权合约基本信息（一次性全量，幂等）"""
    for exchange in exchanges:
        logger.info("下载 %s 期权合约基本信息...", exchange)
        try:
            df = client.get_options_contracts(exchange=exchange)
        except Exception as exc:
            logger.error("%s 合约信息下载失败: %s", exchange, exc)
            continue

        if df.empty:
            logger.warning("%s: 未获取到合约信息", exchange)
            continue

        saved = db.upsert_dataframe(CONTRACTS_TABLE, df)
        logger.info("%s: 写入 %d 条合约信息", exchange, saved)


def download_options_daily(
    start_date: str,
    end_date: str,
    exchanges: list[str],
    db: DBManager,
    client: TushareClient,
    update_mode: bool = False,
) -> None:
    """
    按交易日逐日下载期权日线行情。

    Parameters
    ----------
    start_date : str
        起始日期（update_mode 时自动从 DB 最新日期+1开始）
    end_date : str
        结束日期
    exchanges : list[str]
        交易所列表，如 ["CFFEX", "SSE"]
    db : DBManager
    client : TushareClient
    update_mode : bool
    """
    # 先同步合约基本信息
    download_options_contracts(exchanges, db, client)

    checker = DataQualityChecker()
    total_saved = 0

    for exchange in exchanges:
        if update_mode:
            latest = db.get_latest_date(DAILY_TABLE)
            fetch_start = _next_day(latest) if latest else start_date
        else:
            fetch_start = start_date

        if fetch_start > end_date:
            logger.info("%s: 已是最新，跳过", exchange)
            continue

        logger.info("获取 %s 交易日历 %s ~ %s...", exchange, fetch_start, end_date)
        trade_dates = _get_trade_dates(client, exchange, fetch_start, end_date)
        if not trade_dates:
            logger.warning("%s: 无交易日，跳过", exchange)
            continue

        logger.info("%s: 共 %d 个交易日需要下载", exchange, len(trade_dates))

        # 按月汇总进度；连续 10 个空交易日自动跳至下月
        EMPTY_SKIP_THRESHOLD = 10
        current_month = None   # "YYYYMM"
        month_saved = 0
        empty_streak = 0       # 连续空返回计数
        skip_until: str | None = None  # 跳过至该日期（含）前的所有交易日

        for j, trade_date in enumerate(trade_dates, 1):
            # 跳过逻辑：当前日期在 skip_until 之前则略过
            if skip_until and trade_date < skip_until:
                continue
            skip_until = None

            month = trade_date[:6]  # "YYYYMM"
            if month != current_month:
                if current_month is not None:
                    print(
                        f"  {current_month[:4]}-{current_month[4:]} 月完成，"
                        f"{month_saved} 行",
                        flush=True,
                    )
                    month_saved = 0
                current_month = month
                print(
                    f"  正在下载 {month[:4]}-{month[4:]} 月 ({exchange})...",
                    flush=True,
                )

            try:
                df = client.get_options_daily(exchange=exchange, trade_date=trade_date)
            except Exception as exc:
                logger.warning("%s %s 下载失败（跳过）: %s", exchange, trade_date, exc)
                time.sleep(1)
                continue

            if df.empty:
                logger.debug("%s %s: 无数据", exchange, trade_date)
                empty_streak += 1
                if empty_streak >= EMPTY_SKIP_THRESHOLD:
                    next_month = _first_day_of_next_month(trade_date)
                    logger.info(
                        "%s: 连续 %d 个交易日无数据，跳至 %s",
                        exchange, empty_streak, next_month,
                    )
                    print(
                        f"  连续 {empty_streak} 日无数据，跳至 {next_month[:4]}-{next_month[4:6]} 月",
                        flush=True,
                    )
                    skip_until = next_month
                    empty_streak = 0
                time.sleep(1)
                continue

            empty_streak = 0
            saved = db.upsert_dataframe(DAILY_TABLE, df)
            total_saved += saved
            month_saved += saved
            time.sleep(1)

            if j % 10 == 0:
                print(
                    f"  进度 {j}/{len(trade_dates)} 天  合计写入 {total_saved} 行",
                    flush=True,
                )
                logger.info("  [%s] 进度 %d/%d 天，合计写入 %d 行",
                            exchange, j, len(trade_dates), total_saved)

        if current_month is not None:
            print(
                f"  {current_month[:4]}-{current_month[4:]} 月完成，{month_saved} 行",
                flush=True,
            )

        # 对最近一批数据做质量检查
        try:
            recent_df = db.query(
                f"SELECT * FROM {DAILY_TABLE} WHERE trade_date >= ? ORDER BY trade_date",
                (fetch_start,),
            )
            if not recent_df.empty:
                result = checker.check_options_daily(recent_df)
                logger.info(
                    "%s 质量检查: total=%d, is_clean=%s, violations=%d",
                    exchange,
                    result["total_rows"],
                    result["is_clean"],
                    len(result["ohlc_violations"]),
                )
        except Exception as exc:
            logger.warning("质量检查失败: %s", exc)

    # 统计实际日期范围
    try:
        row = db._conn.execute(
            f"SELECT MIN(trade_date), MAX(trade_date) FROM {DAILY_TABLE}"
        ).fetchone()
        date_range = f"{row[0]} ~ {row[1]}" if row and row[0] else "无数据"
    except Exception:
        date_range = "无法查询"

    logger.info("期权日线下载完成，共写入 %d 行到 %s，数据库日期范围: %s",
                total_saved, DAILY_TABLE, date_range)
    print(f"\n下载完成：共写入 {total_saved} 行，数据库日期范围 {date_range}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "下载股指期权历史日线数据。\n"
            "品种上市日期参考：IO(沪深300)=20191223，MO(中证1000)=20220722"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--start", default="20191223",
        help="起始日期 YYYYMMDD（默认 20191223，即 IO 上市日）",
    )
    parser.add_argument("--end", default=None, help="结束日期 YYYYMMDD（默认今天）")
    parser.add_argument(
        "--exchange", nargs="+", default=["CFFEX"],
        help="交易所列表（默认 CFFEX）",
    )
    parser.add_argument(
        "--update", action="store_true",
        help="增量模式：自动从数据库最新日期续传",
    )
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    end_date = args.end or _today()

    config = load_config(args.config)
    client = TushareClient(token=config.tushare_token)

    with DBManager(config.db_path) as db:
        download_options_daily(
            start_date=args.start,
            end_date=end_date,
            exchanges=args.exchange,
            db=db,
            client=client,
            update_mode=args.update,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
