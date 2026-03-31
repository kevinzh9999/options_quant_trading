"""
download_futures_daily.py
-------------------------
批量下载股指期货历史日线数据并存入 SQLite 数据库。

用法：
    python data/download_scripts/download_futures_daily.py \
        --start 20150101 --end 20241231

    # 增量模式（从数据库最新日期续传）
    python data/download_scripts/download_futures_daily.py --update

支持增量下载：若数据库中已有数据，自动从最新日期续传。
下载完成后对数据进行简单质量检查并输出摘要。
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# 将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import load_config
from data.quality_check import DataQualityChecker
from data.sources.tushare_client import TushareClient
from data.storage.db_manager import DBManager

logger = logging.getLogger(__name__)

# 需要下载的股指期货品种（Tushare 交易所后缀格式的品种前缀）
UNDERLYING_CODES = ["IF", "IH", "IC", "IM"]
EXCHANGE = "CFFEX"
TABLE = "futures_daily"


def _today() -> str:
    return date.today().strftime("%Y%m%d")


def _next_day(yyyymmdd: str) -> str:
    """返回指定日期的下一天（YYYYMMDD）"""
    d = date(int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8]))
    return (d + timedelta(days=1)).strftime("%Y%m%d")


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


def download_futures_daily(
    start_date: str,
    end_date: str,
    db: DBManager,
    client: TushareClient,
    update_mode: bool = False,
) -> None:
    """
    下载股指期货日线数据并存入数据库。

    Parameters
    ----------
    start_date : str
        默认起始日期（update_mode=True 时自动从 DB 最新日期续传）
    end_date : str
        结束日期 YYYYMMDD
    db : DBManager
        数据库管理器
    client : TushareClient
        Tushare 客户端
    update_mode : bool
        增量模式：自动从 DB 最新日期+1天开始，忽略 start_date
    """
    # 先获取所有该交易所合约的 ts_code
    logger.info("获取 %s 期货合约列表...", EXCHANGE)
    mapping_df = client.get_futures_mapping(exchange=EXCHANGE)
    if mapping_df.empty:
        logger.warning("未获取到合约映射，跳过下载")
        return

    # 去重取全部 ts_code
    all_codes = sorted(mapping_df["ts_code"].dropna().unique())
    # 只保留 UNDERLYING_CODES 相关品种
    codes = [c for c in all_codes if any(c.startswith(u) for u in UNDERLYING_CODES)]
    logger.info("共找到 %d 个合约: %s... (截断显示)", len(codes), codes[:5])

    checker = DataQualityChecker()
    total_saved = 0

    for i, ts_code in enumerate(codes, 1):
        if update_mode:
            latest = db.get_latest_date(TABLE, ts_code=ts_code)
            fetch_start = _next_day(latest) if latest else start_date
        else:
            fetch_start = start_date

        if fetch_start > end_date:
            logger.debug("[%d/%d] %s: 已是最新，跳过", i, len(codes), ts_code)
            continue

        logger.info("[%d/%d] %s: 下载 %s ~ %s",
                    i, len(codes), ts_code, fetch_start, end_date)

        all_dfs = []
        for y_start, y_end in _year_ranges(fetch_start, end_date):
            print(f"  正在下载 {y_start[:4]} 年 ({ts_code})...", flush=True)
            try:
                df_year = client.get_futures_daily(ts_code, y_start, y_end)
            except Exception as exc:
                logger.error("%s %s 下载失败: %s", ts_code, y_start[:4], exc)
                time.sleep(1)
                continue

            if df_year.empty:
                print(f"  {y_start[:4]} 年无数据", flush=True)
                time.sleep(1)
                continue

            saved_year = db.upsert_dataframe(TABLE, df_year)
            total_saved += saved_year
            print(f"  {y_start[:4]} 年完成，{saved_year} 行", flush=True)
            all_dfs.append(df_year)
            time.sleep(1)

        if not all_dfs:
            logger.debug("%s: 无新数据", ts_code)
            continue

        df = pd.concat(all_dfs, ignore_index=True)

        # 简单质量检查
        result = checker.check_futures_daily(df, ts_code)
        if not result["is_clean"]:
            logger.warning(
                "%s 质量检查: ohlc_violations=%d, volume_anomalies=%d",
                ts_code,
                len(result["ohlc_violations"]),
                len(result["volume_anomalies"]),
            )

    # 统计实际日期范围
    try:
        conn = db._conn
        row = conn.execute(
            f"SELECT MIN(trade_date), MAX(trade_date) FROM {TABLE}"
        ).fetchone()
        date_range = f"{row[0]} ~ {row[1]}" if row and row[0] else "无数据"
    except Exception:
        date_range = "无法查询"

    logger.info("下载完成，共写入 %d 行到 %s，数据库日期范围: %s",
                total_saved, TABLE, date_range)
    print(f"\n下载完成：共写入 {total_saved} 行，数据库日期范围 {date_range}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载股指期货历史日线数据")
    parser.add_argument("--start", default="20150101", help="起始日期 YYYYMMDD（默认 20150101）")
    parser.add_argument("--end", default=None, help="结束日期 YYYYMMDD（默认今天）")
    parser.add_argument(
        "--update", action="store_true",
        help="增量模式：自动从数据库最新日期续传，忽略 --start"
    )
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    end_date = args.end or _today()

    config = load_config(args.config)
    client = TushareClient(token=config.tushare_token)

    with DBManager(config.db_path) as db:
        download_futures_daily(
            start_date=args.start,
            end_date=end_date,
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
