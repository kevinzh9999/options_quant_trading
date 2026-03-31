"""
download_index_daily.py
-----------------------
批量下载指数日线数据并存入 SQLite 数据库。

支持的指数（标的现货指数）：
  000852.SH  中证1000（IM / MO 期货/期权标的）
  000300.SH  沪深300（IF / IO 标的）
  000016.SH  上证50（IH 标的）
  000905.SH  中证500（IC 标的）

用法：
    # 全量下载（从 2015-01-01 起）
    python data/download_scripts/download_index_daily.py --start 20150101

    # 增量更新（从数据库最新日期续传）
    python data/download_scripts/download_index_daily.py --update
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import load_config
from data.sources.tushare_client import TushareClient
from data.storage.db_manager import DBManager

logger = logging.getLogger(__name__)

# 需要下载的指数
INDEX_CODES = [
    "000852.SH",   # 中证1000：IM/MO 标的
    "000300.SH",   # 沪深300：IF/IO 标的
    "000016.SH",   # 上证50：IH 标的
    "000905.SH",   # 中证500：IC 标的
]
TABLE = "index_daily"


def _today() -> str:
    return date.today().strftime("%Y%m%d")


def _next_day(yyyymmdd: str) -> str:
    d = date(int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8]))
    return (d + timedelta(days=1)).strftime("%Y%m%d")


def _year_ranges(start: str, end: str) -> list[tuple[str, str]]:
    """按年拆分区间，避免超 Tushare 单次行数限制。"""
    start_year, end_year = int(start[:4]), int(end[:4])
    ranges = []
    for year in range(start_year, end_year + 1):
        y_start = max(start, f"{year}0101")
        y_end   = min(end,   f"{year}1231")
        if y_start <= y_end:
            ranges.append((y_start, y_end))
    return ranges


def download_index_daily(
    start_date: str,
    end_date: str,
    db: DBManager,
    client: TushareClient,
    update_mode: bool = False,
    index_codes: list[str] = None,
) -> None:
    """
    下载指定指数的日线数据并写入 index_daily 表。

    Parameters
    ----------
    start_date : str
        全量模式的起始日期（update_mode=True 时从 DB 最新日期续传）
    end_date : str
        结束日期 YYYYMMDD
    db : DBManager
        数据库管理器
    client : TushareClient
        Tushare 客户端
    update_mode : bool
        True = 增量模式，自动从 DB 最新日期 +1 天开始
    index_codes : list[str], optional
        指定指数列表，默认 INDEX_CODES
    """
    codes = index_codes or INDEX_CODES
    total_saved = 0

    for ts_code in codes:
        if update_mode:
            latest = db.get_latest_date(TABLE, ts_code=ts_code)
            fetch_start = _next_day(latest) if latest else start_date
        else:
            fetch_start = start_date

        if fetch_start > end_date:
            logger.debug("%s: 已是最新，跳过", ts_code)
            print(f"  {ts_code}: 已是最新，跳过", flush=True)
            continue

        logger.info("%s: 下载 %s ~ %s", ts_code, fetch_start, end_date)
        print(f"\n{'='*50}", flush=True)
        print(f"  {ts_code}: {fetch_start} ~ {end_date}", flush=True)

        all_dfs: list[pd.DataFrame] = []
        for y_start, y_end in _year_ranges(fetch_start, end_date):
            print(f"  正在下载 {y_start[:4]} 年...", flush=True)
            try:
                df_year = client.get_index_daily(ts_code, y_start, y_end)
            except Exception as exc:
                logger.error("%s %s 下载失败: %s", ts_code, y_start[:4], exc)
                print(f"  {y_start[:4]} 年下载失败: {exc}", flush=True)
                time.sleep(1)
                continue

            if df_year.empty:
                print(f"  {y_start[:4]} 年无数据", flush=True)
                time.sleep(1)
                continue

            saved = db.upsert_dataframe(TABLE, df_year)
            total_saved += saved
            print(f"  {y_start[:4]} 年完成，写入 {saved} 行", flush=True)
            all_dfs.append(df_year)
            time.sleep(0.6)

        if not all_dfs:
            print(f"  {ts_code}: 无新数据", flush=True)

    # 结果摘要
    try:
        row = db._conn.execute(
            f"SELECT MIN(trade_date), MAX(trade_date), COUNT(*) FROM {TABLE}"
        ).fetchone()
        if row and row[0]:
            print(f"\n{'='*50}", flush=True)
            print(f"index_daily 日期范围: {row[0]} ~ {row[1]}，共 {row[2]} 行", flush=True)
    except Exception:
        pass

    print(f"\n下载完成：共写入 {total_saved} 行", flush=True)
    logger.info("index_daily 下载完成，共写入 %d 行", total_saved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载指数日线数据（CSI1000/CSI300/SSE50/CSI500）")
    parser.add_argument("--start",  default="20150101",
                        help="起始日期 YYYYMMDD（默认 20150101）")
    parser.add_argument("--end",    default=None,
                        help="结束日期 YYYYMMDD（默认今天）")
    parser.add_argument("--update", action="store_true",
                        help="增量模式：自动从数据库最新日期续传，忽略 --start")
    parser.add_argument("--codes",  nargs="+", default=None,
                        help="指定指数代码列表，默认全部四个")
    parser.add_argument("--config", default="config/config.yaml",
                        help="配置文件路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    end_date = args.end or _today()

    config = load_config(args.config)
    client = TushareClient(token=config.tushare_token)

    with DBManager(config.db_path) as db:
        db.initialize_tables()   # 确保 index_daily 表已创建
        download_index_daily(
            start_date=args.start,
            end_date=end_date,
            db=db,
            client=client,
            update_mode=args.update,
            index_codes=args.codes,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
