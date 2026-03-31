"""
download_futures_min.py
-----------------------
批量下载股指期货历史分钟线数据（默认 5 分钟线），存入 SQLite 数据库。
分钟线数据用于计算已实现波动率（Realized Volatility）。

用法：
    python data/download_scripts/download_futures_min.py \
        --start 20200101 --end 20241231 --freq 5min

    # 增量模式
    python data/download_scripts/download_futures_min.py --update --freq 5min

注意：
- 分钟线数据量较大，请确保有足够磁盘空间和 Tushare 积分
- 按天逐日下载，支持断点续传（遇错跳过当天，不中断整体流程）
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import load_config
from data.sources.tushare_client import TushareClient
from data.storage.db_manager import DBManager

logger = logging.getLogger(__name__)

UNDERLYING_CODES = ["IF", "IH", "IC", "IM"]
EXCHANGE = "CFFEX"
TABLE = "futures_min"


def _today() -> str:
    return date.today().strftime("%Y%m%d")


def _date_range(start: str, end: str) -> list[str]:
    """生成 [start, end] 内所有日历日期（YYYYMMDD）列表"""
    s = date(int(start[:4]), int(start[4:6]), int(start[6:8]))
    e = date(int(end[:4]), int(end[4:6]), int(end[6:8]))
    out = []
    cur = s
    while cur <= e:
        out.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return out


def _next_day(yyyymmdd: str) -> str:
    d = date(int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8]))
    return (d + timedelta(days=1)).strftime("%Y%m%d")


def download_futures_min(
    start_date: str,
    end_date: str,
    freq: str,
    db: DBManager,
    client: TushareClient,
    update_mode: bool = False,
) -> None:
    """
    按日逐日下载主力股指期货分钟线，写入 futures_min 表。

    Parameters
    ----------
    start_date : str
        起始日期（update_mode 时忽略，使用 DB 最新+1）
    end_date : str
        结束日期
    freq : str
        K 线周期，如 "1min" / "5min" / "15min"
    db : DBManager
    client : TushareClient
    update_mode : bool
        增量模式
    """
    # 获取合约列表
    logger.info("获取 %s 期货合约列表...", EXCHANGE)
    mapping_df = client.get_futures_mapping(exchange=EXCHANGE)
    if mapping_df.empty:
        logger.warning("未获取到合约映射，跳过下载")
        return

    all_codes = sorted(mapping_df["ts_code"].dropna().unique())
    codes = [c for c in all_codes if any(c.startswith(u) for u in UNDERLYING_CODES)]
    logger.info("共 %d 个合约", len(codes))

    total_saved = 0

    for i, ts_code in enumerate(codes, 1):
        if update_mode:
            # futures_min 按 datetime 存储，用 get_latest_date 查 datetime 列
            # futures_min 无 trade_date 列，直接查 datetime 字段
            latest_dt = db.query_scalar(
                f"SELECT MAX(datetime) FROM {TABLE} WHERE ts_code = ?",
                (ts_code,),
            )
            fetch_start = _next_day(latest_dt[:10].replace("-", "")) if latest_dt else start_date
        else:
            fetch_start = start_date

        if fetch_start > end_date:
            logger.debug("[%d/%d] %s: 已是最新，跳过", i, len(codes), ts_code)
            continue

        dates = _date_range(fetch_start, end_date)
        logger.info("[%d/%d] %s: 逐日下载 %s ~ %s (%d 天)",
                    i, len(codes), ts_code, fetch_start, end_date, len(dates))

        for j, trade_date in enumerate(dates, 1):
            try:
                df = client.get_futures_min(ts_code, trade_date, trade_date, freq=freq)
            except Exception as exc:
                logger.warning("%s %s 下载失败（跳过）: %s", ts_code, trade_date, exc)
                continue

            if df.empty:
                continue

            saved = db.upsert_dataframe(TABLE, df)
            total_saved += saved
            if j % 20 == 0 or j == len(dates):
                logger.info("  进度 %d/%d 天，本合约累计写入 %d 行",
                            j, len(dates), total_saved)

    logger.info("分钟线下载完成，共写入 %d 行到 %s", total_saved, TABLE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载股指期货历史分钟线数据")
    parser.add_argument("--start", default="20200101", help="起始日期 YYYYMMDD")
    parser.add_argument("--end", default=None, help="结束日期 YYYYMMDD（默认今天）")
    parser.add_argument(
        "--freq", default="5min",
        choices=["1min", "5min", "15min"],
        help="K 线周期（默认 5min）",
    )
    parser.add_argument(
        "--update", action="store_true",
        help="增量模式：自动从最新日期续传",
    )
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    end_date = args.end or _today()

    config = load_config(args.config)
    client = TushareClient(token=config.tushare_token)

    with DBManager(config.db_path) as db:
        download_futures_min(
            start_date=args.start,
            end_date=end_date,
            freq=args.freq,
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
