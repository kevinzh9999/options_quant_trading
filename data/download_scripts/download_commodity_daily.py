"""
download_commodity_daily.py
---------------------------
职责：下载商品期货日线数据（CTP 品种）。

支持品种：
- 能化：原油（SC）、天然气（LU）、燃料油（FU）
- 黑色：螺纹钢（RB）、热轧卷板（HC）、铁矿石（I）、焦炭（J）、焦煤（JM）
- 有色：铜（CU）、铝（AL）、锌（ZN）、镍（NI）、黄金（AU）、白银（AG）
- 农产品：豆粕（M）、菜粕（RM）、棉花（CF）、白糖（SR）、玉米（C）
- 金融：股指期货（IF/IH/IC/IM，亦由此脚本下载）

用法：
    python -m data.download_scripts.download_commodity_daily \
        --start 20200101 --end 20241231 --exchange SHFE DCE CZCE INE

    # 按品种筛选
    python -m data.download_scripts.download_commodity_daily \
        --start 20200101 --symbols RB.SHF CU.SHF I.DCE

    # 增量模式
    python -m data.download_scripts.download_commodity_daily --update
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# 确保项目根目录在 Python 路径中
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import load_config
from data.quality_check import DataQualityChecker
from data.sources.tushare_client import TushareClient
from data.storage.db_manager import DBManager, get_db

logger = logging.getLogger(__name__)

TABLE = "futures_daily"

# 各交易所主力品种代码（Tushare 格式）
EXCHANGE_PRODUCTS = {
    "SHFE": ["CU", "AL", "ZN", "NI", "SN", "PB", "AU", "AG", "RB", "HC", "FU", "RU", "BU"],
    "DCE":  ["I", "J", "JM", "M", "Y", "P", "C", "CS", "A", "B", "L", "PP", "V", "EG", "EB"],
    "CZCE": ["CF", "SR", "RM", "OI", "MA", "TA", "ZC", "AP", "CJ", "PF"],
    "INE":  ["SC", "LU", "NR"],
    "GFEX": ["SI", "LC"],
    "CFFEX": ["IF", "IH", "IC", "IM", "T", "TF", "TS"],
}

# Tushare 交易所后缀映射（TqSdk 前缀 → Tushare 后缀）
_EXCHANGE_SUFFIX = {
    "SHFE": "SHF",
    "DCE":  "DCE",
    "CZCE": "ZCE",
    "INE":  "INE",
    "GFEX": "GFX",
    "CFFEX": "CFX",
}


def _today() -> str:
    return date.today().strftime("%Y%m%d")


def _next_day(yyyymmdd: str) -> str:
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


def _get_contracts_for_exchange(
    exchange: str,
    client: TushareClient,
) -> list[str]:
    """
    通过 fut_mapping 接口获取某交易所的全部合约 ts_code 列表。
    只返回属于 EXCHANGE_PRODUCTS 中定义的主力品种的合约。
    """
    products = EXCHANGE_PRODUCTS.get(exchange, [])
    if not products:
        logger.warning("交易所 %s 未在 EXCHANGE_PRODUCTS 中定义", exchange)
        return []

    mapping_df = client.get_futures_mapping(exchange=exchange)
    if mapping_df.empty:
        return []

    all_codes = mapping_df["ts_code"].dropna().unique().tolist()
    # 只保留属于指定品种的合约
    filtered = [c for c in all_codes if any(c.startswith(p) for p in products)]
    return sorted(filtered)


def download_commodity_daily(
    start_date: str,
    end_date: str,
    exchanges: list[str],
    db: DBManager,
    client: TushareClient,
    update_mode: bool = False,
    symbols: list[str] | None = None,
) -> None:
    """
    下载商品期货日线数据并存入数据库。

    Parameters
    ----------
    start_date : str
        开始日期（update_mode 时按合约自动续传）
    end_date : str
        结束日期
    exchanges : list[str]
        交易所代码列表，如 ['SHFE', 'DCE']
    db : DBManager
    client : TushareClient
    update_mode : bool
        增量模式：按合约查 DB 最新日期，自动续传
    symbols : list[str], optional
        指定合约列表（Tushare ts_code 格式），不为 None 时忽略 exchanges 参数
    """
    # 确定要下载的合约列表
    if symbols:
        codes = list(symbols)
        logger.info("按指定合约下载: %s", codes[:10])
    else:
        codes = []
        for exchange in exchanges:
            logger.info("获取 %s 合约列表...", exchange)
            exchange_codes = _get_contracts_for_exchange(exchange, client)
            codes.extend(exchange_codes)
            logger.info("  %s: %d 个合约", exchange, len(exchange_codes))

    if not codes:
        logger.warning("无合约可下载，退出")
        return

    logger.info("共 %d 个合约待下载", len(codes))

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
                df_year = client.get_commodity_daily(ts_code, y_start, y_end)
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

        # 质量检查
        result = checker.check_futures_daily(df, ts_code)
        if not result["is_clean"]:
            logger.warning(
                "%s 质量检查异常: ohlc_violations=%d, volume_anomalies=%d",
                ts_code,
                len(result["ohlc_violations"]),
                len(result["volume_anomalies"]),
            )

        if i % 10 == 0:
            logger.info("进度 %d/%d，累计写入 %d 行", i, len(codes), total_saved)

    # 优化查询性能
    try:
        db._conn.execute(f"ANALYZE {TABLE}")
        logger.info("已对 %s 执行 ANALYZE", TABLE)
    except Exception:
        pass

    # 统计实际日期范围
    try:
        row = db._conn.execute(
            f"SELECT MIN(trade_date), MAX(trade_date) FROM {TABLE}"
        ).fetchone()
        date_range = f"{row[0]} ~ {row[1]}" if row and row[0] else "无数据"
    except Exception:
        date_range = "无法查询"

    logger.info("商品期货日线下载完成，共写入 %d 行到 %s，数据库日期范围: %s",
                total_saved, TABLE, date_range)
    print(f"\n下载完成：共写入 {total_saved} 行，数据库日期范围 {date_range}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载商品期货日线数据")
    parser.add_argument("--start", default="20150101", help="开始日期 YYYYMMDD")
    parser.add_argument("--end", default=None, help="结束日期 YYYYMMDD（默认今天）")
    parser.add_argument(
        "--exchange",
        nargs="+",
        default=list(EXCHANGE_PRODUCTS.keys()),
        help="交易所代码（默认全部）",
    )
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="指定合约 ts_code 列表，如 RB2410.SHF CU2412.SHF（优先于 --exchange）",
    )
    parser.add_argument(
        "--update", action="store_true",
        help="增量模式：按合约自动从数据库最新日期续传",
    )
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    end_date = args.end or _today()

    config = load_config(args.config)
    client = TushareClient(token=config.tushare_token)

    with get_db(config) as db:
        download_commodity_daily(
            start_date=args.start,
            end_date=end_date,
            exchanges=args.exchange,
            db=db,
            client=client,
            update_mode=args.update,
            symbols=args.symbols,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
