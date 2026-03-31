"""
run_daily.py
------------
每日收盘后运行的主脚本（约 15:15 后执行）。

多策略执行流程：
1. 数据更新 - 拉取当日期货（股指+商品）/期权日线数据
2. 数据质量检查 - 验证数据完整性
3. 已实现波动率计算 - 从分钟线计算当日 RV（vol_arb 策略使用）
4. 模型更新 - GARCH 拟合、协整检验更新、HMM 状态检测
5. 波动率曲面构建 - 计算当日各合约 IV（vol_arb 策略使用）
6. 多策略信号生成 - 调度所有启用的策略生成当日信号
7. 组合风控检查 - 单策略风控 + 组合级风控
8. 订单生成与执行 - dry_run 模式默认开启
9. 结果输出 - 打印信号摘要和操作建议

用法：
    python run_daily.py [--date YYYYMMDD] [--live] [--strategy vol_arb trend_following]

参数：
    --date       指定运行日期，默认为今天
    --live       实盘模式（关闭 dry_run，谨慎使用！）
    --strategy   指定运行的策略（默认运行配置中所有 enabled 策略）
    --verbose    显示详细日志
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# 将当前目录加入 sys.path（支持直接运行）
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from data import UnifiedDataAPI
from models.volatility.realized_vol import compute_realized_vol  # noqa: F401
from models.volatility.garch_model import GJRGARCHModel  # noqa: F401
from models.pricing.vol_surface import VolSurface  # noqa: F401
from strategies.vol_arb.signal import VRPSignalGenerator  # noqa: F401
from risk.risk_checker import RiskChecker
from risk.portfolio_risk import PortfolioRiskManager  # noqa: F401
from risk.position_sizer import PositionSizer
from execution.order_manager import OrderManager

logger = logging.getLogger(__name__)

# 期权策略标的
OPTION_UNDERLYINGS = ["IO", "MO"]
# 期货策略标的
FUTURES_UNDERLYINGS = ["IF", "IH", "IC", "IM"]


def setup_logging(log_dir: str, level: str) -> None:
    """配置日志：同时输出到控制台和文件"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logger.info("日志已初始化，输出文件: %s", log_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="多策略每日运行脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--date",
        default=datetime.today().strftime("%Y%m%d"),
        help="运行日期，格式 YYYYMMDD，默认今天",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="实盘模式（默认为 dry_run，加此参数才会实际下单）",
    )
    parser.add_argument(
        "--strategy",
        nargs="+",
        choices=["vol_arb", "trend_following", "spread_trading", "mean_reversion"],
        default=None,
        help="指定运行的策略（默认运行配置中所有 enabled 策略）",
    )
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--verbose", action="store_true", help="显示详细日志")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trade_date = args.date
    dry_run = not args.live

    # ------------------------------------------------------------------
    # Step 0: 加载配置
    # ------------------------------------------------------------------
    config = load_config(args.config)
    log_level = "DEBUG" if args.verbose else config.log_level
    setup_logging(config.log_dir, log_level)

    logger.info("=" * 60)
    logger.info(
        "多策略运行开始 | 日期: %s | 模式: %s",
        trade_date,
        "实盘" if not dry_run else "干跑",
    )
    logger.info("=" * 60)

    if not dry_run:
        logger.warning("⚠ 实盘模式已开启，订单将实际提交！")

    # ------------------------------------------------------------------
    # Step 0b: 每日数据记录（账户/持仓/成交/模型输出）
    #   独立运行，不依赖后续策略步骤，确保收盘记录完整。
    #   即使策略层尚未实现，本步骤仍正常工作。
    # ------------------------------------------------------------------
    logger.info("[Step 0b] 每日数据记录（daily_record.run_eod）...")
    try:
        from scripts.daily_record import run_eod
        run_eod(
            trade_date=trade_date,
            use_tq=not dry_run,          # 实盘模式才连 TQ
            update_market_data=True,     # 始终触发增量数据更新
        )
        logger.info("[Step 0b] 每日数据记录完成")
    except Exception as exc:
        logger.warning("[Step 0b] 每日数据记录失败（不影响后续策略步骤）: %s", exc)

    # ------------------------------------------------------------------
    # Step 1: 数据更新（期货日线 + 期权日线 + 商品期货）
    # ------------------------------------------------------------------
    logger.info("[Step 1] 数据更新...")
    data_api = UnifiedDataAPI(config)
    raise NotImplementedError("TODO: 实现每日数据更新（期货 + 商品 + 期权）")

    # ------------------------------------------------------------------
    # Step 2: 数据质量检查
    # ------------------------------------------------------------------
    logger.info("[Step 2] 数据质量检查...")
    raise NotImplementedError("TODO: 实现数据质量检查")

    # ------------------------------------------------------------------
    # Step 3: 已实现波动率计算（vol_arb 策略所需）
    # ------------------------------------------------------------------
    logger.info("[Step 3] 计算已实现波动率...")
    raise NotImplementedError("TODO: 实现 RV 计算")

    # ------------------------------------------------------------------
    # Step 4: 模型更新
    #   4a. GARCH 模型更新（vol_arb）
    #   4b. 协整关系检验更新（spread_trading）
    #   4c. HMM 市场状态更新（全局）
    # ------------------------------------------------------------------
    logger.info("[Step 4] 模型更新（GARCH / 协整 / HMM）...")
    raise NotImplementedError("TODO: 实现多模型更新")

    # ------------------------------------------------------------------
    # Step 5: 波动率曲面构建（vol_arb 策略所需）
    # ------------------------------------------------------------------
    logger.info("[Step 5] 构建波动率曲面...")
    raise NotImplementedError("TODO: 实现波动率曲面构建")

    # ------------------------------------------------------------------
    # Step 6: 多策略信号生成
    # ------------------------------------------------------------------
    logger.info("[Step 6] 多策略信号生成...")
    raise NotImplementedError("TODO: 遍历所有启用策略调用 strategy.run(trade_date, market_data)")

    # ------------------------------------------------------------------
    # Step 7: 风控检查（单策略 + 组合级）
    # ------------------------------------------------------------------
    logger.info("[Step 7] 运行风控检查...")
    raise NotImplementedError("TODO: 实现单策略风控 + PortfolioRiskManager 检查")

    # ------------------------------------------------------------------
    # Step 8: 订单生成与执行（dry_run 或实盘）
    # ------------------------------------------------------------------
    logger.info("[Step 8] 生成订单 (dry_run=%s)...", dry_run)
    raise NotImplementedError("TODO: 实现订单生成和执行（通过 OrderManager + TqExecutor）")

    # ------------------------------------------------------------------
    # Step 9: 结果输出
    # ------------------------------------------------------------------
    logger.info("[Step 9] 输出结果摘要...")
    raise NotImplementedError("TODO: 打印各策略信号摘要、组合 Greeks、风控状态")

    logger.info("=" * 60)
    logger.info("策略运行完成")


if __name__ == "__main__":
    main()
