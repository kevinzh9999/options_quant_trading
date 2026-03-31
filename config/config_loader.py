"""
config_loader.py
----------------
职责：加载并管理系统配置。

优先级：环境变量 > config.yaml > 默认值

- 配置文件不存在时不报错，使用环境变量和内置默认值继续运行
- 点号分隔路径访问（如 "tushare.token"）
- 环境变量命名规则：点号转下划线后大写（"tushare.token" → "TUSHARE_TOKEN"）
- 敏感信息在日志中打印掩码（只显示前4位）
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# 项目根目录（config/ 的上级）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 自动加载项目根目录的 .env 文件（不存在时静默跳过）
load_dotenv(_PROJECT_ROOT / ".env")

# 默认配置文件路径
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"

# 敏感字段列表（在日志中打码）
_SENSITIVE_KEYS = {"tushare.token", "tq.account", "tq.password", "tq.broker_password"}


class ConfigError(Exception):
    """配置错误基类"""


# ======================================================================
# ConfigLoader — 主配置类
# ======================================================================

class ConfigLoader:
    """
    配置加载器。

    优先级：环境变量 > config.yaml > 传入的 default。
    配置文件不存在时不报错，使用环境变量和内置默认值继续运行。

    Parameters
    ----------
    config_path : str | Path | None
        配置文件路径，默认为项目根目录下的 config/config.yaml
    """

    def __init__(self, config_path: Optional[str | Path] = None) -> None:
        self._path: Optional[Path] = (
            Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        )
        self._raw: dict[str, Any] = {}
        self._load()

    # ------------------------------------------------------------------
    # 文件加载
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """从配置文件加载 YAML，文件不存在时静默继续。"""
        if self._path and self._path.exists():
            with self._path.open("r", encoding="utf-8") as f:
                self._raw = yaml.safe_load(f) or {}
            logger.debug("配置文件已加载: %s", self._path)
        else:
            if self._path:
                logger.debug(
                    "配置文件不存在 (%s)，使用环境变量和默认值", self._path
                )
            self._raw = {}

    # ------------------------------------------------------------------
    # 核心 get()
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """
        通过点号分隔路径获取配置值。

        优先级：
        1. 环境变量（key 转大写，点号变下划线："tushare.token" → "TUSHARE_TOKEN"）
        2. config.yaml 中的值
        3. 传入的 default

        Parameters
        ----------
        key : str
            点分路径，如 "tushare.token" 或 "strategies.vol_arb.vrp_threshold"
        default : Any
            无法从环境变量或文件中找到时的默认值

        Returns
        -------
        Any
            配置值
        """
        # 1. 环境变量优先
        env_key = key.upper().replace(".", "_")
        env_val = os.environ.get(env_key)
        if env_val is not None:
            if key in _SENSITIVE_KEYS:
                logger.debug(
                    "从环境变量 %s 读取敏感配置: %s***",
                    env_key,
                    str(env_val)[:4],
                )
            return env_val

        # 2. YAML 文件
        node: Any = self._raw
        for k in key.split("."):
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    # ------------------------------------------------------------------
    # 业务 getter
    # ------------------------------------------------------------------

    def get_tushare_token(self) -> str:
        """
        获取 Tushare Token。

        优先从环境变量 TUSHARE_TOKEN 读取，其次读配置文件。
        返回空字符串而不是抛出异常（允许无 token 启动）。
        """
        token: str = self.get("tushare.token", "")
        if not token or token.startswith("YOUR_"):
            return ""
        if os.environ.get("TUSHARE_TOKEN"):
            logger.info("Tushare token 来自环境变量: %s***", str(token)[:4])
        return token

    def get_tq_config(self) -> Dict[str, str]:
        """
        获取天勤配置。

        Returns
        -------
        dict
            account       : 天勤平台账户（TqAuth email）
            password      : 天勤平台密码（TqAuth password）
            broker        : 期货公司代码，宏源期货为 "H宏源期货"（TqAccount broker_id）
            account_id    : 期货公司资金账号（TqAccount account_id）
            broker_password: 期货账户密码（TqAccount password，env: TQ_BROKER_PASSWORD）
        """
        cfg = {
            "broker":          str(self.get("tq.broker",          "H宏源期货")),
            "account_id":      str(self.get("tq.account_id",      "")),
            "account":         str(self.get("tq.account",         "")),
            "password":        str(self.get("tq.password",        "")),
            "broker_password": str(self.get("tq.broker_password", "")),
        }
        # 掩码日志
        if cfg["account"] and not cfg["account"].startswith("YOUR_"):
            logger.debug("TQ account: %s***", cfg["account"][:4])
        return cfg

    def get_db_path(self) -> str:
        """
        获取数据库文件路径（绝对路径）。

        默认为项目根目录下的 data/storage/trading.db。
        若配置文件中是相对路径，则相对于项目根目录解析。
        """
        raw_path: str = self.get("database.path", "data/storage/trading.db")
        p = Path(raw_path)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        return str(p)

    def get_tmp_dir(self) -> str:
        """项目级 tmp 目录，所有运行时临时文件放这里。"""
        tmp_dir = _PROJECT_ROOT / "tmp"
        tmp_dir.mkdir(exist_ok=True)
        return str(tmp_dir)

    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """
        获取指定策略的完整配置字典。

        Parameters
        ----------
        strategy_name : str
            策略名，如 "vol_arb"、"trend_following"

        Returns
        -------
        dict
            策略配置字典，不存在时返回空字典
        """
        cfg = self.get(f"strategies.{strategy_name}", {})
        return cfg if isinstance(cfg, dict) else {}

    # ------------------------------------------------------------------
    # 便捷属性（与 Config 子类共用）
    # ------------------------------------------------------------------

    @property
    def tushare_token(self) -> str:
        return self.get_tushare_token()

    @property
    def tq_account(self) -> str:
        val: str = self.get("tq.account", "")
        return val if not str(val).startswith("YOUR_") else ""

    @property
    def tq_password(self) -> str:
        val: str = self.get("tq.password", "")
        return val if not str(val).startswith("YOUR_") else ""

    @property
    def tq_broker(self) -> str:
        return str(self.get("tq.broker", "H宏源期货"))

    @property
    def tq_broker_password(self) -> str:
        val: str = self.get("tq.broker_password", "")
        return val if not str(val).startswith("YOUR_") else ""

    @property
    def tq_account_id(self) -> str:
        return str(self.get("tq.account_id", ""))

    @property
    def db_path(self) -> str:
        # 保留旧属性名用于向后兼容
        return self.get_db_path()

    @property
    def vrp_threshold(self) -> float:
        return float(self.get("strategy.vrp_threshold", 0.05))

    @property
    def holding_period(self) -> int:
        return int(self.get("strategy.holding_period", 5))

    @property
    def rv_window(self) -> int:
        return int(self.get("strategy.rv_window", 20))

    @property
    def garch_lookback(self) -> int:
        return int(self.get("strategy.garch_lookback", 252))

    @property
    def max_margin_ratio(self) -> float:
        return float(self.get("risk.max_margin_ratio", 0.50))

    @property
    def max_daily_loss_ratio(self) -> float:
        return float(self.get("risk.max_daily_loss_ratio", 0.02))

    @property
    def max_delta_exposure(self) -> float:
        return float(self.get("risk.max_delta_exposure", 500_000))

    @property
    def max_vega_exposure(self) -> float:
        return float(self.get("risk.max_vega_exposure", 100_000))

    @property
    def log_level(self) -> str:
        return str(self.get("logging.level", "INFO"))

    @property
    def log_dir(self) -> str:
        return str(self.get("logging.dir", "logs/"))

    def __repr__(self) -> str:
        source = str(self._path) if self._path else "(内存)"
        return f"ConfigLoader(source={source!r})"


# ======================================================================
# Config — 向后兼容子类
# ======================================================================

class Config(ConfigLoader):
    """
    向后兼容的配置类。

    支持两种构造方式：
    1. ``Config(raw_dict)``  — 直接传入原始字典（测试/内嵌使用）
    2. ``Config(config_path)`` 或 ``Config()`` — 从文件加载（生产使用）

    继承 ConfigLoader 的全部接口。
    """

    def __init__(
        self,
        raw_or_path: dict[str, Any] | str | Path | None = None,
    ) -> None:
        if isinstance(raw_or_path, dict):
            # 旧接口：直接注入原始字典
            self._path = None
            self._raw = dict(raw_or_path)
            self._apply_env_overrides()
        else:
            super().__init__(raw_or_path)

    def _apply_env_overrides(self) -> None:
        """旧接口专用：将特定环境变量写入 _raw（保持旧行为）。"""
        _ENV_MAP = {
            "TUSHARE_TOKEN":      ("tushare", "token"),
            "TQ_ACCOUNT":         ("tq",      "account"),
            "TQ_PASSWORD":        ("tq",      "password"),
            "TQ_BROKER":          ("tq",      "broker"),
            "TQ_ACCOUNT_ID":      ("tq",      "account_id"),
            "TQ_BROKER_PASSWORD": ("tq",      "broker_password"),
        }
        for env_key, (section, field) in _ENV_MAP.items():
            val = os.environ.get(env_key)
            if val:
                self._raw.setdefault(section, {})[field] = val

    # 保留旧的 _get 以防止有代码直接调用它
    def _get(self, dotted_key: str, default: Any = None) -> Any:
        return self.get(dotted_key, default)


# ======================================================================
# 公开加载函数
# ======================================================================

def load_config(path: str | Path | None = None) -> Config:
    """
    加载配置文件并返回 Config 实例。

    配置文件不存在时不报错，仅使用环境变量和内置默认值。
    调用方可用 ``cfg.get_tushare_token()`` 检查 token 是否已设置。

    Parameters
    ----------
    path : str | Path | None
        配置文件路径，默认读取 config/config.yaml

    Returns
    -------
    Config
        已加载的配置对象
    """
    return Config(path)
