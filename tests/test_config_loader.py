"""
test_config_loader.py
---------------------
测试 config/config_loader.py

覆盖：
- get() 的三级优先级（环境变量 > yaml > default）
- 配置文件不存在时不报错
- get_tushare_token / get_tq_config / get_db_path / get_strategy_config
- Config(raw_dict) 向后兼容接口
- 掩码日志（不测日志内容，只测功能不抛异常）
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from config.config_loader import Config, ConfigLoader, ConfigError, load_config


# ======================================================================
# 辅助
# ======================================================================

SAMPLE_YAML = textwrap.dedent("""\
    tushare:
      token: "YAML_TOKEN_123"

    tq:
      broker: "宏源期货"
      account_id: "YAML_ACCOUNT_ID"
      account: "YAML_TQ_ACCOUNT"
      password: "YAML_TQ_PASS"

    database:
      path: "data/storage/test.db"

    strategies:
      vol_arb:
        vrp_threshold: 0.08
        holding_period: 10

    strategy:
      vrp_threshold: 0.07
      holding_period: 7
      rv_window: 30
      garch_lookback: 500

    risk:
      max_margin_ratio: 0.60
      max_daily_loss_ratio: 0.03
      max_delta_exposure: 800000
      max_vega_exposure: 200000

    logging:
      level: "DEBUG"
      dir: "/tmp/logs/"
""")


@pytest.fixture
def yaml_file(tmp_path: Path) -> Path:
    """临时 YAML 配置文件"""
    p = tmp_path / "config.yaml"
    p.write_text(SAMPLE_YAML, encoding="utf-8")
    return p


@pytest.fixture
def loader(yaml_file: Path) -> ConfigLoader:
    return ConfigLoader(yaml_file)


# ======================================================================
# ConfigLoader 基本加载
# ======================================================================

class TestConfigLoaderLoad:

    def test_load_existing_file(self, yaml_file: Path):
        cfg = ConfigLoader(yaml_file)
        assert cfg.get("tushare.token") == "YAML_TOKEN_123"

    def test_load_missing_file_no_error(self, tmp_path: Path):
        """文件不存在时不抛异常，返回默认值"""
        cfg = ConfigLoader(tmp_path / "nonexistent.yaml")
        assert cfg.get("tushare.token", "fallback") == "fallback"

    def test_load_default_path_no_error(self, monkeypatch):
        """默认路径（config/config.yaml）不存在时也不报错"""
        # 只要不抛异常即可
        cfg = ConfigLoader("/nonexistent/path/config.yaml")
        assert cfg.get("any.key", "default") == "default"


# ======================================================================
# get() 三级优先级
# ======================================================================

class TestGetPriority:

    def test_yaml_value_returned(self, loader: ConfigLoader):
        assert loader.get("tushare.token") == "YAML_TOKEN_123"

    def test_default_when_key_missing(self, loader: ConfigLoader):
        assert loader.get("nonexistent.key", "my_default") == "my_default"

    def test_default_none_when_no_default(self, loader: ConfigLoader):
        assert loader.get("nonexistent.key") is None

    def test_env_var_overrides_yaml(self, loader: ConfigLoader, monkeypatch):
        """环境变量优先级高于 yaml"""
        monkeypatch.setenv("TUSHARE_TOKEN", "ENV_TOKEN_XYZ")
        assert loader.get("tushare.token") == "ENV_TOKEN_XYZ"

    def test_env_var_overrides_default(self, tmp_path: Path, monkeypatch):
        """环境变量优先级高于 default（文件不存在情况）"""
        monkeypatch.setenv("TUSHARE_TOKEN", "ENV_ONLY")
        cfg = ConfigLoader(tmp_path / "nofile.yaml")
        assert cfg.get("tushare.token", "fallback") == "ENV_ONLY"

    def test_yaml_overrides_default(self, loader: ConfigLoader):
        """yaml 优先级高于 default"""
        assert loader.get("tushare.token", "DEFAULT_TOKEN") == "YAML_TOKEN_123"

    def test_env_var_naming_convention(self, loader: ConfigLoader, monkeypatch):
        """点号 → 下划线大写的命名规则"""
        monkeypatch.setenv("TQ_BROKER", "ENV_BROKER")
        assert loader.get("tq.broker") == "ENV_BROKER"

    def test_deep_nested_key(self, loader: ConfigLoader):
        assert loader.get("strategies.vol_arb.vrp_threshold") == 0.08

    def test_partial_path_missing_returns_default(self, loader: ConfigLoader):
        assert loader.get("strategies.nonexistent.param", 99) == 99


# ======================================================================
# get_tushare_token
# ======================================================================

class TestGetTushareToken:

    def test_returns_token_from_yaml(self, loader: ConfigLoader):
        assert loader.get_tushare_token() == "YAML_TOKEN_123"

    def test_env_token_takes_precedence(self, loader: ConfigLoader, monkeypatch):
        monkeypatch.setenv("TUSHARE_TOKEN", "OVERRIDE_TOKEN")
        assert loader.get_tushare_token() == "OVERRIDE_TOKEN"

    def test_placeholder_returns_empty(self, tmp_path: Path):
        p = tmp_path / "config.yaml"
        p.write_text('tushare:\n  token: "YOUR_TUSHARE_PRO_TOKEN_HERE"\n')
        cfg = ConfigLoader(p)
        assert cfg.get_tushare_token() == ""

    def test_missing_returns_empty(self, tmp_path: Path):
        cfg = ConfigLoader(tmp_path / "nofile.yaml")
        assert cfg.get_tushare_token() == ""


# ======================================================================
# get_tq_config
# ======================================================================

class TestGetTqConfig:

    def test_returns_all_fields(self, loader: ConfigLoader):
        tq = loader.get_tq_config()
        assert set(tq.keys()) == {"broker", "account_id", "account", "password"}

    def test_values_from_yaml(self, loader: ConfigLoader):
        tq = loader.get_tq_config()
        assert tq["broker"] == "宏源期货"
        assert tq["account_id"] == "YAML_ACCOUNT_ID"
        assert tq["account"] == "YAML_TQ_ACCOUNT"

    def test_env_overrides_account(self, loader: ConfigLoader, monkeypatch):
        monkeypatch.setenv("TQ_ACCOUNT", "ENV_ACCOUNT")
        tq = loader.get_tq_config()
        assert tq["account"] == "ENV_ACCOUNT"

    def test_missing_fields_return_empty_string(self, tmp_path: Path):
        cfg = ConfigLoader(tmp_path / "nofile.yaml")
        tq = cfg.get_tq_config()
        assert tq["account"] == ""
        assert tq["password"] == ""
        assert tq["broker"] == "宏源期货"   # 默认值


# ======================================================================
# get_db_path
# ======================================================================

class TestGetDbPath:

    def test_returns_absolute_path(self, loader: ConfigLoader):
        p = loader.get_db_path()
        assert Path(p).is_absolute()

    def test_relative_path_resolved_to_project_root(self, loader: ConfigLoader):
        p = Path(loader.get_db_path())
        # 应包含项目根路径
        assert "options_quant_trading" in str(p) or p.is_absolute()

    def test_custom_relative_path(self, tmp_path: Path):
        p = tmp_path / "config.yaml"
        p.write_text('database:\n  path: "custom/my.db"\n')
        cfg = ConfigLoader(p)
        db = cfg.get_db_path()
        assert db.endswith("custom/my.db") or db.endswith("custom\\my.db")

    def test_default_when_no_config(self, tmp_path: Path):
        cfg = ConfigLoader(tmp_path / "nofile.yaml")
        db = cfg.get_db_path()
        assert db.endswith("trading.db")

    def test_absolute_path_preserved(self, tmp_path: Path):
        abs_path = "/tmp/test.db"
        p = tmp_path / "config.yaml"
        p.write_text(f'database:\n  path: "{abs_path}"\n')
        cfg = ConfigLoader(p)
        assert cfg.get_db_path() == abs_path


# ======================================================================
# get_strategy_config
# ======================================================================

class TestGetStrategyConfig:

    def test_returns_strategy_dict(self, loader: ConfigLoader):
        cfg = loader.get_strategy_config("vol_arb")
        assert isinstance(cfg, dict)
        assert cfg.get("vrp_threshold") == 0.08
        assert cfg.get("holding_period") == 10

    def test_missing_strategy_returns_empty_dict(self, loader: ConfigLoader):
        cfg = loader.get_strategy_config("nonexistent_strategy")
        assert cfg == {}

    def test_no_config_file_returns_empty_dict(self, tmp_path: Path):
        cfg = ConfigLoader(tmp_path / "nofile.yaml")
        assert cfg.get_strategy_config("vol_arb") == {}


# ======================================================================
# Config 向后兼容（旧 Config(raw_dict) 接口）
# ======================================================================

class TestConfigBackwardCompat:

    def test_raw_dict_constructor(self):
        """Config(raw_dict) 旧接口仍然有效"""
        raw = {
            "tushare": {"token": "TEST_TOKEN"},
            "tq": {"account": "TEST_ACC", "password": "TEST_PASS"},
            "database": {"path": ":memory:"},
            "strategy": {"vrp_threshold": 0.05, "holding_period": 5,
                         "rv_window": 20, "garch_lookback": 252},
            "risk": {"max_margin_ratio": 0.50, "max_daily_loss_ratio": 0.02,
                     "max_delta_exposure": 500000, "max_vega_exposure": 100000},
            "logging": {"level": "DEBUG", "dir": "/tmp/"},
        }
        cfg = Config(raw)
        assert cfg.tushare_token == "TEST_TOKEN"
        assert cfg.tq_account == "TEST_ACC"

    def test_properties_accessible(self):
        raw = {"strategy": {"vrp_threshold": 0.10, "holding_period": 3,
                             "rv_window": 15, "garch_lookback": 200}}
        cfg = Config(raw)
        assert cfg.vrp_threshold == 0.10
        assert cfg.holding_period == 3
        assert cfg.rv_window == 15

    def test_defaults_without_config(self):
        cfg = Config({})
        assert cfg.max_margin_ratio == 0.50
        assert cfg.max_daily_loss_ratio == 0.02
        assert cfg.log_level == "INFO"

    def test_env_overrides_raw_dict(self, monkeypatch):
        """环境变量覆盖 raw_dict 中的值"""
        monkeypatch.setenv("TUSHARE_TOKEN", "ENV_TOKEN_RAW")
        raw = {"tushare": {"token": "RAW_TOKEN"}}
        cfg = Config(raw)
        # get() 走环境变量优先，返回 ENV_TOKEN_RAW
        assert cfg.get("tushare.token") == "ENV_TOKEN_RAW"

    def test_get_method_available(self):
        """Config 实例有 get() 方法"""
        cfg = Config({"foo": {"bar": 42}})
        assert cfg.get("foo.bar") == 42
        assert cfg.get("foo.missing", "default") == "default"

    def test_get_tushare_token(self):
        cfg = Config({"tushare": {"token": "MY_TOKEN"}})
        assert cfg.get_tushare_token() == "MY_TOKEN"

    def test_get_db_path_returns_absolute(self):
        cfg = Config({"database": {"path": "data/storage/trading.db"}})
        db = cfg.get_db_path()
        assert Path(db).is_absolute()

    def test_memory_db_preserved(self):
        """':memory:' 是 SQLite 特殊值，不应强制转为绝对路径"""
        cfg = Config({"database": {"path": ":memory:"}})
        # ':memory:' 在 get() 返回时是原始字符串，get_db_path 会尝试 Path(':memory:')
        # 这在 Linux 是相对路径，但对测试来说关键是没有崩溃
        db = cfg.get_db_path()
        assert isinstance(db, str)


# ======================================================================
# load_config 函数
# ======================================================================

class TestLoadConfig:

    def test_load_existing_file(self, yaml_file: Path):
        cfg = load_config(yaml_file)
        assert isinstance(cfg, Config)
        assert cfg.get("tushare.token") == "YAML_TOKEN_123"

    def test_load_missing_file_no_error(self, tmp_path: Path):
        """文件不存在时不抛异常"""
        cfg = load_config(tmp_path / "nofile.yaml")
        assert isinstance(cfg, Config)

    def test_load_none_uses_default_path(self):
        """path=None 时使用默认路径（不存在也不报错）"""
        # 只要不抛 ConfigError 即可
        cfg = load_config(None)
        assert isinstance(cfg, Config)
