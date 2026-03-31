"""
conftest.py
-----------
pytest 共享 fixtures 和测试工具函数。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import Config


def make_test_config() -> Config:
    """
    创建用于测试的 Config 对象（使用测试占位符，不读取真实配置文件）。
    """
    raw = {
        "tushare": {"token": "TEST_TOKEN"},
        "tq": {"account": "TEST_ACCOUNT", "password": "TEST_PASSWORD"},
        "database": {"path": ":memory:"},  # SQLite 内存数据库
        "strategy": {
            "vrp_threshold": 0.05,
            "holding_period": 5,
            "rv_window": 20,
            "garch_lookback": 252,
        },
        "risk": {
            "max_margin_ratio": 0.50,
            "max_daily_loss_ratio": 0.02,
            "max_delta_exposure": 500000,
            "max_vega_exposure": 100000,
        },
        "logging": {"level": "DEBUG", "dir": "/tmp/test_logs/"},
    }
    # 绕过 _validate（测试用 token 是占位符）
    cfg = Config(raw)
    return cfg


@pytest.fixture
def test_config() -> Config:
    """pytest fixture：返回测试配置"""
    return make_test_config()
