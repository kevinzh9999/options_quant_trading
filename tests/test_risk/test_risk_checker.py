"""
test_risk_checker.py
--------------------
测试风控检查模块（risk/risk_checker.py）。
"""

from __future__ import annotations

import pytest

from risk.risk_checker import CheckStatus, RiskChecker
from tests.conftest import make_test_config


class TestRiskCheckerMargin:

    def test_margin_pass(self):
        """保证金占用低于阈值时应通过"""
        checker = RiskChecker(make_test_config())
        result = checker.check_margin(
            account_balance=1_000_000,
            current_margin=200_000,      # 20%，低于 50% 阈值
            additional_margin=100_000,   # 新增 10%，合计 30%，仍低于阈值
        )
        assert result.passed
        assert result.status == CheckStatus.PASS

    def test_margin_warn(self):
        """保证金占用接近阈值（> 80%）时应为 WARN"""
        checker = RiskChecker(make_test_config())
        result = checker.check_margin(
            account_balance=1_000_000,
            current_margin=380_000,    # 38%
            additional_margin=60_000,  # 新增 6%，合计 44%，接近 50% 阈值
        )
        assert result.status == CheckStatus.WARN

    def test_margin_fail(self):
        """保证金占用超过阈值时应失败"""
        checker = RiskChecker(make_test_config())
        result = checker.check_margin(
            account_balance=1_000_000,
            current_margin=450_000,    # 45%
            additional_margin=100_000, # 新增 10%，合计 55%，超过 50% 阈值
        )
        assert not result.passed
        assert result.status == CheckStatus.FAIL


class TestRiskCheckerDailyLoss:

    def test_daily_loss_pass(self):
        """未达单日亏损限额时应通过"""
        checker = RiskChecker(make_test_config())
        result = checker.check_daily_loss(
            account_balance=1_000_000,
            daily_pnl=-10_000,  # -1%，低于 2% 限额
        )
        assert result.passed

    def test_daily_loss_fail(self):
        """超过单日亏损限额时应失败"""
        checker = RiskChecker(make_test_config())
        result = checker.check_daily_loss(
            account_balance=1_000_000,
            daily_pnl=-25_000,  # -2.5%，超过 2% 限额
        )
        assert not result.passed

    def test_daily_profit_always_pass(self):
        """当日盈利时应始终通过"""
        checker = RiskChecker(make_test_config())
        result = checker.check_daily_loss(
            account_balance=1_000_000,
            daily_pnl=50_000,  # 盈利
        )
        assert result.passed


class TestRiskCheckerLiquidity:

    def test_liquidity_pass(self):
        """成交量和持仓量充足时应通过"""
        checker = RiskChecker(make_test_config())
        result = checker.check_liquidity(
            ts_code="IO2406-C-3800.CFX",
            volume=5000, oi=20000, order_volume=2
        )
        assert result.passed

    def test_liquidity_fail_low_volume(self):
        """成交量过低时应失败"""
        checker = RiskChecker(make_test_config())
        result = checker.check_liquidity(
            ts_code="IO2406-C-3800.CFX",
            volume=50,   # 低于默认 min_volume=100
            oi=20000, order_volume=2
        )
        assert not result.passed
