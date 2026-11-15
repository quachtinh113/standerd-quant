"""Tầng L2 — Alpha (Finding Alphas): toán tử, thư viện, đánh giá chống overfit, cầu nối sang forecast."""
from . import operators
from .bridge import alpha_to_forecast, build_alpha_forecasts
from .evaluate import AlphaReport, evaluate_alpha, expected_max_sharpe, robustness_report
from .library import REGISTRY, register

__all__ = ["operators", "REGISTRY", "register", "alpha_to_forecast", "build_alpha_forecasts",
           "AlphaReport", "evaluate_alpha", "expected_max_sharpe", "robustness_report"]
