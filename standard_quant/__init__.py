"""Standard Quant — hệ thống giao dịch có hệ thống theo kiến trúc 5 khối của Robert Carver.

Data Rules → Forecast Generation → Volatility Targeting → Portfolio Construction → Execution
"""
from .config import Config
from .pipeline import StandardQuantPipeline, run_backtest

__all__ = ["Config", "StandardQuantPipeline", "run_backtest"]
