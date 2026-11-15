"""Standard Quant — hệ thống giao dịch có hệ thống theo kiến trúc 5 khối của Robert Carver.

Data Rules → Forecast Generation → Volatility Targeting → Portfolio Construction → Execution

Nạp LƯỜI (PEP 562): `import standard_quant.l5_execution` không được kéo theo pandas/numpy/sklearn.
"""
__all__ = ["Config", "StandardQuantPipeline", "run_backtest"]


def __getattr__(name: str):
    if name == "Config":
        from .config import Config
        return Config
    if name in ("StandardQuantPipeline", "run_backtest"):
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(f"module 'standard_quant' has no attribute {name!r}")
