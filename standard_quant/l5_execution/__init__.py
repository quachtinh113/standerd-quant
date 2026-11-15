"""L5 — Execution Isolation (Citadel-style): đọc tín hiệu → bắn lệnh → ghi latency. Không gì khác.

Ranh giới cứng (được test canh giữ, xem tests/test_l5_isolation.py):
  • chỉ import thư viện chuẩn + MetaTrader5 (lười, trong MT5Gateway);
  • không import numpy/pandas/sklearn/scipy, không import bất kỳ khối s1–s5 hay pipeline;
  • điểm giao với phần còn lại của hệ thống: bảng `signals` trong SQLite (store.py).
"""
from .executor import ExecLimits, Executor, latency_summary
from .gateway import FakeGateway, Gateway, MT5Gateway, SendResult
from .store import Signal, connect, publish, set_flag

__all__ = ["ExecLimits", "Executor", "latency_summary", "FakeGateway", "Gateway", "MT5Gateway", "SendResult",
           "Signal", "connect", "publish", "set_flag"]
