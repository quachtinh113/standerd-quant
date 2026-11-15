"""Báo cáo tổng hợp: trạng thái tín hiệu, latency/TCA, rủi ro, cảnh báo → dict/JSON."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from ..l5_execution import store
from ..l5_execution.executor import latency_summary
from .health import Alert, HealthLimits, check_health


def build_report(con: sqlite3.Connection, ledger_path: str | Path | None = None, limits: HealthLimits | None = None,
                 expect_worker: bool = True) -> dict:
    alerts = check_health(con, limits, ledger_path, expect_worker=expect_worker)
    ladder = store.get_value(con, "ladder_state")
    hb = store.get_value(con, "heartbeat_ns")
    return {
        "generated_at": time.time(),
        "signals": store.status_counts(con),
        "latency": latency_summary(con),
        "halt": store.get_flag(con, "halt"),
        "reduce_only": store.get_flag(con, "reduce_only"),
        "ladder": json.loads(ladder) if ladder else None,
        "worker_heartbeat_age_s": None if hb is None else (time.time_ns() - int(hb)) / 1e9,
        "last_asof": store.get_value(con, "last_asof"),
        "alerts": [a.__dict__ for a in alerts],
        "status": "CRITICAL" if any(a.level == "CRITICAL" for a in alerts) else
                  "WARNING" if alerts else "OK",
    }


def write_report(rep: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    tmp.replace(p)                                     # ghi nguyên tử: người đọc không thấy file dở dang
