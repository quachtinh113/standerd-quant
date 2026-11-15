"""L6 — kiểm tra sức khoẻ hệ thống từ hộp thư/latency/sổ kiểm toán. Chỉ đọc; chỉ thư viện chuẩn."""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from ..ledger import HashChainLedger
from ..l5_execution import store
from ..l5_execution.executor import latency_summary


@dataclass(frozen=True)
class Alert:
    level: str            # INFO | WARNING | CRITICAL
    code: str
    message: str


@dataclass
class HealthLimits:
    heartbeat_max_age_s: float = 30.0
    pending_max_age_s: float = 60.0
    max_roundtrip_p95_ms: float = 1500.0
    max_slippage_bps_mean: float = 5.0
    max_reject_rate: float = 0.30
    min_orders_for_rates: int = 10


def check_health(con: sqlite3.Connection, limits: HealthLimits | None = None, ledger_path: str | Path | None = None,
                 now_ns: int | None = None, expect_worker: bool = True) -> list[Alert]:
    L, now, out = limits or HealthLimits(), now_ns or time.time_ns(), []

    hb = store.get_value(con, "heartbeat_ns")
    if expect_worker:
        if hb is None:
            out.append(Alert("CRITICAL", "NO_HEARTBEAT", "worker L5 chưa từng báo nhịp"))
        elif (now - int(hb)) / 1e9 > L.heartbeat_max_age_s:
            out.append(Alert("CRITICAL", "WORKER_STALE", f"worker L5 im lặng {(now - int(hb)) / 1e9:.0f}s"))

    counts = store.status_counts(con)
    if counts.get("UNKNOWN", 0):
        out.append(Alert("CRITICAL", "UNKNOWN_ORDERS", f"{counts['UNKNOWN']} lệnh không rõ kết quả — cần đối soát thủ công"))
    old = con.execute("SELECT COUNT(*) n FROM signals WHERE status IN ('PENDING','CLAIMED') AND created_ns < ?",
                      (now - int(L.pending_max_age_s * 1e9),)).fetchone()["n"]
    if old:
        out.append(Alert("WARNING", "STUCK_SIGNALS", f"{old} tín hiệu tồn đọng > {L.pending_max_age_s:.0f}s"))

    done = sum(counts.get(k, 0) for k in ("SENT", "REJECTED", "FAILED"))
    if done >= L.min_orders_for_rates:
        rate = (counts.get("REJECTED", 0) + counts.get("FAILED", 0)) / done
        if rate > L.max_reject_rate:
            out.append(Alert("WARNING", "HIGH_REJECT_RATE", f"tỉ lệ từ chối/thất bại {rate:.0%}"))
    lat = latency_summary(con)
    if lat.get("n", 0) >= L.min_orders_for_rates:
        if lat["roundtrip_p95_ms"] > L.max_roundtrip_p95_ms:
            out.append(Alert("WARNING", "HIGH_LATENCY", f"p95 khứ hồi {lat['roundtrip_p95_ms']:.0f}ms"))
        if lat["slippage_bps_mean"] > L.max_slippage_bps_mean:
            out.append(Alert("WARNING", "HIGH_SLIPPAGE", f"trượt giá TB {lat['slippage_bps_mean']:.1f}bps"))

    if store.get_flag(con, "halt"):
        out.append(Alert("CRITICAL", "HALT_ACTIVE", "cờ halt đang bật: chỉ cho phép đóng vị thế"))
    if ledger_path is not None:
        v = HashChainLedger(ledger_path).verify()
        if not v.ok:
            out.append(Alert("CRITICAL", "LEDGER_TAMPERED", f"sổ kiểm toán hỏng tại #{v.first_break}: {v.reason}"))
    return out


def position_drift(targets: dict[str, float], actual: dict[str, float], tol_lots: float = 0.011) -> list[Alert]:
    """So vị thế mục tiêu (do pipeline tính) với vị thế thực ở broker; lệch quá dung sai = cảnh báo."""
    out = []
    for s in sorted(set(targets) | set(actual)):
        diff = actual.get(s, 0.0) - targets.get(s, 0.0)
        if abs(diff) > tol_lots:
            out.append(Alert("WARNING", "POSITION_DRIFT", f"{s}: thực {actual.get(s, 0.0):+.2f} vs mục tiêu {targets.get(s, 0.0):+.2f}"))
    return out


def unknown_for_review(con: sqlite3.Connection, broker_net: dict[str, float]) -> list[dict]:
    """Danh sách lệnh UNKNOWN kèm vị thế ròng hiện có ở broker, để NGƯỜI quyết định. Không tự phán xử."""
    rows = con.execute("SELECT signal_id, symbol, side, volume, reduce_only, detail FROM signals WHERE status='UNKNOWN'").fetchall()
    return [{**dict(r), "broker_net_lots": broker_net.get(r["symbol"], 0.0)} for r in rows]


def resolve_unknown(con: sqlite3.Connection, signal_id: str, outcome: str, note: str,
                    ledger: HashChainLedger | None = None) -> None:
    """Người xác nhận kết quả: outcome ∈ {'SENT','NOT_SENT'}. NOT_SENT → REJECTED (không gửi lại tự động)."""
    if outcome not in ("SENT", "NOT_SENT"):
        raise ValueError("outcome phải là SENT hoặc NOT_SENT")
    row = con.execute("SELECT status FROM signals WHERE signal_id=?", (signal_id,)).fetchone()
    if row is None or row["status"] != "UNKNOWN":
        raise ValueError("chỉ giải quyết được tín hiệu đang UNKNOWN")
    store.mark(con, signal_id, "SENT" if outcome == "SENT" else "REJECTED", f"manual:{outcome}:{note}")
    if ledger:
        ledger.append("UNKNOWN_RESOLVED", {"signal_id": signal_id, "outcome": outcome, "note": note})
