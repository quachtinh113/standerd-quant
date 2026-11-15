"""Sổ kiểm toán chuỗi băm (tamper-evident), JSONL. Chỉ thư viện chuẩn.

Bố cục trường như audit_risk_ledger.jsonl của 05_Quant (seq, prev_record_hash, record_hash = "sha256:…"),
nhưng thuật toán băm ở đây là của riêng Standard Quant (JSON chuẩn hoá, sort_keys) — KHÔNG chứng minh
tương thích byte với bộ kiểm của vibe_ledger. Không dùng chéo hai bộ kiểm.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

GENESIS = "sha256:genesis"


def _digest(rec: dict) -> str:
    body = {k: v for k, v in rec.items() if k != "record_hash"}
    return "sha256:" + hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


@dataclass
class VerifyResult:
    ok: bool
    count: int
    first_break: int | None = None
    reason: str = ""


class HashChainLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _tail(self) -> tuple[int, str]:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return 0, GENESIS
        last = None
        with open(self.path, "rb") as f:
            for line in f:
                if line.strip():
                    last = line
        rec = json.loads(last)
        return rec["seq"], rec["record_hash"]

    def append(self, event_type: str, payload: dict | None = None) -> dict:
        with self._lock:
            seq, prev = self._tail()
            rec = {"timestamp": datetime.now(timezone.utc).isoformat(), "event_type": event_type,
                   **(payload or {}), "seq": seq + 1, "prev_record_hash": prev}
            rec["record_hash"] = _digest(rec)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
                f.flush()
                os.fsync(f.fileno())
            return rec

    def verify(self) -> VerifyResult:
        if not self.path.exists():
            return VerifyResult(True, 0)
        prev, n = GENESIS, 0
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    return VerifyResult(False, n, n + 1, "json hỏng")
                n += 1
                if rec.get("seq") != n:
                    return VerifyResult(False, n, n, "seq không liên tục")
                if rec.get("prev_record_hash") != prev:
                    return VerifyResult(False, n, n, "đứt chuỗi băm")
                if rec.get("record_hash") != _digest(rec):
                    return VerifyResult(False, n, n, "nội dung bị sửa")
                prev = rec["record_hash"]
        return VerifyResult(True, n)
