"""Executor — chỉ làm ĐÚNG MỘT việc: đọc tín hiệu → bắn lệnh → ghi latency.

KHÔNG có: chỉ báo, forecast, định cỡ vị thế, đọc dữ liệu giá lịch sử, pandas/numpy/sklearn.
Chỉ có: kiểm tra bất biến an toàn (so sánh/đối chiếu, không tính tín hiệu), gửi, ghi nhận.
"""
from __future__ import annotations

import sqlite3
import statistics
import time
from dataclasses import dataclass, field

from . import store
from .gateway import RETCODE_RETRYABLE, Gateway, SendResult
from .store import Signal


@dataclass
class ExecLimits:
    """Giới hạn cứng, cấu hình tĩnh — không suy ra từ dữ liệu thị trường."""
    allowed_symbols: frozenset[str] = frozenset()      # rỗng = từ chối tất cả (an toàn mặc định)
    max_lot_per_order: float = 1.0
    max_slippage_bps: float = 30.0                     # |giá live − ref_price| tối đa trước khi gửi (0 = tắt)
    require_stop_loss: bool = True
    max_retries: int = 1                               # chỉ cho retcode "chưa khớp" (requote/price changed)
    allowed_magics: frozenset[int] = frozenset()       # rỗng = không giới hạn


@dataclass
class Executor:
    con: sqlite3.Connection
    gateway: Gateway
    limits: ExecLimits
    stats: dict[str, int] = field(default_factory=dict)

    def _bump(self, k: str) -> None:
        self.stats[k] = self.stats.get(k, 0) + 1

    # ---- kiểm tra bất biến (từ chối, không sửa) ----
    def _veto(self, s: Signal) -> str | None:
        L = self.limits
        if s.side not in ("BUY", "SELL"):
            return "bad_side"
        if not s.volume > 0:
            return "bad_volume"
        if s.symbol not in L.allowed_symbols:
            return "symbol_not_allowed"
        if L.allowed_magics and s.magic not in L.allowed_magics:
            return "magic_not_allowed"
        if s.volume > L.max_lot_per_order:
            return "exceeds_max_lot"
        halted = store.get_flag(self.con, "halt")
        if halted and not s.reduce_only:
            return "halted"
        if store.get_flag(self.con, "reduce_only") and not s.reduce_only:
            return "reduce_only_mode"
        if not s.reduce_only and L.require_stop_loss:
            if s.sl <= 0:
                return "missing_stop_loss"
            if s.ref_price > 0 and ((s.side == "BUY" and s.sl >= s.ref_price) or (s.side == "SELL" and s.sl <= s.ref_price)):
                return "stop_loss_wrong_side"
        return None

    def process(self, s: Signal, t_read_ns: int) -> str:
        veto = self._veto(s)
        if veto:
            store.mark(self.con, s.signal_id, "REJECTED", veto)
            self._record(s, t_read_ns, 0, 0, SendResult(False, comment=veto), "REJECTED:" + veto)
            self._bump("rejected")
            return "REJECTED"

        if self.limits.max_slippage_bps > 0 and s.ref_price > 0 and not s.reduce_only:
            live = self.gateway.last_price(s.symbol, s.side)
            if live > 0 and abs(live - s.ref_price) / s.ref_price * 1e4 > self.limits.max_slippage_bps:
                store.mark(self.con, s.signal_id, "REJECTED", "price_moved")
                self._record(s, t_read_ns, 0, 0, SendResult(False, comment="price_moved"), "REJECTED:price_moved")
                self._bump("rejected")
                return "REJECTED"

        res, t_send, t_ack = SendResult(False), 0, 0
        for attempt in range(self.limits.max_retries + 1):
            t_send = time.perf_counter_ns()
            try:
                res = self.gateway.send(s)
            except Exception as e:                          # ngoại lệ giữa chừng = kết quả không rõ
                res = SendResult(False, 0, 0, 0.0, f"exception:{type(e).__name__}", unknown=True)
            t_ack = time.perf_counter_ns()
            if res.ok or res.unknown or res.retcode not in RETCODE_RETRYABLE:
                break
        if res.ok:
            status, outcome = "SENT", "SENT"
        elif res.unknown:
            status, outcome = "UNKNOWN", "UNKNOWN"          # không retry, chờ đối soát
        else:
            status, outcome = "FAILED", f"FAILED:{res.retcode}"
        store.mark(self.con, s.signal_id, status, f"{res.retcode} {res.comment}")
        self._record(s, t_read_ns, t_send, t_ack, res, outcome)
        self._bump(status.lower())
        return status

    def _record(self, s: Signal, t_read_ns: int, t_send: int, t_ack: int, res: SendResult, outcome: str) -> None:
        roundtrip_ms = (t_ack - t_send) / 1e6 if t_ack and t_send else 0.0
        queue_ms = (t_read_ns - s.created_ns) / 1e6
        slip = 0.0
        if res.ok and s.ref_price > 0 and res.fill_price > 0:
            sign = 1 if s.side == "BUY" else -1
            slip = sign * (res.fill_price - s.ref_price) / s.ref_price * 1e4       # + = bất lợi
        self.con.execute(
            "INSERT OR REPLACE INTO latency VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (s.signal_id, s.created_ns, t_read_ns, t_send, t_ack, queue_ms, roundtrip_ms, queue_ms + roundtrip_ms,
             s.ref_price, res.fill_price, slip, res.retcode, res.ticket, outcome))

    def run_once(self) -> int:
        """Xử lý hết hàng đợi hiện có. Trả số tín hiệu đã xử lý."""
        n = 0
        while (s := store.claim_next(self.con)) is not None:
            self.process(s, time.time_ns())
            n += 1
        return n

    def run_forever(self, poll_ms: int = 100, should_stop=lambda: False) -> None:
        self.gateway.connect()
        orphans = store.recover_orphans(self.con)
        if orphans:
            self._bump("orphans_unknown")
        while not should_stop():
            store.set_value(self.con, "heartbeat_ns", str(time.time_ns()))      # để L6 biết worker còn sống
            if self.run_once() == 0:
                time.sleep(poll_ms / 1000.0)


def latency_summary(con: sqlite3.Connection) -> dict[str, float]:
    """p50/p95/p99 của ms_total, ms_roundtrip và trượt giá trung bình trên các lệnh đã gửi."""
    rows = con.execute("SELECT ms_queue, ms_roundtrip, ms_total, slippage_bps FROM latency WHERE outcome='SENT'").fetchall()
    if not rows:
        return {"n": 0}

    def pct(xs: list[float], q: float) -> float:
        xs = sorted(xs)
        return xs[min(int(q * len(xs)), len(xs) - 1)]

    tot, rt, qu = [r["ms_total"] for r in rows], [r["ms_roundtrip"] for r in rows], [r["ms_queue"] for r in rows]
    return {"n": len(rows), "queue_p50_ms": pct(qu, .5), "roundtrip_p50_ms": pct(rt, .5), "roundtrip_p95_ms": pct(rt, .95),
            "total_p50_ms": pct(tot, .5), "total_p95_ms": pct(tot, .95), "total_p99_ms": pct(tot, .99),
            "slippage_bps_mean": statistics.fmean(r["slippage_bps"] for r in rows)}
