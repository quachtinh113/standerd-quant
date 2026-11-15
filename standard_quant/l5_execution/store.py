"""Hộp thư tín hiệu (SQLite/WAL) — điểm giao DUY NHẤT giữa tầng tính toán (L1–L4) và L5.

Chỉ dùng thư viện chuẩn. Upstream ghi (`publish`), L5 đọc (`claim_next`). Mọi chuyển trạng thái
là nguyên tử; mỗi tín hiệu được xử lý tối đa MỘT lần (idempotent + claim) — thà bỏ lệnh còn hơn bắn đôi.

Trạng thái: PENDING → CLAIMED → SENT | REJECTED | FAILED | UNKNOWN ;  PENDING → EXPIRED
  UNKNOWN = đã claim nhưng không rõ kết quả (crash/timeout): KHÔNG tự gửi lại, chờ đối soát thủ công.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  signal_id TEXT NOT NULL UNIQUE,
  created_ns INTEGER NOT NULL,
  expires_ns INTEGER NOT NULL,
  symbol TEXT NOT NULL, side TEXT NOT NULL, volume REAL NOT NULL,
  sl REAL NOT NULL DEFAULT 0, tp REAL NOT NULL DEFAULT 0, ref_price REAL NOT NULL DEFAULT 0,
  reduce_only INTEGER NOT NULL DEFAULT 0, magic INTEGER NOT NULL DEFAULT 0, reason TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'PENDING', status_ns INTEGER NOT NULL, detail TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS ix_status ON signals(status, id);
CREATE TABLE IF NOT EXISTS latency(
  signal_id TEXT PRIMARY KEY,
  t_created_ns INTEGER, t_read_ns INTEGER, t_send_ns INTEGER, t_ack_ns INTEGER,
  ms_queue REAL, ms_roundtrip REAL, ms_total REAL,
  ref_price REAL, fill_price REAL, slippage_bps REAL, retcode INTEGER, ticket INTEGER, outcome TEXT
);
CREATE TABLE IF NOT EXISTS control(key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


@dataclass(frozen=True)
class Signal:
    signal_id: str
    symbol: str
    side: str                 # BUY | SELL
    volume: float
    sl: float = 0.0
    tp: float = 0.0
    ref_price: float = 0.0    # giá quyết định ở upstream (để đo trượt giá, không để tính toán)
    reduce_only: bool = False
    magic: int = 0
    reason: str = ""
    created_ns: int = 0
    expires_ns: int = 0


def connect(path: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(path), timeout=5.0, isolation_level=None)   # autocommit; ta tự BEGIN
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA busy_timeout=5000")
    con.executescript(SCHEMA)
    return con


def publish(con: sqlite3.Connection, sig: Signal, ttl_s: float = 30.0) -> bool:
    """Ghi tín hiệu. Trả False nếu signal_id đã tồn tại (không ghi đè)."""
    now = time.time_ns()
    created = sig.created_ns or now
    expires = sig.expires_ns or created + int(ttl_s * 1e9)
    cur = con.execute(
        "INSERT OR IGNORE INTO signals(signal_id,created_ns,expires_ns,symbol,side,volume,sl,tp,ref_price,"
        "reduce_only,magic,reason,status,status_ns) VALUES(?,?,?,?,?,?,?,?,?,?,?,?, 'PENDING', ?)",
        (sig.signal_id, created, expires, sig.symbol, sig.side, sig.volume, sig.sl, sig.tp, sig.ref_price,
         int(sig.reduce_only), sig.magic, sig.reason, now))
    return cur.rowcount == 1


def claim_next(con: sqlite3.Connection, now_ns: int | None = None) -> Signal | None:
    """Lấy tín hiệu PENDING cũ nhất còn hạn và đánh dấu CLAIMED trong một giao dịch nguyên tử.
    Tín hiệu hết hạn bị đánh dấu EXPIRED (không bao giờ bắn lệnh trên tín hiệu cũ)."""
    now = now_ns or time.time_ns()
    con.execute("BEGIN IMMEDIATE")
    try:
        while True:
            row = con.execute("SELECT * FROM signals WHERE status='PENDING' ORDER BY id LIMIT 1").fetchone()
            if row is None:
                con.execute("COMMIT")
                return None
            if row["expires_ns"] < now:
                con.execute("UPDATE signals SET status='EXPIRED', status_ns=?, detail='ttl' WHERE id=?", (now, row["id"]))
                continue
            con.execute("UPDATE signals SET status='CLAIMED', status_ns=? WHERE id=?", (now, row["id"]))
            con.execute("COMMIT")
            return Signal(row["signal_id"], row["symbol"], row["side"], row["volume"], row["sl"], row["tp"],
                          row["ref_price"], bool(row["reduce_only"]), row["magic"], row["reason"],
                          row["created_ns"], row["expires_ns"])
    except BaseException:
        con.execute("ROLLBACK")
        raise


def mark(con: sqlite3.Connection, signal_id: str, status: str, detail: str = "") -> None:
    con.execute("UPDATE signals SET status=?, status_ns=?, detail=? WHERE signal_id=?",
                (status, time.time_ns(), detail[:500], signal_id))


def recover_orphans(con: sqlite3.Connection) -> int:
    """Khi khởi động: mọi CLAIMED còn sót là kết quả không rõ → UNKNOWN. Không bao giờ tự gửi lại."""
    cur = con.execute("UPDATE signals SET status='UNKNOWN', status_ns=?, detail='orphan after restart' "
                      "WHERE status='CLAIMED'", (time.time_ns(),))
    return cur.rowcount


def set_flag(con: sqlite3.Connection, key: str, value: bool) -> None:
    con.execute("INSERT INTO control(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, "1" if value else "0"))


def get_flag(con: sqlite3.Connection, key: str) -> bool:
    row = con.execute("SELECT value FROM control WHERE key=?", (key,)).fetchone()
    return bool(row and row["value"] == "1")


def set_value(con: sqlite3.Connection, key: str, value: str) -> None:
    con.execute("INSERT INTO control(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value))


def get_value(con: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = con.execute("SELECT value FROM control WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def inflight_symbols(con: sqlite3.Connection) -> dict[str, str]:
    """Symbol còn tín hiệu chưa chốt (PENDING/CLAIMED) hoặc kết quả không rõ (UNKNOWN) → {symbol: trạng thái}.
    Upstream dùng để KHÔNG phát thêm tín hiệu cho symbol đó (tránh nhân đôi vị thế)."""
    rows = con.execute("SELECT symbol, status FROM signals WHERE status IN ('PENDING','CLAIMED','UNKNOWN') "
                       "ORDER BY id").fetchall()
    return {r["symbol"]: r["status"] for r in rows}


def status_counts(con: sqlite3.Connection) -> dict[str, int]:
    return {r["status"]: r["n"] for r in con.execute("SELECT status, COUNT(*) n FROM signals GROUP BY status")}
