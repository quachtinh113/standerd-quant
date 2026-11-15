"""Canh giữ ranh giới L5: cấm tính toán chỉ báo/ML trong tầng thực thi + hành vi an toàn của executor."""
import ast
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
L5 = ROOT / "standard_quant" / "l5_execution"

from standard_quant.l5_execution import ExecLimits, Executor, FakeGateway, Signal, connect, latency_summary, publish, set_flag
from standard_quant.l5_execution import store

ALLOWED_TOP = {"__future__", "sqlite3", "time", "dataclasses", "pathlib", "typing", "statistics", "argparse", "json",
               "MetaTrader5"}
FORBIDDEN_NAMES = {"numpy", "pandas", "sklearn", "scipy", "talib", "ta", "statsmodels", "torch"}
LIMITS = ExecLimits(frozenset({"XAUUSD"}), max_lot_per_order=1.0, max_slippage_bps=30.0, allowed_magics=frozenset({7}))


# ---------------- Cách ly tĩnh/động ----------------
def _imports(path: Path) -> list[tuple[str, int]]:
    out = []
    for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(n, ast.Import):
            out += [(a.name, n.lineno) for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            out.append((("." * n.level) + (n.module or ""), n.lineno))
    return out


def test_l5_imports_only_stdlib_and_itself():
    for f in L5.glob("*.py"):
        for mod, line in _imports(f):
            if mod.startswith("."):
                assert mod.lstrip(".").split(".")[0] in {"store", "gateway", "executor", ""} and mod.count(".") == 1, \
                    f"{f.name}:{line} import tương đối ra ngoài l5_execution: {mod}"
                continue
            top = mod.split(".")[0]
            assert top not in FORBIDDEN_NAMES, f"{f.name}:{line} cấm import {mod} trong L5"
            assert top in ALLOWED_TOP, f"{f.name}:{line} import ngoài whitelist: {mod}"
            assert top != "standard_quant", f"{f.name}:{line} L5 không được import phần còn lại của hệ thống"


def test_importing_l5_does_not_load_numeric_stack():
    code = (f"import sys; sys.path.insert(0, r'{ROOT}'); import standard_quant.l5_execution as m; "
            "bad = [k for k in ('numpy','pandas','sklearn','scipy') if k in sys.modules]; "
            "assert not bad, bad; print('clean')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "clean" in r.stdout, r.stderr


def test_no_indicator_vocabulary_in_l5():
    banned = ("ewm", "rolling", "ewmac", "atr", "sma", "rsi", "forecast", "vol_scalar", "kelly", "predict")
    for f in L5.glob("*.py"):
        text = f.read_text(encoding="utf-8").lower()
        tokens = {t for t in ast.walk(ast.parse(text)) if isinstance(t, ast.Name)}
        names = {t.id for t in tokens} | {n.attr for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Attribute)}
        assert not [b for b in banned if b in names], f"{f.name}: từ vựng chỉ báo {[b for b in banned if b in names]}"


# ---------------- Hành vi ----------------
@pytest.fixture
def env(tmp_path):
    con = connect(tmp_path / "sig.db")
    gw = FakeGateway(price=2000.0)
    return con, gw, Executor(con, gw, LIMITS)


def sig(i="a1", **kw):
    d = dict(signal_id=i, symbol="XAUUSD", side="BUY", volume=0.1, sl=1980.0, tp=0.0, ref_price=2000.0, magic=7)
    return Signal(**{**d, **kw})


def status(con, i):
    return con.execute("SELECT status, detail FROM signals WHERE signal_id=?", (i,)).fetchone()


def test_happy_path_sends_once_and_logs_latency(env):
    con, gw, ex = env
    assert publish(con, sig(), ttl_s=30)
    assert ex.run_once() == 1 and ex.run_once() == 0
    assert len(gw.sent) == 1 and gw.sent[0].sl == 1980.0
    assert status(con, "a1")["status"] == "SENT"
    row = con.execute("SELECT * FROM latency WHERE signal_id='a1'").fetchone()
    assert row["outcome"] == "SENT" and row["ms_total"] >= row["ms_roundtrip"] >= 0 and row["ticket"] > 0
    assert latency_summary(con)["n"] == 1


def test_publish_is_idempotent(env):
    con, gw, ex = env
    assert publish(con, sig()) and not publish(con, sig())
    ex.run_once()
    assert len(gw.sent) == 1


def test_stale_signal_never_sent(env):
    con, gw, ex = env
    publish(con, sig(created_ns=time.time_ns() - int(60e9)), ttl_s=1)
    ex.run_once()
    assert not gw.sent and status(con, "a1")["status"] == "EXPIRED"


@pytest.mark.parametrize("kw,reason", [
    (dict(sl=0.0), "missing_stop_loss"),
    (dict(sl=2010.0), "stop_loss_wrong_side"),
    (dict(volume=5.0), "exceeds_max_lot"),
    (dict(symbol="EURUSD"), "symbol_not_allowed"),
    (dict(magic=99), "magic_not_allowed"),
    (dict(side="HOLD"), "bad_side"),
    (dict(volume=0.0), "bad_volume"),
])
def test_vetoes(env, kw, reason):
    con, gw, ex = env
    publish(con, sig(**kw))
    ex.run_once()
    st = status(con, "a1")
    assert not gw.sent and st["status"] == "REJECTED" and st["detail"] == reason


def test_halt_blocks_opens_but_allows_reduce(env):
    con, gw, ex = env
    set_flag(con, "halt", True)
    publish(con, sig("open"))
    publish(con, sig("close", side="SELL", sl=0.0, reduce_only=True))
    ex.run_once()
    assert status(con, "open")["detail"] == "halted"
    assert status(con, "close")["status"] == "SENT" and len(gw.sent) == 1


def test_price_moved_guard(env):
    con, gw, ex = env
    gw.price = 2100.0                                   # lệch 500 bps so với ref 2000
    publish(con, sig())
    ex.run_once()
    assert status(con, "a1")["detail"] == "price_moved" and not gw.sent


def test_retry_only_on_not_filled_codes(env):
    con, gw, ex = env
    gw.retcodes = [10004, 10009]                        # requote → retry → OK
    publish(con, sig("r1"))
    ex.run_once()
    assert len(gw.sent) == 2 and status(con, "r1")["status"] == "SENT"
    gw.retcodes = [10006]                               # rejected cứng → không retry
    publish(con, sig("r2"))
    ex.run_once()
    assert len(gw.sent) == 3 and status(con, "r2")["status"] == "FAILED"


def test_timeout_is_unknown_and_never_resent(env):
    con, gw, ex = env
    gw.timeout = True
    publish(con, sig())
    ex.run_once()
    assert len(gw.sent) == 1 and status(con, "a1")["status"] == "UNKNOWN"
    ex.run_once()
    assert len(gw.sent) == 1


def test_orphaned_claim_becomes_unknown_after_restart(tmp_path):
    con = connect(tmp_path / "s.db")
    publish(con, sig())
    assert store.claim_next(con).signal_id == "a1"      # claim rồi "crash" trước khi gửi
    con2 = connect(tmp_path / "s.db")
    assert store.recover_orphans(con2) == 1
    gw = FakeGateway()
    Executor(con2, gw, LIMITS).run_once()
    assert not gw.sent and status(con2, "a1")["status"] == "UNKNOWN"


def test_slippage_recorded_adverse_positive(env):
    con, gw, ex = env
    gw.slip = 1.0                                       # BUY khớp cao hơn 1.0 → bất lợi
    publish(con, sig())
    ex.run_once()
    assert con.execute("SELECT slippage_bps FROM latency").fetchone()[0] == pytest.approx(5.0)


def test_two_workers_never_double_claim(tmp_path):
    a, b = connect(tmp_path / "s.db"), connect(tmp_path / "s.db")
    publish(a, sig())
    got = [store.claim_next(a), store.claim_next(b)]
    assert [g is not None for g in got].count(True) == 1


def test_upstream_publisher_bridge(tmp_path):
    from standard_quant.contracts import OrderIntent
    from standard_quant.s5_execution.publisher import publish_intents
    con = connect(tmp_path / "s.db")
    its = [OrderIntent("XAUUSD", "BUY", 0.1, 1980.0, None, "REBALANCE"),
           OrderIntent("XAUUSD", "SELL", 0.05, 0.0, None, "REDUCE/CLOSE")]
    assert publish_intents(con, its, "2025-01-02", {"XAUUSD": 2000.0}, magic=7) == 2
    assert publish_intents(con, its, "2025-01-02", {"XAUUSD": 2000.0}, magic=7) == 0     # chạy lại không nhân đôi
    gw = FakeGateway(2000.0)
    Executor(con, gw, LIMITS).run_once()
    assert [s.reduce_only for s in gw.sent] == [False, True]
