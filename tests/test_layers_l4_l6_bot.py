"""L4 (rủi ro), L6 (giám sát) và bot end-to-end."""
import ast
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from standard_quant import Config
from standard_quant.account import AccountState, Pos
from standard_quant.bot import Bot
from standard_quant.contracts import Instrument, OrderIntent
from standard_quant.io_feeds import CallableFeed, StaticAccount
from standard_quant.l4_risk import RiskDesk, RiskLimits, load_ladder, save_ladder
from standard_quant.l5_execution import ExecLimits, Executor, FakeGateway, Signal, connect, publish, store
from standard_quant.l6_monitoring import (Alert, Dispatcher, HealthLimits, build_report, check_health,
                                          position_drift, resolve_unknown, unknown_for_review)
from standard_quant.ledger import HashChainLedger
from standard_quant.s1_data_rules import synthetic_prices
from standard_quant.s5_execution.risk import DrawdownLadder

CFG = ROOT / "config" / "standard.json"
INST = {"XAUUSD": Instrument("XAUUSD", point_value=100.0, max_lot=5.0),
        "EURUSD": Instrument("EURUSD", point_value=100000.0, max_lot=10.0)}
WED = datetime(2025, 1, 8, 12, 0)                      # thứ Tư 12:00 UTC


def it(sym="XAUUSD", side="BUY", vol=0.1, sl=1900.0, reason="REBALANCE"):
    return OrderIntent(sym, side, vol, sl, None, reason)


# ---------------- Ledger ----------------
def test_ledger_chain_and_tamper_detection(tmp_path):
    L = HashChainLedger(tmp_path / "l.jsonl")
    for k in range(3):
        L.append("E", {"k": k})
    assert L.verify().ok and L.verify().count == 3
    lines = (tmp_path / "l.jsonl").read_text().splitlines()
    rec = json.loads(lines[1]); rec["k"] = 99
    lines[1] = json.dumps(rec)
    (tmp_path / "l.jsonl").write_text("\n".join(lines) + "\n")
    v = HashChainLedger(tmp_path / "l.jsonl").verify()
    assert not v.ok and v.first_break == 2


# ---------------- L4 ----------------
@pytest.fixture
def desk(tmp_path):
    con = connect(tmp_path / "s.db")
    return RiskDesk(RiskLimits(), DrawdownLadder(), HashChainLedger(tmp_path / "l.jsonl"), con), con


def acct(eq=10_000.0, held=None, margin=0.0, ml=0.0):
    return AccountState(eq, margin, ml, held or {})


def test_desk_approves_normal_and_blocks_weekend_rollover(desk):
    d, _ = desk
    px = {"XAUUSD": 2000.0}
    assert d.review([it()], acct(), WED, px, INST).approved
    sat = datetime(2025, 1, 11, 12, 0)
    assert d.review([it()], acct(), sat, px, INST).rejected[0][1] == "weekend_flat"
    roll = datetime(2025, 1, 8, 23, 50)
    assert d.review([it()], acct(), roll, px, INST).rejected[0][1] == "rollover"


def test_desk_never_blocks_exits(desk):
    d, _ = desk
    out = d.review([it(side="SELL", reason="REDUCE/CLOSE", sl=0.0)], acct(held={"XAUUSD": Pos(0.1, 2000)}),
                   datetime(2025, 1, 11, 12, 0), {"XAUUSD": 2000.0}, INST)
    assert out.approved and not out.rejected


def test_desk_limits(desk):
    d, _ = desk
    px = {"XAUUSD": 2000.0, "EURUSD": 1.1}
    assert d.review([it(vol=2.0)], acct(), WED, px, INST).rejected[0][1] == "leverage_cap"      # 400k/10k = 40×
    assert d.review([it(sl=0.0)], acct(), WED, px, INST).rejected[0][1] == "missing_stop_loss"
    assert d.review([it(vol=9.0)], acct(), WED, px, INST).rejected[0][1] == "exceeds_symbol_lot_cap"
    assert d.review([it("GBPUSD")], acct(), WED, px, INST).rejected[0][1] == "unknown_instrument"
    assert d.review([it()], acct(margin=500, ml=120), WED, px, INST).rejected[0][1] == "margin_level_low"
    d.limits.max_open_positions = 1
    assert d.review([it("EURUSD", vol=0.1, sl=1.0)], acct(held={"XAUUSD": Pos(0.1, 2000)}), WED, px, INST
                    ).rejected[0][1] == "max_open_positions"


def test_desk_leverage_counts_cumulative_intents(desk):
    d, _ = desk
    out = d.review([it(vol=0.5), it(vol=0.5)], acct(), WED, {"XAUUSD": 2000.0}, INST)   # mỗi cái 10×; cộng dồn 20× > 15×
    assert len(out.approved) == 1 and out.rejected[0][1] == "leverage_cap"


def test_halt_sets_l5_flag_ledger_and_blocks_opens(desk):
    d, con = desk
    d.update_equity(10_000)
    out = d.review([it(), it(side="SELL", reason="REDUCE/CLOSE", sl=0)], acct(9_000), WED, {"XAUUSD": 2000.0}, INST)
    assert out.halted and store.get_flag(con, "halt")
    assert [i.reason for i in out.approved] == ["REDUCE/CLOSE"] and out.rejected[0][1] == "halted"
    assert any(json.loads(l)["event_type"] == "TIER_CHANGE" for l in (d.ledger.path).read_text().splitlines())


def test_manual_halt_file(tmp_path):
    con = connect(tmp_path / "s.db")
    hf = tmp_path / "HALT"
    d = RiskDesk(RiskLimits(halt_file=hf), DrawdownLadder(), None, con)
    assert d.review([it()], acct(), WED, {"XAUUSD": 2000.0}, INST).approved
    hf.write_text("x")
    assert d.review([it()], acct(), WED, {"XAUUSD": 2000.0}, INST).rejected[0][1] == "halted"
    assert store.get_flag(con, "halt")


def test_ladder_state_survives_restart(tmp_path):
    con = connect(tmp_path / "s.db")
    L = DrawdownLadder(); L.update(10_000); L.update(9_000); L.tier(9_000)
    save_ladder(con, L)
    L2 = load_ladder(connect(tmp_path / "s.db"), DrawdownLadder())
    assert L2.hwm == 10_000 and L2.halted_latch and L2.tier(9_000) == "TIER_3_HALTED"


# ---------------- L6 ----------------
def _sig(i, **kw):
    return Signal(**{**dict(signal_id=i, symbol="XAUUSD", side="BUY", volume=0.1, sl=1900.0, ref_price=2000.0, magic=7), **kw})


def test_health_detects_stale_worker_unknown_and_stuck(tmp_path):
    con = connect(tmp_path / "s.db")
    codes = lambda **kw: {a.code for a in check_health(con, **kw)}
    assert "NO_HEARTBEAT" in codes()
    store.set_value(con, "heartbeat_ns", str(time.time_ns() - int(120e9)))
    assert "WORKER_STALE" in codes()
    store.set_value(con, "heartbeat_ns", str(time.time_ns()))
    assert codes() == set()
    publish(con, _sig("old", created_ns=time.time_ns() - int(300e9)), ttl_s=3600)
    assert "STUCK_SIGNALS" in codes()
    store.claim_next(con); store.mark(con, "old", "UNKNOWN")
    assert "UNKNOWN_ORDERS" in codes()
    assert build_report(con)["status"] == "CRITICAL"


def test_health_latency_slippage_reject_rate(tmp_path):
    con = connect(tmp_path / "s.db")
    store.set_value(con, "heartbeat_ns", str(time.time_ns()))
    gw = FakeGateway(2000.0, delay_s=0.0, slip=2.0)               # BUY +2 → 10 bps bất lợi
    ex = Executor(con, gw, ExecLimits(frozenset({"XAUUSD"}), 1.0, 0.0, True, 1))
    for k in range(12):
        publish(con, _sig(f"s{k}"))
    ex.run_once()
    assert "HIGH_SLIPPAGE" in {a.code for a in check_health(con)}
    for k in range(12):
        publish(con, _sig(f"r{k}", sl=0.0))
    ex.run_once()
    assert "HIGH_REJECT_RATE" in {a.code for a in check_health(con)}


def test_resolve_unknown_requires_human_and_never_resends(tmp_path):
    con = connect(tmp_path / "s.db")
    led = HashChainLedger(tmp_path / "l.jsonl")
    publish(con, _sig("u1")); store.claim_next(con); store.mark(con, "u1", "UNKNOWN")
    assert unknown_for_review(con, {"XAUUSD": 0.1})[0]["broker_net_lots"] == 0.1
    resolve_unknown(con, "u1", "SENT", "thấy vị thế trên MT5", led)
    assert store.status_counts(con) == {"SENT": 1} and led.verify().count == 1
    with pytest.raises(ValueError):
        resolve_unknown(con, "u1", "SENT", "lần 2")


def test_position_drift_and_dispatcher_cooldown(tmp_path):
    assert position_drift({"A": 0.10}, {"A": 0.10}) == []
    assert position_drift({"A": 0.10}, {"A": 0.0})[0].code == "POSITION_DRIFT"
    got = []
    class Sink:
        def emit(self, a): got.append(a.code)
    d = Dispatcher([Sink()], cooldown_s=60)
    a = [Alert("CRITICAL", "X", "m")]
    d.dispatch(a, now=1000.0); d.dispatch(a, now=1010.0); d.dispatch(a, now=1100.0)
    assert got == ["X", "X"]


def test_health_flags_tampered_ledger(tmp_path):
    con = connect(tmp_path / "s.db")
    led = HashChainLedger(tmp_path / "l.jsonl"); led.append("A"); led.append("B")
    p = tmp_path / "l.jsonl"
    p.write_text(p.read_text().replace('"A"', '"Z"'))
    assert "LEDGER_TAMPERED" in {a.code for a in check_health(con, ledger_path=p, expect_worker=False)}


def test_l6_cli_exit_code(tmp_path):
    con = connect(tmp_path / "s.db"); con.close()
    r = subprocess.run([sys.executable, "-m", "standard_quant.l6_monitoring", "--db", str(tmp_path / "s.db"), "--no-worker"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    r = subprocess.run([sys.executable, "-m", "standard_quant.l6_monitoring", "--db", str(tmp_path / "s.db")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 2                                           # thiếu heartbeat = CRITICAL


def test_l6_imports_stdlib_only():
    code = (f"import sys; sys.path.insert(0, r'{ROOT}'); import standard_quant.l6_monitoring; "
            "bad=[k for k in ('numpy','pandas','sklearn','scipy') if k in sys.modules]; assert not bad, bad; print('clean')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert "clean" in r.stdout, r.stderr


# ---------------- Bot end-to-end ----------------
def _prices(n=700):
    return synthetic_prices({"XAUUSD": 1900.0, "EURUSD": 1.10, "BTCUSD": 30000.0}, n_days=n, seed=11)


def _bot(tmp_path, equity=10_000.0, prices=None, now=None):
    px = prices or _prices()
    last = max(p.index[-1] for p in px.values())
    now_ts = (last + pd.Timedelta(hours=10)).to_pydatetime() if now is None else now
    acct = StaticAccount(equity)
    bot = Bot(Config.load(CFG), tmp_path / "s.db", CallableFeed(lambda: px), acct, magic=7,
              ledger_path=tmp_path / "l.jsonl", now_fn=lambda: now_ts, limits=RiskLimits(max_leverage=1e9))
    return bot, acct


def test_bot_cycle_publishes_executes_and_is_idempotent(tmp_path):
    bot, _ = _bot(tmp_path)
    r = bot.run_cycle()
    assert r.status == "RAN" and r.published > 0, r
    assert bot.run_cycle().status == "SKIPPED_SAME_BAR"
    gw = FakeGateway(1.0)
    ex = Executor(bot.con, gw, ExecLimits(frozenset({"XAUUSD", "EURUSD", "BTCUSD"}), 100.0, 0.0, True, 1, frozenset({7})))
    assert ex.run_once() == r.published
    assert all(s.sl > 0 for s in gw.sent if not s.reduce_only) and store.status_counts(bot.con) == {"SENT": r.published}
    assert bot.ledger.verify().ok


def test_bot_does_not_double_publish_while_inflight(tmp_path):
    bot, _ = _bot(tmp_path)
    first = bot.run_cycle()
    assert first.published > 0
    store.set_value(bot.con, "last_asof", "")                         # giả lập chu kỳ mới, L5 chưa xử lý gì
    second = bot.run_cycle()
    assert second.published == 0 and all(why.startswith("in_flight") for _, why in second.rejected)


def test_bot_halts_on_drawdown_and_only_exits_pass(tmp_path):
    bot, acct = _bot(tmp_path)
    bot.run_cycle()
    bot.con.execute("UPDATE signals SET status='SENT'")
    store.set_value(bot.con, "last_asof", "")
    bot.ladder.hwm = 10_000.0
    acct.s = AccountState(9_000.0, held={"XAUUSD": Pos(0.3, 1900.0)})  # DD 10% → halt; mọi tín hiệu cũ coi như đã xử lý
    r = bot.run_cycle()
    assert r.halted and r.tier == "TIER_3_HALTED" and store.get_flag(bot.con, "halt")
    rows = bot.con.execute("SELECT reduce_only FROM signals WHERE status='PENDING'").fetchall()
    assert all(x["reduce_only"] == 1 for x in rows)                    # chỉ lệnh giảm/đóng lọt qua


def test_bot_skips_stale_data_and_ledgers_it(tmp_path):
    px = _prices()
    bot, _ = _bot(tmp_path, prices=px, now=datetime(2030, 1, 1, 10, 0))
    r = bot.run_cycle()
    assert r.status == "SKIPPED_STALE_DATA" and r.published == 0
    assert any("stale_data" in l for l in (tmp_path / "l.jsonl").read_text().splitlines())


def test_bot_publishes_broker_symbols_and_inflight_uses_them(tmp_path):
    px = _prices()
    last = max(p.index[-1] for p in px.values())
    smap = {"XAUUSD": "XAUUSDm", "EURUSD": "EURUSDm", "BTCUSD": "BTCUSDm"}
    now_ts = (last + pd.Timedelta(hours=10)).to_pydatetime()
    bot = Bot(Config.load(CFG), tmp_path / "s.db", CallableFeed(lambda: px), StaticAccount(10_000.0), magic=7,
              ledger_path=tmp_path / "l.jsonl", now_fn=lambda: now_ts, limits=RiskLimits(max_leverage=1e9), symbol_map=smap)
    r = bot.run_cycle()
    syms = {x["symbol"] for x in bot.con.execute("SELECT symbol FROM signals")}
    assert r.published > 0 and syms <= set(smap.values())
    store.set_value(bot.con, "last_asof", "")
    assert bot.run_cycle().published == 0                              # in-flight nhận ra tên broker
