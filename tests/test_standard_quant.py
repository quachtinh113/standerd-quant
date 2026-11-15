import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from standard_quant import Config, StandardQuantPipeline, run_backtest
from standard_quant.contracts import Instrument, OrderIntent
from standard_quant.s1_data_rules import blended_daily_vol, clean_prices, synthetic_prices, validate_bars
from standard_quant.s2_forecast import FORECAST_CAP, combine_forecasts, ewmac
from standard_quant.s3_vol_targeting.sizing import buffer_position, round_to_lots, vol_scalar
from standard_quant.s4_portfolio import instrument_diversification_multiplier
from standard_quant.s5_execution import DrawdownLadder, PaperBroker, Position, plan_orders

CFG = Path(__file__).resolve().parents[1] / "config" / "standard.json"
PRICES = lambda n=900: synthetic_prices({"XAUUSD": 1900.0, "EURUSD": 1.10, "BTCUSD": 30000.0}, n_days=n)


# ---------- Khối 1 ----------
def test_clean_prices_drops_spike_and_never_backfills():
    idx = pd.bdate_range("2020-01-01", periods=10)
    s = pd.Series([100, 101, 102, 1000, 103, 104, 105, 106, 107, 108.0], index=idx)
    c = clean_prices(s)
    assert c.iloc[3] == 102                      # spike bị thay bằng ffill, không dùng giá tương lai
    assert not c.isna().any()
    s2 = s.copy(); s2.iloc[0] = np.nan
    assert np.isnan(clean_prices(s2).iloc[0])    # đầu chuỗi NaN giữ NaN (không bfill)


def test_validate_bars_flags_bad_ohlc():
    idx = pd.bdate_range("2020-01-01", periods=2)
    bad = pd.DataFrame({"open": [1, 1], "high": [0.5, 2], "low": [1, 1], "close": [1, 1]}, index=idx)
    assert validate_bars(bad)


def test_vol_is_causal():
    p = PRICES()["XAUUSD"]
    full = blended_daily_vol(p)
    cut = blended_daily_vol(p.iloc[:500])
    pd.testing.assert_series_equal(full.iloc[:500], cut)


# ---------- Khối 2 ----------
def test_forecast_causal_capped_and_scaled():
    p = PRICES(2000)["EURUSD"]
    v = blended_daily_vol(p)
    f_full = ewmac(p, v, 16)
    f_cut = ewmac(p.iloc[:700], v.iloc[:700], 16)
    pd.testing.assert_series_equal(f_full.iloc[:700], f_cut)         # zero look-ahead
    assert f_full.abs().max() <= FORECAST_CAP
    assert 5 < f_full.abs().mean() < 16                               # |trung bình| quanh 10


def test_combine_forecasts_capped():
    idx = pd.bdate_range("2020-01-01", periods=300)
    a = pd.Series(20.0, index=idx); b = pd.Series(20.0, index=idx)
    assert combine_forecasts({"a": a, "b": b}).abs().max() <= FORECAST_CAP


# ---------- Khối 3 ----------
def test_vol_scalar_hits_cash_target():
    inst = Instrument("X", point_value=100.0)
    lots = vol_scalar(10_000, 0.20, 2.0, inst)                        # vol giá 2 → 200$/lot/ngày
    assert lots * 2.0 * 100.0 == pytest.approx(10_000 * 0.20 / 16)    # √256 = 16


def test_buffer_moves_only_to_band_edge():
    assert buffer_position(10, 9.5) == 9.5                            # trong vùng → giữ
    assert buffer_position(10, 5) == pytest.approx(9.0)               # ngoài → về biên, không về tâm


def test_round_to_lots_respects_min_and_max():
    inst = Instrument("X", min_lot=0.01, lot_step=0.01, max_lot=1.0)
    assert round_to_lots(0.002, inst) == 0.0
    assert round_to_lots(-7.0, inst) == -1.0


# ---------- Khối 4 ----------
def test_idm_between_1_and_cap_and_defaults_to_1():
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.normal(size=(300, 3)), columns=list("abc"))
    idm = instrument_diversification_multiplier(r, {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})
    assert 1.0 < idm <= 2.5
    assert instrument_diversification_multiplier(None, {"a": 1.0}) == 1.0


# ---------- Khối 5 ----------
INST = {"XAUUSD": Instrument("XAUUSD", point_value=100.0, max_lot=5.0)}


def test_every_opening_order_has_broker_stop():
    out = plan_orders({"XAUUSD": 0.5}, {}, INST, {"XAUUSD": 2000.0}, {"XAUUSD": 20.0})
    assert out and all(o.sl > 0 for o in out)
    assert out[0].side == "BUY" and out[0].sl < 2000.0


def test_no_dca_on_underwater_position():
    cur = {"XAUUSD": Position(0.2, 2000.0)}
    out = plan_orders({"XAUUSD": 1.0}, cur, INST, {"XAUUSD": 1900.0}, {"XAUUSD": 20.0})
    assert out == []                                                  # lỗ → không tăng
    out = plan_orders({"XAUUSD": 1.0}, cur, INST, {"XAUUSD": 2100.0}, {"XAUUSD": 20.0})
    assert out and out[0].side == "BUY"                               # lãi → được tăng


def test_halted_only_reduces():
    cur = {"XAUUSD": Position(0.5, 2000.0)}
    assert plan_orders({"XAUUSD": 1.0}, cur, INST, {"XAUUSD": 2100.0}, {"XAUUSD": 20.0}, halted=True) == []
    out = plan_orders({"XAUUSD": -1.0}, cur, INST, {"XAUUSD": 2100.0}, {"XAUUSD": 20.0}, halted=True)
    assert len(out) == 1 and out[0].side == "SELL" and out[0].volume == pytest.approx(0.5)
    assert plan_orders({"XAUUSD": 1.0}, {}, INST, {"XAUUSD": 2100.0}, {"XAUUSD": 20.0}, halted=True) == []


def test_flip_splits_into_close_then_open():
    cur = {"XAUUSD": Position(0.5, 2000.0)}
    out = plan_orders({"XAUUSD": -0.5}, cur, INST, {"XAUUSD": 2000.0}, {"XAUUSD": 20.0})
    assert [o.reason for o in out] == ["REDUCE/CLOSE", "REBALANCE"]
    assert out[1].sl > 2000.0                                         # SELL → SL phía trên


def test_drawdown_ladder_tiers():
    L = DrawdownLadder()
    L.update(10_000)
    assert L.tier(10_000) == "NORMAL" and L.multiplier(10_000) == 1.0
    assert L.tier(9_650) == "TIER_1_WARNING"
    assert L.multiplier(9_450) == 0.5
    assert L.tier(9_200) == "TIER_3_HALTED"
    assert L.tier(9_900) == "TIER_3_HALTED"                           # latch: không tự mở lại
    L.reset_halt(9_900)
    assert L.tier(9_900) == "NORMAL"


def test_daily_loss_cap_halts_only_that_day():
    L = DrawdownLadder(daily_loss_cap=300.0)
    L.update(10_000)
    L.update(10_000, new_day=True)
    L.update(9_650)                                                   # cuối ngày: equity 9650
    assert L.tier(9_650) == "TIER_3_HALTED"                           # lỗ ngày 350 ≥ 300
    L.update(9_650, new_day=True)                                     # ngày mới
    assert L.tier(9_650) == "TIER_1_WARNING"                          # không còn halt


def test_paper_broker_pnl_and_cost():
    b = PaperBroker({"XAUUSD": Instrument("XAUUSD", point_value=100.0, cost_per_lot_rt=10.0)}, 10_000)
    b.send(OrderIntent("XAUUSD", "BUY", 1.0, 1900.0, None), 2000.0)
    b.send(OrderIntent("XAUUSD", "SELL", 1.0, 0.0, None), 2010.0)
    assert b.positions() == {}
    assert b.cash == pytest.approx(10_000 + 10 * 100 - 10.0)          # +1000 PnL − 10 phí


# ---------- Hệ thống ----------
def test_config_rejects_martingale_and_same_bar_fill():
    with pytest.raises(ValueError):
        Config(max_dca_layers=1)
    with pytest.raises(ValueError):
        Config(exec_lag=0)


def test_pipeline_no_lookahead_end_to_end():
    cfg = Config.load(CFG)
    prices = PRICES(700)
    full = StandardQuantPipeline(cfg, prices)
    cut = StandardQuantPipeline(cfg, {k: v.iloc[:500] for k, v in prices.items()})
    pd.testing.assert_frame_equal(full.forecasts.iloc[:500], cut.forecasts, check_exact=False, atol=1e-9)


def test_backtest_runs_and_halt_is_respected():
    r = run_backtest(Config.load(CFG), PRICES(900))
    assert np.isfinite(r["final_equity"]) and r["cost_paid"] >= 0
    assert set(r["tier_counts"]) <= {"NORMAL", "TIER_1_WARNING", "TIER_2_DERISK", "TIER_3_HALTED"}


# ---------- Tầng L2 Alpha ----------
from standard_quant.s2_forecast.alpha import (REGISTRY, alpha_to_forecast, evaluate_alpha,
                                              expected_max_sharpe, operators as ops, robustness_report)


def _panel(n=900):
    close = pd.DataFrame(PRICES(n))
    vol = pd.DataFrame({c: blended_daily_vol(close[c]) for c in close})
    return {"close": close, "returns": close.pct_change(), "vol": vol}


def test_operators_are_causal():
    d = _panel(600)
    cut = {k: v.iloc[:400] for k, v in d.items()}
    for name in ("ts_zscore", "ts_rank", "ts_decay_linear", "ts_mean"):
        fn = getattr(ops, name)
        pd.testing.assert_frame_equal(fn(d["close"], 20).iloc[:400], fn(cut["close"], 20))
    pd.testing.assert_frame_equal(ops.ts_corr(d["close"], d["vol"], 20).iloc[:400],
                                  ops.ts_corr(cut["close"], cut["vol"], 20))
    with pytest.raises(ValueError):
        ops.ts_delay(d["close"], -1)


def test_cross_sectional_operators():
    x = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 2.0], "c": [5.0, 2.0]})
    assert ops.cs_demean(x).sum(axis=1).abs().max() < 1e-12
    assert ops.cs_scale(x).abs().sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    r = ops.cs_rank(x)
    assert r.iloc[0]["a"] < r.iloc[0]["b"] < r.iloc[0]["c"]


def test_ts_rank_and_decay_values():
    x = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    assert ops.ts_rank(x, 4).iloc[-1, 0] == pytest.approx(1.0)
    assert ops.ts_decay_linear(x, 2).iloc[-1, 0] == pytest.approx((3 * 1 + 4 * 2) / 3)


def test_all_library_alphas_causal_and_finite():
    d = _panel(700)
    cut = {k: v.iloc[:450] for k, v in d.items()}
    for name, fn in REGISTRY.items():
        full, part = fn(d), fn(cut)
        pd.testing.assert_frame_equal(full.iloc[:450], part, check_exact=False, atol=1e-9, obj=name)
        assert np.isfinite(full.dropna().to_numpy()).all()


def test_alpha_to_forecast_scaled_and_capped():
    d = _panel(900)
    f = alpha_to_forecast(REGISTRY["mom_vol_adj"](d))
    assert f.abs().max().max() <= FORECAST_CAP
    assert 6 < f.abs().stack().mean() < 14


def test_expected_max_sharpe_grows_with_trials():
    assert expected_max_sharpe(1, 5) == 0.0
    assert expected_max_sharpe(100, 5) > expected_max_sharpe(10, 5) > 0
    assert expected_max_sharpe(100, 10) < expected_max_sharpe(100, 5)


def test_noise_alpha_is_rejected_by_overfit_guards():
    d = _panel(1200)
    rng = np.random.default_rng(1)
    noise = pd.DataFrame(rng.normal(size=d["close"].shape), index=d["close"].index, columns=d["close"].columns)
    rep = evaluate_alpha(noise, d["returns"], n_trials=50)
    assert not rep.passes and rep.hurdle > 0
    rob = robustness_report(noise, d["returns"])
    assert {"base", "delay2", "cost_x2"} <= set(rob)


def test_pipeline_accepts_alphas_and_stays_causal():
    cfg = Config.load(CFG)
    cfg.alphas = ["mom_vol_adj", "range_position"]
    prices = PRICES(700)
    full = StandardQuantPipeline(cfg, prices)
    cut = StandardQuantPipeline(cfg, {k: v.iloc[:500] for k, v in prices.items()})
    pd.testing.assert_frame_equal(full.forecasts.iloc[:500], cut.forecasts, check_exact=False, atol=1e-9)
    assert np.isfinite(run_backtest(cfg, prices)["final_equity"])


# ---------- Tầng L3 Meta-labeling ----------
from standard_quant.s2_forecast.meta import (PurgedKFold, apply_meta, average_uniqueness, bet_size, build_events,
                                             deflated_sharpe, pbo_cscv, probabilistic_sharpe, triple_barrier,
                                             walk_forward_probability)


def test_triple_barrier_labels_by_hand():
    idx = pd.bdate_range("2020-01-01", periods=12)
    close = pd.Series([100, 101, 105, 106, 100, 99, 98, 97, 96, 95, 94, 93.0], index=idx)
    vol = pd.Series(1.0, index=idx)
    side = pd.Series([1, -1] + [0] * 10, index=idx)
    out = triple_barrier(close, side, vol, pt_mult=1.0, sl_mult=1.0, horizon=4)   # rào ±1×1×2 = ±2
    assert out.loc[idx[0], "barrier"] == "pt" and out.loc[idx[0], "label"] == 1   # long, +5 ở phiên 2
    assert out.loc[idx[0], "t1"] == idx[2]
    assert out.loc[idx[1], "barrier"] == "sl" and out.loc[idx[1], "label"] == 0   # short, giá lên 105 → thua


def test_events_unresolved_tail_dropped():
    idx = pd.bdate_range("2020-01-01", periods=30)
    close = pd.Series(np.linspace(100, 110, 30), index=idx)
    out = triple_barrier(close, pd.Series(1.0, index=idx), pd.Series(0.5, index=idx), horizon=5)
    assert out.index.max() == idx[-6]                       # 5 phiên cuối chưa đủ rào dọc → bỏ


def test_uniqueness_overlap_lowers_weight():
    idx = pd.bdate_range("2020-01-01", periods=10)
    t1 = pd.Series([idx[3], idx[3], idx[9]], index=[idx[0], idx[1], idx[6]])
    u = average_uniqueness(t1, idx)
    assert u.iloc[2] == 1.0 and u.iloc[0] < 1.0


def test_purged_kfold_has_no_overlap_or_embargo_leak():
    idx = pd.bdate_range("2020-01-01", periods=300)
    t0 = idx[:280]
    t1 = pd.Series([idx[i + 10] for i in range(280)], index=t0)
    for tr, te in PurgedKFold(5, embargo_bars=5).split(t0, t1, idx):
        te_s, te_e = t0[te].min(), t1.iloc[te].max()
        assert not (set(tr) & set(te))
        for i in tr:
            assert t1.iloc[i] < te_s or t0[i] > te_e                 # không chồng lấn với test
            if t0[i] > te_e:
                assert idx.get_loc(t0[i]) - idx.get_loc(te_e) > 5     # ngoài embargo


def test_bet_size_properties():
    p = pd.Series([0.3, 0.5, 0.6, 0.9, np.nan])
    g = bet_size(p, 0.5, "gate")
    assert g.iloc[:4].tolist() == [0.0, 1.0, 1.0, 1.0] and np.isnan(g.iloc[4])
    c = bet_size(p, 0.5, "continuous")
    assert c.iloc[0] == 0.0 and 0 <= c.iloc[1] < c.iloc[2] < c.iloc[3] <= 1.0
    with pytest.raises(ValueError):
        bet_size(p, 0.5, "bogus")


def test_apply_meta_never_flips_sign_and_burn_in():
    idx = pd.bdate_range("2020-01-01", periods=4)
    f = pd.DataFrame({"a": [10.0, -10.0, 5.0, -5.0]}, index=idx)
    p = pd.DataFrame({"a": [np.nan, 0.9, 0.2, 0.7]}, index=idx)
    out = apply_meta(f, p, 0.5, "gate", burn_in="pass")
    assert out["a"].tolist() == [10.0, -10.0, 0.0, -5.0]
    assert apply_meta(f, p, 0.5, "gate", burn_in="flat")["a"].iloc[0] == 0.0
    assert (np.sign(apply_meta(f, p.fillna(0.99), 0.5, "continuous")["a"]) * np.sign(f["a"]) >= 0).all()


def test_meta_walk_forward_is_causal():
    prices = PRICES(1000)
    cfg = Config.load(CFG)
    pipe = StandardQuantPipeline(cfg, prices)
    kw = dict(min_events=150, refit_every=63)
    pf, _ = walk_forward_probability(pipe.prices, pipe.vols, pipe.forecasts, **kw)
    cut = 760
    pc, _ = walk_forward_probability(pipe.prices.iloc[:cut], pipe.vols.iloc[:cut], pipe.forecasts.iloc[:cut], **kw)
    assert pc.notna().to_numpy().sum() > 100                              # có dự đoán thật để so sánh
    pd.testing.assert_frame_equal(pf.iloc[:cut - 63], pc.iloc[:cut - 63])  # bỏ khối cuối: biên cửa sổ khác nhau


def test_meta_trailing_feature_uses_only_resolved_labels():
    from standard_quant.s2_forecast.meta.features import _trailing_hit
    idx = pd.bdate_range("2020-01-01", periods=6)
    lab = pd.DataFrame({"t1": [idx[3]], "label": [1]}, index=[idx[0]])
    th = _trailing_hit(idx, lab, 20)
    assert th.iloc[3] == 0.5 and th.iloc[4] == 1.0                        # chỉ thấy nhãn sau khi t1 < t


def test_dsr_psr_pbo():
    rng = np.random.default_rng(3)
    good = pd.Series(rng.normal(0.002, 0.01, 1500))
    assert probabilistic_sharpe(good, 0.0) > 0.95
    assert deflated_sharpe(good, 1000) < probabilistic_sharpe(good, 0.0)  # thử nhiều → khắt khe hơn
    trials = pd.DataFrame(rng.normal(0, 0.01, (800, 20)))
    r = pbo_cscv(trials, 8)
    assert r["n_combinations"] == 70 and 0.2 < r["pbo"] < 0.8             # nhiễu thuần: PBO ~ 0.5


def test_pipeline_with_meta_runs():
    cfg = Config.load(CFG)
    cfg.meta_enabled, cfg.meta_min_events = True, 150
    pipe = StandardQuantPipeline(cfg, PRICES(900))
    assert pipe.meta_prob.notna().any().any()
    assert (pipe.forecasts.abs().fillna(0) <= pipe.forecasts_raw.abs().fillna(0) + 1e-12).all().all()   # meta chỉ thu nhỏ
    assert np.isfinite(run_backtest(cfg, PRICES(900))["final_equity"])


def test_purged_cv_auc_runs_on_events():
    from standard_quant.s2_forecast.meta import purged_cv_auc
    pipe = StandardQuantPipeline(Config.load(CFG), PRICES(900))
    ev = build_events(pipe.prices, pipe.vols, pipe.forecasts)
    auc = purged_cv_auc(ev, pipe.prices.index)
    assert 0.3 < auc < 0.7                                   # dữ liệu tổng hợp: không có thông tin thật
