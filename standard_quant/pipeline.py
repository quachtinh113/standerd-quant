"""Pipeline: nối 5 khối theo thứ tự Carver. Backtest và live dùng CÙNG `step()`."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .contracts import PipelineSnapshot
from .s1_data_rules import blended_daily_vol, clean_prices
from .s2_forecast.alpha import build_alpha_forecasts
from .s2_forecast.meta import apply_meta, walk_forward_probability
from .s2_forecast import breakout, combine_forecasts, ewmac
from .s3_vol_targeting.sizing import TRADING_DAYS
from .s4_portfolio import instrument_diversification_multiplier, instrument_weights, target_positions
from .s5_execution import DrawdownLadder, PaperBroker, plan_orders


class StandardQuantPipeline:
    def __init__(self, cfg: Config, prices: dict[str, pd.Series]):
        self.cfg = cfg
        self.symbols = [s for s in cfg.instruments if s in prices]
        if not self.symbols:
            raise ValueError("không có instrument nào có dữ liệu giá")
        # Khối 1
        px = pd.DataFrame({s: clean_prices(prices[s]) for s in self.symbols}).sort_index().ffill(limit=3)
        self.prices = px.dropna(how="all")
        self.vols = pd.DataFrame({s: blended_daily_vol(self.prices[s].dropna()) for s in self.symbols})
        self.returns = self.prices.pct_change(fill_method=None)
        # Khối 2
        self._alpha_fc = build_alpha_forecasts(cfg.alphas, self.prices, self.vols) if cfg.alphas else {}
        self.forecasts = pd.DataFrame({s: self._forecast(s) for s in self.symbols})
        self.forecasts_raw = self.forecasts                       # trước khi lọc ML (để so sánh/audit)
        self.meta_prob = self.meta_events = None
        if cfg.meta_enabled:                                       # L3: chỉ nhân hệ số [0,1], không đảo chiều
            self.meta_prob, self.meta_events = walk_forward_probability(
                self.prices, self.vols, self.forecasts_raw, min_events=cfg.meta_min_events)
            self.forecasts = apply_meta(self.forecasts_raw, self.meta_prob, cfg.meta_threshold,
                                        cfg.meta_mode, cfg.meta_burn_in)
        self._idm_cache: dict[int, float] = {}

    def _forecast(self, sym: str) -> pd.Series:
        p, v = self.prices[sym].dropna(), self.vols[sym].dropna()
        rules = {f"ewmac{f}": ewmac(p, v.reindex(p.index).ffill(), f) for f in self.cfg.ewmac_speeds}
        rules |= {f"brk{n}": breakout(p, n) for n in self.cfg.breakout_lookbacks}
        rules |= {f"alpha_{n}": f[sym] for n, f in self._alpha_fc.items()}
        return combine_forecasts(rules)

    def _idm(self, i: int) -> float:
        """IDM hiệu chỉnh lại mỗi 21 phiên từ lợi suất ≤ t-1."""
        k = i // 21
        if k not in self._idm_cache:
            hist = self.returns.iloc[: max(k * 21, 0)]            # ≤ t-1
            w = instrument_weights(hist, self.symbols, self.cfg.weights_method)
            self._idm_cache[k] = instrument_diversification_multiplier(hist, w, self.cfg.idm_cap)
        return self._idm_cache[k]

    def step(self, i: int, equity: float, broker: PaperBroker, ladder: DrawdownLadder) -> PipelineSnapshot:
        """Một chu kỳ quyết định tại hàng i (đóng cửa phiên i). Trả snapshot; intents chưa khớp."""
        t = self.prices.index[i]
        hist = self.returns.iloc[:i]
        w = instrument_weights(hist, self.symbols, self.cfg.weights_method)
        insts = {s: self.cfg.instruments[s] for s in self.symbols}
        fc = {s: float(self.forecasts[s].iloc[i]) for s in self.symbols}
        vol = {s: float(self.vols[s].iloc[i]) for s in self.symbols}
        px = {s: float(self.prices[s].iloc[i]) for s in self.symbols}

        mult = ladder.multiplier(equity)                        # Risk overlay
        halted = ladder.is_halted(equity)
        tgt = target_positions(fc, vol, insts, equity, self.cfg.vol_target_annual, w, self._idm(i), mult)
        cur = broker.positions()
        intents = plan_orders(tgt, cur, insts, px, vol, halted=halted,
                              buffer_fraction=self.cfg.buffer_fraction, sl_vol_mult=self.cfg.sl_vol_mult)
        return PipelineSnapshot(asof=t, forecasts=fc, target_positions=tgt,
                                current_positions={s: p.volume for s, p in cur.items()},
                                intents=intents, risk_multiplier=mult, halted=halted)


def run_backtest(cfg: Config, prices: dict[str, pd.Series]) -> dict:
    pipe = StandardQuantPipeline(cfg, prices)
    broker = PaperBroker({s: cfg.instruments[s] for s in pipe.symbols}, cfg.capital)
    ladder = DrawdownLadder(cfg.dd_tier1, cfg.dd_tier2, cfg.dd_tier3, cfg.daily_loss_cap)
    warmup = 80
    pending: list = []
    equity_curve, tiers, turnover, halt_days = [], [], 0.0, 0
    idx = pipe.prices.index

    for i in range(len(idx)):
        row = {s: float(pipe.prices[s].iloc[i]) for s in pipe.symbols}
        # khớp lệnh của phiên trước ở giá phiên này (exec_lag = 1)
        for it in pending:
            p = row.get(it.symbol)
            if p is not None and np.isfinite(p):
                broker.send(it, p)
                turnover += it.volume
        pending = []
        eq = broker.equity(row)
        ladder.update(eq, new_day=True)
        tier = ladder.tier(eq)
        if ladder.halted_latch:
            halt_days += 1
            if halt_days >= cfg.halt_cooldown_days:          # proxy cho việc con người duyệt mở lại
                ladder.reset_halt(eq)
                halt_days = 0
        equity_curve.append(eq)
        tiers.append(tier)
        if i < warmup or i == len(idx) - 1:
            continue
        snap = pipe.step(i, eq, broker, ladder)
        pending = snap.intents

    eqs = pd.Series(equity_curve, index=idx)
    rets = eqs.pct_change().dropna()
    dd = (eqs / eqs.cummax() - 1).min()
    ann_ret = (eqs.iloc[-1] / eqs.iloc[0]) ** (TRADING_DAYS / max(len(eqs), 1)) - 1
    ann_vol = rets.std() * np.sqrt(TRADING_DAYS)
    return {
        "equity": eqs,
        "final_equity": float(eqs.iloc[-1]),
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(rets.mean() / rets.std() * np.sqrt(TRADING_DAYS)) if rets.std() > 0 else 0.0,
        "max_drawdown": float(dd),
        "turnover_lots": float(turnover),
        "cost_paid": float(broker.cost_paid),
        "tier_counts": pd.Series(tiers).value_counts().to_dict(),
    }
