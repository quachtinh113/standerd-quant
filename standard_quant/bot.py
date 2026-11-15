"""Bot hoàn chỉnh: một chu kỳ quyết định nối L1→L2→L3→vol→portfolio→L4 (rủi ro)→hộp thư→(L5 worker)→L6.

run_cycle() chạy MỘT lần sau khi nến D1 đóng. Idempotent: chạy lại cùng phiên không phát thêm tín hiệu.
L5 là tiến trình riêng (python -m standard_quant.l5_execution); bot KHÔNG gọi broker để đặt lệnh.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from .config import Config
from .io_feeds import AccountFeed, DataFeed
from .l4_risk import RiskDesk, RiskLimits, load_ladder, save_ladder
from .l5_execution import store
from .ledger import HashChainLedger
from .pipeline import StandardQuantPipeline
from .s5_execution.publisher import publish_intents
from .s5_execution.risk import DrawdownLadder


@dataclass
class CycleReport:
    status: str                       # RAN | SKIPPED_SAME_BAR | SKIPPED_STALE_DATA | NO_DATA
    asof: str = ""
    published: int = 0
    rejected: list[tuple[str, str]] = field(default_factory=list)      # (symbol side, lý do)
    tier: str = "NORMAL"
    halted: bool = False
    forecasts: dict[str, float] = field(default_factory=dict)
    targets: dict[str, float] = field(default_factory=dict)


class Bot:
    def __init__(self, cfg: Config, db_path: str | Path, feed: DataFeed, account: AccountFeed, *,
                 magic: int, ledger_path: str | Path, limits: RiskLimits | None = None,
                 symbol_map: dict[str, str] | None = None, max_stale_days: int = 5, signal_ttl_s: float = 3600.0,
                 now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self.cfg, self.feed, self.account, self.magic = cfg, feed, account, magic
        self.con = store.connect(db_path)
        self.ledger = HashChainLedger(ledger_path)
        self.ladder = load_ladder(self.con, DrawdownLadder(cfg.dd_tier1, cfg.dd_tier2, cfg.dd_tier3, cfg.daily_loss_cap))
        self.desk = RiskDesk(limits or RiskLimits(), self.ladder, self.ledger, self.con)
        self.symbol_map = symbol_map or {}
        self.max_stale_days, self.ttl, self.now_fn = max_stale_days, signal_ttl_s, now_fn

    def run_cycle(self) -> CycleReport:
        now = self.now_fn()
        prices = self.feed.load()
        prices = {s: p.dropna() for s, p in prices.items() if len(p.dropna())}
        if not prices:
            self.ledger.append("CYCLE_SKIPPED", {"reason": "no_data"})
            return CycleReport("NO_DATA")
        last_bar = max(p.index[-1] for p in prices.values())
        asof = pd.Timestamp(last_bar).strftime("%Y-%m-%d")
        if store.get_value(self.con, "last_asof") == asof:
            return CycleReport("SKIPPED_SAME_BAR", asof)
        age_days = (pd.Timestamp(now.replace(tzinfo=None)).normalize() - pd.Timestamp(last_bar).normalize()).days
        if age_days > self.max_stale_days:
            self.ledger.append("CYCLE_SKIPPED", {"reason": "stale_data", "asof": asof, "age_days": age_days})
            return CycleReport("SKIPPED_STALE_DATA", asof)

        acct = self.account.state()
        self.desk.update_equity(acct.equity, new_day=True)             # ladder + cờ halt xuống L5
        pipe = StandardQuantPipeline(self.cfg, prices)
        i = len(pipe.prices) - 1
        snap = pipe.step(i, acct.equity, acct, self.ladder)

        # Không phát thêm cho symbol còn tín hiệu chưa chốt / UNKNOWN: tránh nhân đôi vị thế
        inflight = store.inflight_symbols(self.con)
        rep = CycleReport("RAN", asof, tier=self.ladder.tier(acct.equity), halted=snap.halted,
                          forecasts=snap.forecasts, targets=snap.target_positions)
        ok_intents = []
        for it in snap.intents:
            bsym = self.symbol_map.get(it.symbol, it.symbol)
            if bsym in inflight:
                rep.rejected.append((f"{it.symbol} {it.side}", f"in_flight:{inflight[bsym]}"))
            else:
                ok_intents.append(it)

        last_px = {s: float(pipe.prices[s].iloc[i]) for s in pipe.symbols}
        decision = self.desk.review(ok_intents, acct, now, last_px, self.cfg.instruments)
        rep.rejected += [(f"{it.symbol} {it.side}", why) for it, why in decision.rejected]
        rep.tier, rep.halted = decision.tier, decision.halted

        rep.published = publish_intents(self.con, decision.approved, asof, last_px, self.magic, self.ttl, self.symbol_map)
        self.ledger.append("CYCLE", {"asof": asof, "equity": acct.equity, "tier": rep.tier, "published": rep.published,
                                     "rejected": len(rep.rejected), "targets": snap.target_positions})
        store.set_value(self.con, "last_asof", asof)
        save_ladder(self.con, self.ladder)
        return rep
