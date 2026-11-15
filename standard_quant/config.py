"""Cấu hình một nơi duy nhất (JSON). Không có magic number rải rác trong các khối."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .contracts import Instrument


@dataclass
class Config:
    capital: float = 10_000.0
    vol_target_annual: float = 0.20
    # Khối 2
    ewmac_speeds: list[int] = field(default_factory=lambda: [8, 16, 32, 64])
    breakout_lookbacks: list[int] = field(default_factory=lambda: [40, 80])
    alphas: list[str] = field(default_factory=list)   # tên trong s2_forecast.alpha.REGISTRY
    # L3 Meta-labeling (lọc nhiễu ML)
    meta_enabled: bool = False
    meta_threshold: float = 0.5
    meta_mode: str = "gate"                  # gate | continuous
    meta_burn_in: str = "pass"               # pass | flat: xử lý giai đoạn chưa đủ dữ liệu huấn luyện
    meta_min_events: int = 400
    # Khối 3
    buffer_fraction: float = 0.10
    # Khối 4
    weights_method: str = "equal"
    idm_cap: float = 2.5
    # Khối 5 / Risk
    sl_vol_mult: float = 6.0
    exec_lag: int = 1                        # 1 = khớp ở phiên kế tiếp (thận trọng)
    dd_tier1: float = 0.03
    dd_tier2: float = 0.05
    dd_tier3: float = 0.075
    daily_loss_cap: float | None = None
    halt_cooldown_days: int = 20             # backtest: số phiên chờ trước khi mở lại sau halt DD
    max_dca_layers: int = 0                  # bất biến: luôn 0
    instruments: dict[str, Instrument] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_dca_layers != 0:
            raise ValueError("max_dca_layers phải bằng 0 (cấm martingale/DCA)")
        if not 0 < self.vol_target_annual <= 0.5:
            raise ValueError("vol_target_annual ngoài (0, 0.5]")
        if self.exec_lag < 1:
            raise ValueError("exec_lag >= 1 (không khớp trên chính nến tạo tín hiệu)")

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        insts = {s: Instrument(symbol=s, **v) for s, v in raw.pop("instruments", {}).items()}
        return cls(instruments=insts, **raw)
