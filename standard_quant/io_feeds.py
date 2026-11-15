"""Nguồn dữ liệu & tài khoản cho bot. CsvFeed/StaticAccount dùng cho test/paper; MT5Feed/MT5Account là adapter
MetaTrader5 (import lười, CHƯA kiểm chứng trên terminal thật)."""
from __future__ import annotations

from typing import Callable, Protocol

import pandas as pd

from .account import AccountState, Pos


class DataFeed(Protocol):
    def load(self) -> dict[str, pd.Series]: ...       # close D1, chỉ gồm các nến ĐÃ ĐÓNG


class AccountFeed(Protocol):
    def state(self) -> AccountState: ...


class CsvFeed:
    def __init__(self, paths: dict[str, str]):
        self.paths = paths

    def load(self) -> dict[str, pd.Series]:
        from .s1_data_rules import load_csv_close
        return {s: load_csv_close(p) for s, p in self.paths.items()}


class CallableFeed:
    """Bọc một hàm trả dict giá — tiện cho test và cho nguồn dữ liệu tuỳ biến."""
    def __init__(self, fn: Callable[[], dict[str, pd.Series]]):
        self.fn = fn

    def load(self) -> dict[str, pd.Series]:
        return self.fn()


class StaticAccount:
    def __init__(self, equity: float, held: dict[str, Pos] | None = None, margin: float = 0.0, margin_level: float = 0.0):
        self.s = AccountState(equity, margin, margin_level, dict(held or {}))

    def state(self) -> AccountState:
        return self.s


class MT5Feed:
    """D1 close từ MT5. Bỏ nến hiện tại (đang hình thành) → chỉ nến đã đóng (zero look-ahead)."""
    def __init__(self, symbol_map: dict[str, str], bars: int = 1500):
        self.map, self.bars, self.mt5 = symbol_map, bars, None

    def load(self) -> dict[str, pd.Series]:
        import MetaTrader5 as mt5
        self.mt5 = mt5
        out = {}
        for name, broker_sym in self.map.items():
            r = mt5.copy_rates_from_pos(broker_sym, mt5.TIMEFRAME_D1, 1, self.bars)     # start_pos=1: bỏ nến đang chạy
            if r is None or len(r) == 0:
                raise RuntimeError(f"không lấy được dữ liệu {broker_sym}: {mt5.last_error()}")
            df = pd.DataFrame(r)
            out[name] = pd.Series(df["close"].to_numpy(float), index=pd.to_datetime(df["time"], unit="s"), name=name)
        return out


class MT5Account:
    """Equity/margin + vị thế ròng theo symbol, lọc đúng magic của bot."""
    def __init__(self, symbol_map: dict[str, str], magic: int):
        self.map, self.magic = symbol_map, magic

    def state(self) -> AccountState:
        import MetaTrader5 as mt5
        info = mt5.account_info()
        if info is None:
            raise RuntimeError(f"account_info lỗi: {mt5.last_error()}")
        rev = {v: k for k, v in self.map.items()}
        net: dict[str, list[float]] = {}
        for p in mt5.positions_get() or []:
            if p.magic != self.magic or p.symbol not in rev:
                continue
            sgn = 1.0 if p.type == mt5.POSITION_TYPE_BUY else -1.0
            acc = net.setdefault(rev[p.symbol], [0.0, 0.0])        # [vol có dấu, Σ vol×giá]
            acc[0] += sgn * p.volume
            acc[1] += p.volume * p.price_open
        held = {s: Pos(v, (w / abs(v)) if v else 0.0) for s, (v, w) in net.items() if v}
        return AccountState(info.equity, info.margin, info.margin_level, held)
