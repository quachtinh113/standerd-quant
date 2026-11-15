"""Cổng gửi lệnh: Protocol + FakeGateway (test) + MT5Gateway (mỏng, import MetaTrader5 lười).

Gateway KHÔNG tính toán gì ngoài việc dựng request và đọc retcode. Chưa được kiểm thử trên tài khoản thật.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from .store import Signal


@dataclass
class SendResult:
    ok: bool
    retcode: int = 0
    ticket: int = 0
    fill_price: float = 0.0
    comment: str = ""
    unknown: bool = False      # timeout/ngắt kết nối: lệnh CÓ THỂ đã vào sàn → không retry, không coi là thất bại


class Gateway(Protocol):
    def connect(self) -> None: ...
    def last_price(self, symbol: str, side: str) -> float: ...
    def send(self, sig: Signal) -> SendResult: ...


class FakeGateway:
    """Giả lập cho test: độ trễ, từ chối, timeout cấu hình được."""

    def __init__(self, price: float = 100.0, delay_s: float = 0.0, retcodes: list[int] | None = None,
                 timeout: bool = False, slip: float = 0.0):
        self.price, self.delay_s, self.retcodes, self.timeout, self.slip = price, delay_s, list(retcodes or []), timeout, slip
        self.sent: list[Signal] = []
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def last_price(self, symbol: str, side: str) -> float:
        return self.price

    def send(self, sig: Signal) -> SendResult:
        self.sent.append(sig)
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.timeout:
            return SendResult(False, 0, 0, 0.0, "timeout", unknown=True)
        code = self.retcodes.pop(0) if self.retcodes else 10009
        ok = code in (10008, 10009)
        px = self.price + (self.slip if sig.side == "BUY" else -self.slip)
        return SendResult(ok, code, 1000 + len(self.sent), px if ok else 0.0, "fake")


RETCODE_OK = (10008, 10009)
RETCODE_RETRYABLE = (10004, 10020)         # requote, price changed — retry an toàn: lệnh chưa khớp


class MT5Gateway:
    """Adapter MetaTrader5 (hedging account). Thiết kế cho Exness: filling mode lấy động từ symbol_info."""

    def __init__(self, deviation_points: int = 20, login: int | None = None, password: str | None = None,
                 server: str | None = None):
        self.deviation, self._cred = deviation_points, (login, password, server)
        self.mt5 = None

    def connect(self) -> None:
        import MetaTrader5 as mt5                       # import lười: máy dev không có MT5 vẫn chạy test
        self.mt5 = mt5
        login, password, server = self._cred
        ok = mt5.initialize(login=login, password=password, server=server) if login else mt5.initialize()
        if not ok:
            raise RuntimeError(f"MT5 initialize thất bại: {mt5.last_error()}")

    def last_price(self, symbol: str, side: str) -> float:
        t = self.mt5.symbol_info_tick(symbol)
        return 0.0 if t is None else (t.ask if side == "BUY" else t.bid)

    def _filling(self, symbol: str) -> int:
        m, info = self.mt5, self.mt5.symbol_info(symbol)
        mode = getattr(info, "filling_mode", 0) if info else 0
        if mode & 2:
            return m.ORDER_FILLING_IOC
        if mode & 1:
            return m.ORDER_FILLING_FOK
        return m.ORDER_FILLING_RETURN

    def _request(self, sig: Signal, price: float, position: int | None = None) -> dict:
        m = self.mt5
        req = {"action": m.TRADE_ACTION_DEAL, "symbol": sig.symbol, "volume": float(sig.volume),
               "type": m.ORDER_TYPE_BUY if sig.side == "BUY" else m.ORDER_TYPE_SELL, "price": price,
               "deviation": self.deviation, "magic": sig.magic, "comment": f"L5:{sig.signal_id}"[:31],
               "type_time": m.ORDER_TIME_GTC, "type_filling": self._filling(sig.symbol)}
        if position is not None:
            req["position"] = position                  # đóng đúng vị thế (hedging) thay vì mở vị thế ngược
        else:
            req["sl"], req["tp"] = float(sig.sl), float(sig.tp)   # SL vật lý nằm TRONG lệnh vào
        return req

    def _one(self, req: dict) -> SendResult:
        res = self.mt5.order_send(req)
        if res is None:
            return SendResult(False, 0, 0, 0.0, str(self.mt5.last_error()), unknown=True)
        return SendResult(res.retcode in RETCODE_OK, res.retcode, getattr(res, "order", 0),
                          getattr(res, "price", 0.0), getattr(res, "comment", ""))

    def send(self, sig: Signal) -> SendResult:
        m = self.mt5
        if sig.reduce_only:                             # đóng: tìm vị thế đối ứng của đúng magic
            want = m.POSITION_TYPE_SELL if sig.side == "BUY" else m.POSITION_TYPE_BUY
            left, last = sig.volume, SendResult(False, 0, 0, 0.0, "no position")
            for p in m.positions_get(symbol=sig.symbol) or []:
                if p.magic != sig.magic or p.type != want or left <= 0:
                    continue
                vol = min(left, p.volume)
                part = Signal(sig.signal_id, sig.symbol, sig.side, vol, 0, 0, sig.ref_price, True, sig.magic)
                last = self._one(self._request(part, self.last_price(sig.symbol, sig.side), p.ticket))
                if not last.ok:
                    return last
                left -= vol
            return last
        return self._one(self._request(sig, self.last_price(sig.symbol, sig.side)))
