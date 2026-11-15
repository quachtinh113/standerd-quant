"""Kênh phát cảnh báo. Có chống lặp theo (code) trong cooldown để không spam."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Protocol

from .health import Alert


class AlertSink(Protocol):
    def emit(self, alert: Alert) -> None: ...


class ConsoleSink:
    def emit(self, alert: Alert) -> None:
        print(f"[{alert.level}] {alert.code}: {alert.message}", file=sys.stderr)


class FileSink:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, alert: Alert) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "level": alert.level, "code": alert.code, "message": alert.message}) + "\n")


class Dispatcher:
    def __init__(self, sinks: list[AlertSink], cooldown_s: float = 900.0):
        self.sinks, self.cooldown = sinks, cooldown_s
        self._last: dict[str, float] = {}

    def dispatch(self, alerts: list[Alert], now: float | None = None) -> list[Alert]:
        """Trả danh sách cảnh báo thực sự được phát (CRITICAL luôn phát lần đầu, rồi theo cooldown)."""
        now, sent = now or time.time(), []
        for a in alerts:
            if now - self._last.get(a.code, -1e18) < self.cooldown:
                continue
            self._last[a.code] = now
            for s in self.sinks:
                s.emit(a)
            sent.append(a)
        return sent
