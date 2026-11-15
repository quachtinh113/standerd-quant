"""Tiến trình L5 độc lập:  python -m standard_quant.l5_execution --db signals.db --symbols XAUUSDm --gateway mt5"""
import argparse
import json

from . import store
from .executor import ExecLimits, Executor, latency_summary
from .gateway import FakeGateway, MT5Gateway


def main() -> None:
    ap = argparse.ArgumentParser(description="L5 execution worker")
    ap.add_argument("--db", required=True)
    ap.add_argument("--symbols", nargs="+", required=True, help="allowlist; bắt buộc")
    ap.add_argument("--gateway", choices=["mt5", "fake"], default="fake")
    ap.add_argument("--max-lot", type=float, default=1.0)
    ap.add_argument("--max-slip-bps", type=float, default=30.0)
    ap.add_argument("--magic", type=int, nargs="*", default=[])
    ap.add_argument("--poll-ms", type=int, default=100)
    ap.add_argument("--summary", action="store_true", help="in thống kê latency rồi thoát")
    a = ap.parse_args()

    con = store.connect(a.db)
    if a.summary:
        print(json.dumps({"latency": latency_summary(con), "status": store.status_counts(con)}, indent=2))
        return
    gw = MT5Gateway() if a.gateway == "mt5" else FakeGateway()
    lim = ExecLimits(frozenset(a.symbols), a.max_lot, a.max_slip_bps, True, 1, frozenset(a.magic))
    Executor(con, gw, lim).run_forever(a.poll_ms)


if __name__ == "__main__":
    main()
