"""python -m standard_quant.l6_monitoring --db signals.db [--ledger ledger.jsonl] [--out report.json]
Mã thoát: 0=OK, 1=WARNING, 2=CRITICAL — dùng được với Task Scheduler/cron để báo động."""
import argparse
import json
import sys

from ..l5_execution import store
from .alerts import ConsoleSink, Dispatcher, FileSink
from .health import resolve_unknown
from .report import build_report, write_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--ledger")
    ap.add_argument("--out")
    ap.add_argument("--alerts-file")
    ap.add_argument("--no-worker", action="store_true", help="không kỳ vọng worker L5 đang chạy")
    ap.add_argument("--resolve", nargs=3, metavar=("SIGNAL_ID", "SENT|NOT_SENT", "NOTE"),
                    help="người xác nhận kết quả một lệnh UNKNOWN")
    a = ap.parse_args()
    con = store.connect(a.db)
    if a.resolve:
        from ..ledger import HashChainLedger
        resolve_unknown(con, a.resolve[0], a.resolve[1], a.resolve[2], HashChainLedger(a.ledger) if a.ledger else None)
    rep = build_report(con, a.ledger, expect_worker=not a.no_worker)
    if a.out:
        write_report(rep, a.out)
    sinks = [ConsoleSink()] + ([FileSink(a.alerts_file)] if a.alerts_file else [])
    from .health import Alert
    Dispatcher(sinks, cooldown_s=0).dispatch([Alert(**x) for x in rep["alerts"]])
    print(json.dumps({k: rep[k] for k in ("status", "signals", "latency", "halt")}, indent=2, default=str))
    sys.exit({"OK": 0, "WARNING": 1, "CRITICAL": 2}[rep["status"]])


if __name__ == "__main__":
    main()
