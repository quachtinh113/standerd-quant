"""Chạy bot (phía quyết định). L5 chạy riêng:  python -m standard_quant.l5_execution ...

  python run_bot.py --db data/signals.db --feed csv --csv XAUUSD=gold.csv --once
  python run_bot.py --db data/signals.db --feed mt5 --map XAUUSD=XAUUSDm --magic 260910 --loop
Mặc định KHÔNG gửi lệnh thật: bot chỉ ghi tín hiệu vào hộp thư; lệnh chỉ đi khi worker L5 --gateway mt5 đang chạy.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from standard_quant import Config
from standard_quant.account import Pos
from standard_quant.bot import Bot
from standard_quant.io_feeds import CsvFeed, MT5Account, MT5Feed, StaticAccount
from standard_quant.l4_risk import RiskLimits


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).parent / "config" / "standard.json"))
    ap.add_argument("--db", required=True)
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--feed", choices=["csv", "mt5"], required=True)
    ap.add_argument("--csv", nargs="*", default=[], help="SYMBOL=path.csv")
    ap.add_argument("--map", nargs="*", default=[], help="SYMBOL=BROKER_SYMBOL (mt5)")
    ap.add_argument("--magic", type=int, default=260910)
    ap.add_argument("--equity", type=float, default=None, help="chỉ cho --feed csv (paper)")
    ap.add_argument("--halt-file", default=None)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=300)
    a = ap.parse_args()

    cfg = Config.load(a.config)
    ledger = a.ledger or str(Path(a.db).with_suffix(".ledger.jsonl"))
    smap = dict(kv.split("=", 1) for kv in a.map)
    if a.feed == "csv":
        feed = CsvFeed(dict(kv.split("=", 1) for kv in a.csv))
        acct = StaticAccount(a.equity or cfg.capital)
    else:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            raise SystemExit(f"MT5 initialize thất bại: {mt5.last_error()}")
        feed, acct = MT5Feed(smap), MT5Account(smap, a.magic)
    lim = RiskLimits(halt_file=Path(a.halt_file) if a.halt_file else None)
    bot = Bot(cfg, a.db, feed, acct, magic=a.magic, ledger_path=ledger, limits=lim, symbol_map=smap)
    while True:
        r = bot.run_cycle()
        print(f"{r.status} asof={r.asof} tier={r.tier} halted={r.halted} published={r.published} rejected={r.rejected}")
        if a.once or not a.loop:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
