import json
import logging
import math
import time
from typing import List, Dict, Any
from strategy import fetch_top_volume_symbols, EXECUTION_TIMEFRAME
from backtest import fetch_historical_klines_paginated, BacktestEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dynamic_asset_filter")

# Vetting Criteria Parameters
MIN_NET_PNL = 0.0          # Must produce positive net PnL in recent backtest
MIN_PROFIT_FACTOR = 1.10   # Must have Profit Factor >= 1.10
MIN_WIN_RATE = 40.0        # Must have Win Rate >= 40%
VETTING_KLINES = 1500      # Recent historical window for vetting backtest
OUTPUT_JSON_FILE = "vetted_symbols.json"

def run_dynamic_market_scan(top_n: int = 40) -> List[str]:
    """
    Scans Top N Volume pairs, runs quick historical backtests, and filters for winning symbols.
    """
    logger.info(f"Starting 24h Dynamic Asset Vetting Routine across Top {top_n} Volume symbols...")
    candidate_symbols = fetch_top_volume_symbols(limit=top_n)
    vetted_assets = []
    performance_records = []

    for idx, sym in enumerate(candidate_symbols, 1):
        logger.info(f"[{idx}/{len(candidate_symbols)}] Vetting candidate pair: {sym}...")
        df_klines = fetch_historical_klines_paginated(
            symbol=sym,
            timeframe=EXECUTION_TIMEFRAME,
            total_klines=VETTING_KLINES
        )

        if df_klines.empty or len(df_klines) < 200:
            logger.warning(f"Skipping {sym}: Insufficient historical kline data.")
            continue

        engine = BacktestEngine(symbol=sym, initial_balance=10000.0)
        res = engine.run(df_klines, warmup_bars=60)

        if res.get("status") == "NO_TRADES_EXECUTED":
            logger.info(f"Skipping {sym}: No trades triggered in vetting window.")
            continue

        net_pnl = res.get("net_pnl", 0.0)
        pf = res.get("profit_factor", 0.0)
        win_rate = res.get("win_rate_pct", 0.0)

        # Apply Vetting Pass Filters
        is_passed = (
            net_pnl > MIN_NET_PNL and 
            (math.isnan(pf) or pf >= MIN_PROFIT_FACTOR) and 
            win_rate >= MIN_WIN_RATE
        )

        status_str = "PASSED" if is_passed else "FAILED"
        logger.info(f"-> {sym} {status_str} | PnL: ${net_pnl:.2f} | PF: {pf} | Win Rate: {win_rate:.1f}%")

        record = {
            "symbol": sym,
            "net_pnl": net_pnl,
            "profit_factor": pf,
            "win_rate_pct": win_rate,
            "total_trades": res.get("total_trades", 0),
            "status": status_str
        }
        performance_records.append(record)

        if is_passed:
            vetted_assets.append(sym)

    # Save vetted symbols to JSON file for live bot evaluation
    output_data = {
        "timestamp": time.time(),
        "vetted_count": len(vetted_assets),
        "vetted_symbols": vetted_assets,
        "details": performance_records
    }

    with open(OUTPUT_JSON_FILE, "w") as f:
        json.dump(output_data, f, indent=4)

    logger.info(f"\n=== VETTING COMPLETE: {len(vetted_assets)} / {len(candidate_symbols)} Symbols Qualified ===")
    logger.info(f"Vetted Symbols: {vetted_assets}")
    logger.info(f"Output saved to '{OUTPUT_JSON_FILE}'.")

    return vetted_assets


if __name__ == "__main__":
    run_dynamic_market_scan(top_n=40)