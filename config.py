import json
import os
import sys
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("trading_agent")

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_SECRET_KEY = os.getenv("BYBIT_SECRET_KEY")
BYBIT_TESTNET = os.getenv("BYBIT_TESTNET", "true").lower() == "true"

CONFIG_FILE = "config.json"
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as f:
        STRATEGY_CONFIG = json.load(f)
else:
    STRATEGY_CONFIG = {}

ACCOUNT_BALANCE = STRATEGY_CONFIG.get("account_balance", 100.0)
RISK_PCT = STRATEGY_CONFIG.get("risk_pct", 1.0)
LEVERAGE = STRATEGY_CONFIG.get("leverage", 5)
ENABLE_LIVE_TRADING = STRATEGY_CONFIG.get("enable_live_trading", True)
WATCHLIST = STRATEGY_CONFIG.get("watchlist", [])

ALLOW_MAINNET_LIVE = os.getenv("ALLOW_MAINNET_LIVE", "false").lower() == "true"
if not BYBIT_TESTNET and ENABLE_LIVE_TRADING and not ALLOW_MAINNET_LIVE:
    logger.critical("🚨 SAFETY INTERVENTION: Attempted to run LIVE TRADING on BYBIT MAINNET without ALLOW_MAINNET_LIVE=true in .env! Aborting execution.")
    sys.exit(1)


def format_ccxt_symbol(symbol: str) -> str:
    clean = symbol.replace("/", "").upper()
    if not clean.endswith(":USDT"):
        return f"{clean[:-4]}/USDT:USDT" if clean.endswith("USDT") else symbol
    return symbol


WATCHLIST = [format_ccxt_symbol(s) for s in STRATEGY_CONFIG.get("watchlist", [])]