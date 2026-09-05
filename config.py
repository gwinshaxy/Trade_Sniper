import json
import os
from dotenv import load_dotenv

# Load API credentials from .env file
load_dotenv()

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_SECRET_KEY = os.getenv("BYBIT_SECRET_KEY")
BYBIT_TESTNET = os.getenv("BYBIT_TESTNET", "true").lower() == "true"

# Load strategy and risk parameters from config.json
CONFIG_FILE = "config.json"

if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as f:
        STRATEGY_CONFIG = json.load(f)
else:
    STRATEGY_CONFIG = {}

# Unpack key settings for easy import across modules
ACCOUNT_BALANCE = STRATEGY_CONFIG.get("account_balance", 100.0)
RISK_PCT = STRATEGY_CONFIG.get("risk_pct", 0.5)
LEVERAGE = STRATEGY_CONFIG.get("leverage", 100)
ENABLE_LIVE_TRADING = STRATEGY_CONFIG.get("enable_live_trading", False)
WATCHLIST = STRATEGY_CONFIG.get("watchlist", [])

def format_ccxt_symbol(symbol: str) -> str:
    clean = symbol.replace("/", "").upper()
    if not clean.endswith(":USDT"):
        return f"{clean[:-4]}/USDT:USDT" if clean.endswith("USDT") else symbol
    return symbol

WATCHLIST = [format_ccxt_symbol(s) for s in STRATEGY_CONFIG.get("watchlist", [])]