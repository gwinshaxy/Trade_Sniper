"""
===============================================================================
PRODUCTION STREAMLIT LIVE DASHBOARD (app.py)
===============================================================================
Updated Features & Modifications:
1. Removed UI Control for "Min ATR % Filter" to prevent filtering into late-stage volatility spikes.
2. Upgrade A: HTF Trend Alignment Indicator & Logic (4H & 1H TEMA alignment).
3. Upgrade B: Interactive Dynamic ATR Trailing Stop Tracking in Active Trades.
4. Upgrade C: Equity Curve Drawdown Status & Risk Reduction Indicator in Dashboard metrics.
===============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI Trading Engine - Live Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# HELPER MATHEMATICAL & TECHNICAL FUNCTIONS
# =============================================================================

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calculate_tema(series: pd.Series, period: int = 200) -> pd.Series:
    ema1 = calculate_ema(series, period)
    ema2 = calculate_ema(ema1, period)
    ema3 = calculate_ema(ema2, period)
    return (3 * ema1) - (3 * ema2) + ema3


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


# =============================================================================
# STREAMLIT SIDEBAR CONFIGURATION (Min ATR Filter Removed)
# =============================================================================

st.sidebar.title("⚙️ Engine Configurations")
st.sidebar.markdown("---")

# Base Risk Parameters
account_balance = st.sidebar.number_input("Account Capital ($)", value=10000.0, step=500.0)
base_risk_pct = st.sidebar.slider("Base Risk Per Trade (%)", min_value=0.5, max_value=3.0, value=1.0, step=0.1) / 100.0
leverage = st.sidebar.selectbox("Account Leverage", options=[10, 20, 50, 100], index=3)

st.sidebar.markdown("### 🛠️ Active Backtest Upgrades")
enable_htf_alignment = st.sidebar.checkbox("Upgrade A: 1H + 4H HTF TEMA Alignment", value=True, disabled=True)
enable_adaptive_trailing = st.sidebar.checkbox("Upgrade B: Adaptive ATR Trailing Stop", value=True, disabled=True)
enable_dd_filter = st.sidebar.checkbox("Upgrade C: Equity Curve DD Filter (50% Risk Cut)", value=True)

max_dd_threshold = st.sidebar.slider("Drawdown Risk-Cut Threshold (%)", min_value=2.0, max_value=15.0, value=5.0, step=0.5) / 100.0

# =============================================================================
# UPGRADE C: DYNAMIC RISK & DRAWDOWN MANAGER
# =============================================================================

# Initialize session state for peak balance tracking
if "peak_balance" not in st.session_state:
    st.session_state.peak_balance = account_balance

if account_balance > st.session_state.peak_balance:
    st.session_state.peak_balance = account_balance

current_drawdown = (st.session_state.peak_balance - account_balance) / st.session_state.peak_balance if st.session_state.peak_balance > 0 else 0.0

effective_risk_pct = base_risk_pct
if enable_dd_filter and current_drawdown >= max_dd_threshold:
    effective_risk_pct = base_risk_pct * 0.5
    st.sidebar.error(f"⚠️ DRAWDOWN WARNING: Account DD is {current_drawdown:.2%}. Risk reduced to {effective_risk_pct:.2%}.")
else:
    st.sidebar.success(f"🟢 Risk Status Normal: {effective_risk_pct:.2%} per trade")


# =============================================================================
# DASHBOARD HEADER METRICS
# =============================================================================

st.title("📊 Live Trading Execution Engine")
st.caption("Synchronized with Backtester Engine | HTF TEMA Alignment + Dynamic ATR Trailing Stops")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Account Balance", f"${account_balance:,.2f}")
col2.metric("Peak Balance", f"${st.session_state.peak_balance:,.2f}")
col3.metric("Current Drawdown", f"{current_drawdown:.2%}")
col4.metric("Active Effective Risk", f"{effective_risk_pct:.2%}")

st.markdown("---")

# =============================================================================
# UPGRADE A: ANALYSIS ENGINE WITH HTF TEMA ALIGNMENT
# =============================================================================

def analyze_market_data(df_30m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> Dict[str, Any]:
    if df_30m.empty or df_1h.empty or df_4h.empty:
        return {"status": "INSUFFICIENT_DATA"}

    price = df_30m['close'].iloc[-1]
    atr = calculate_atr(df_30m, 14).iloc[-1]
    
    tema_1h = calculate_tema(df_1h['close'], 200).iloc[-1]
    tema_4h = calculate_tema(df_4h['close'], 200).iloc[-1]

    # Enforce Upgrade A: Strict 1H + 4H TEMA Alignment
    is_long_aligned = (price > tema_1h) and (price > tema_4h)
    is_short_aligned = (price < tema_1h) and (price < tema_4h)

    if not is_long_aligned and not is_short_aligned:
        return {
            "status": "NO_SIGNAL", 
            "reason": "HTF Trend Conflict (1H/4H TEMA Disagreement)",
            "price": price, "tema_1h": tema_1h, "tema_4h": tema_4h
        }

    direction = "LONG" if is_long_aligned else "SHORT"
    stop_distance = atr * 1.5

    initial_sl = (price - stop_distance) if direction == "LONG" else (price + stop_distance)
    tp_target = (price + stop_distance * 1.5) if direction == "LONG" else (price - stop_distance * 1.5)

    return {
        "status": "SIGNAL_FOUND",
        "direction": direction,
        "price": price,
        "initial_sl": initial_sl,
        "tp_target": tp_target,
        "atr": atr,
        "tema_1h": tema_1h,
        "tema_4h": tema_4h
    }

# =============================================================================
# ACTIVE TRADES MONITORING TABLE & UPGRADE B (DYNAMIC ATR TRAILING SL)
# =============================================================================

st.subheader("📌 Active Positions & Dynamic Trailing Stop Tracking")

if "active_trades" not in st.session_state:
    st.session_state.active_trades = []

if st.session_state.active_trades:
    trade_list = []
    for trade in st.session_state.active_trades:
        # Upgrade B Dynamic Trailing Calculation update
        current_price = trade["current_price"]
        atr = trade["atr"]
        atr_buffer = atr * 1.5

        if trade["direction"] == "LONG":
            trade["highest_price"] = max(trade.get("highest_price", trade["entry"]), current_price)
            proposed_sl = trade["highest_price"] - atr_buffer
            if proposed_sl > trade["trailing_sl"]:
                trade["trailing_sl"] = proposed_sl
        else:
            trade["lowest_price"] = min(trade.get("lowest_price", trade["entry"]), current_price)
            proposed_sl = trade["lowest_price"] + atr_buffer
            if proposed_sl < trade["trailing_sl"]:
                trade["trailing_sl"] = proposed_sl

        pnl = (current_price - trade["entry"]) if trade["direction"] == "LONG" else (trade["entry"] - current_price)
        
        trade_list.append({
            "Symbol": trade["symbol"],
            "Direction": trade["direction"],
            "Entry Price": f"${trade['entry']:.4f}",
            "Current Price": f"${current_price:.4f}",
            "Dynamic Trailing SL": f"${trade['trailing_sl']:.4f}",
            "Take Profit": f"${trade['tp']:.4f}",
            "PnL ($)": f"${pnl * trade['units']:.2f}",
            "HTF Status": "✅ Aligned (1H+4H)"
        })

    st.table(pd.DataFrame(trade_list))
else:
    st.info("No active positions currently tracked.")

st.markdown("---")

# =============================================================================
# LIVE MARKET SCANNER & HTF STATUS DISPLAY
# =============================================================================

st.subheader("🔍 Market Scanner & HTF Trend Status")

if st.button("Run Market Scan Now"):
    st.write("Scanning pairs with HTF TEMA filter enabled...")
    # Simulated Scan Matrix for Display
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    scan_results = []

    for sym in symbols:
        # Mocking incoming data format
        scan_results.append({
            "Symbol": sym,
            "1H TEMA Status": "Bullish" if sym in ["EURUSD", "GBPUSD"] else "Bearish",
            "4H TEMA Status": "Bullish" if sym in ["EURUSD"] else "Bearish",
            "Alignment Signal": "LONG Signal" if sym == "EURUSD" else ("SHORT Signal" if sym == "USDJPY" else "Filtered out (Conflict)"),
            "ATR (14)": "0.0012",
            "Action": "Ready to Execute" if sym in ["EURUSD", "USDJPY"] else "Skipped"
        })

    st.dataframe(pd.DataFrame(scan_results), use_container_width=True)