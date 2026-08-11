import os
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
from lightweight_charts.widgets import StreamlitChart

from app_manager import start_background_tasks

# Initialize background tasks once on startup
start_background_tasks()

# ==========================================
# AUTHENTICATION GATE
# ==========================================
st.set_page_config(page_title="Forex & Crypto Trading Terminal", layout="wide")
load_dotenv()

def check_password():
    """Returns `True` if the user entered the correct password."""
    # Retrieve password from environment variable (default to 'securepass123' if not set)
    expected_password = os.getenv("DASHBOARD_PASSWORD", "securepass123")

    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if st.session_state["password_correct"]:
        return True

    # Show inputs for password
    st.markdown("## 🔒 Restricted Access")
    st.markdown("Please enter the password to access the Trading Terminal.")
    
    password_input = st.text_input("Password", type="password")
    if st.button("Login"):
        if password_input == expected_password:
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("😕 Password incorrect")
    return False

if not check_password():
    st.stop()  # Halts rendering of the rest of the dashboard until authenticated

# ==========================================
# REST OF DASHBOARD INITIALIZATION & LOGIC
# ==========================================
from common import (
    calculate_pnl,
    ensure_schema_updated,
    get_db_connection,
    send_telegram_notification,
)
import strategy

ensure_schema_updated()

# ==========================================
# UNIFIED SYMBOL CONFIGURATION
# ==========================================
symbols_env = os.getenv("SYMBOLS") or os.getenv(
    "SYMBOL", "BTC/USDT,ETH/USDT,BNB/USDT,ADA/USDT,SOL/USDT"
)
env_symbols = [s.strip().upper() for s in symbols_env.split(",")]
env_symbols = [
    s if "/" in s else f"{s[:-4]}/{s[-4:]}" if s.endswith("USDT") else s
    for s in env_symbols
]

# ==========================================
# FETCH DATABASE RECORDS FOR FILTERS
# ==========================================
conn = get_db_connection()
df_all_trades = pd.read_sql("SELECT * FROM trade_setups ORDER BY id DESC;", conn)
conn.close()

db_pairs = (
    df_all_trades['pair'].unique().tolist() if not df_all_trades.empty else []
)
available_pairs = list(dict.fromkeys(env_symbols + db_pairs))
for default_pair in ["BTC/USDT", "ETH/USDT", "EUR/USD", "GBP/USD", "XRP/USDT"]:
    if default_pair not in available_pairs:
        available_pairs.append(default_pair)

# ==========================================
# SIDEBAR CONTROLS & STRATEGY TUNING
# ==========================================
st.sidebar.subheader("🎛️ Dashboard Controls & Strategy Tuning")

st.sidebar.markdown("### Currency Pair Filter")
select_all_pairs = st.sidebar.checkbox("Select All", value=True)
if select_all_pairs:
    selected_pairs = available_pairs
else:
    selected_pairs = st.sidebar.multiselect(
        "Filter Pairs", available_pairs, default=available_pairs
    )

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Strategy Parameters (Harmonized)")

# Asset selection for loading DEAP parameters
config_target_pair = st.sidebar.selectbox("Load Dynamic Config For:", available_pairs, index=0)
dyn_cfg = strategy.load_symbol_config(config_target_pair)

tema_period = st.sidebar.number_input(
    "TEMA Period", value=int(dyn_cfg.get("tema_period", 200)), min_value=10, max_value=500
)
rsi_period = st.sidebar.number_input(
    "RSI Period", value=int(dyn_cfg.get("rsi_period", 14)), min_value=2, max_value=50
)
rsi_thresh = st.sidebar.slider(
    "RSI Threshold", min_value=20, max_value=80, value=int(dyn_cfg.get("rsi_thresh", 42))
)
zone_tolerance_pct = st.sidebar.slider(
    "Volume Zone Proximity (%)",
    min_value=0.1,
    max_value=3.0,
    value=float(dyn_cfg.get("zone_tolerance", 0.0075)) * 100.0,
    step=0.05,
)

vp_detection_pct = st.sidebar.slider(
    "Volume Gap Detection (%)",
    min_value=1.0,
    max_value=20.0,
    value=float(dyn_cfg.get("vp_detection_pct", 0.07)) * 100.0,
    step=0.5,
) / 100.0

use_rsi_filter = st.sidebar.checkbox("Enable RSI Momentum Filter", value=bool(dyn_cfg.get("use_rsi_filter", True)))
use_candlestick_confirm = st.sidebar.checkbox(
    "Require Candlestick Confirmation", value=bool(dyn_cfg.get("use_candlestick_confirm", True))
)

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 AI Fundamental Layer")
use_sentiment_filter = st.sidebar.checkbox(
    "Enable AI Sentiment Filter", value=False
)
min_sentiment_thresh = st.sidebar.slider(
    "Min Sentiment Threshold", min_value=0.0, max_value=0.8, value=0.2, step=0.05
)

st.sidebar.markdown("---")
st.sidebar.subheader("📝 Manual Trade Setup")

pair = st.sidebar.text_input("Execution Pair", "BTC/USDT").upper()
direction = st.sidebar.selectbox("Order Type / Direction", ["LONG", "SHORT"])
overlay_chart = st.sidebar.checkbox(
    "Overlay Trade Positions on Chart", value=True
)
overlay_gaps = st.sidebar.checkbox(
    "Overlay Volume Profile Gaps", value=True
)

entry_price = st.sidebar.number_input(
    "Entry Price ($)", min_value=0.0, format="%.5f"
)
stop_loss = st.sidebar.number_input(
    "Stop Loss ($)", min_value=0.0, format="%.5f"
)
take_profit = st.sidebar.number_input(
    "Take Profit ($)", min_value=0.0, format="%.5f"
)

risk = abs(entry_price - stop_loss) if (entry_price and stop_loss) else 0.0
reward = abs(take_profit - entry_price) if (take_profit and entry_price) else 0.0
rr_ratio = round(reward / risk, 2) if risk > 0 else 0.0
st.sidebar.markdown(f"**Risk : Reward Ratio:** `1:{rr_ratio}`")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Account Parameters")
account_balance = st.sidebar.number_input(
    "Account Balance ($)", min_value=1.0, value=float(os.getenv("ACCOUNT_BALANCE", 10000.0))
)
risk_pct = st.sidebar.number_input(
    "Risk Per Trade (%)", min_value=0.01, max_value=100.0, value=1.0
)

calc_position_size = (
    round((account_balance * (risk_pct / 100.0)) / risk, 4) if risk > 0 else 0.0
)
st.sidebar.caption(f"Calculated Position Size: `{calc_position_size}` units")

submitted = st.sidebar.button("🚀 Execute Order")
if submitted:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
            INSERT INTO trade_setups 
            (pair, direction, entry_price, stop_loss, take_profit, risk_reward_ratio, position_size, account_balance, risk_pct, status, trade_state)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'EXECUTED', 'OPEN');
        """,
        (
            pair,
            direction,
            entry_price,
            stop_loss,
            take_profit,
            rr_ratio,
            calc_position_size,
            account_balance,
            risk_pct,
        ),
    )
    conn.commit()
    cursor.close()
    conn.close()
    st.success("Trade executed successfully!")
    send_telegram_notification(
        f"<b>🚀 NEW MANUAL TRADE EXECUTED</b>\n\n"
        f"<b>Pair:</b> <code>{pair}</code>\n<b>Direction:</b> <code>{direction}</code>\n"
        f"<b>Entry:</b> ${entry_price:,.2f}\n<b>SL:</b> ${stop_loss:,.2f}\n<b>TP:</b> ${take_profit:,.2f}\n"
        f"<b>R:R:</b> 1:{rr_ratio}"
    )
    st.rerun()

# ==========================================
# MAIN DASHBOARD HEADER & METRICS BAR
# ==========================================
st.title("📊 Forex & Crypto Trading Terminal")

closed_trades = (
    df_all_trades[df_all_trades['status'] == 'CLOSED']
    if not df_all_trades.empty
    else pd.DataFrame()
)
net_realized_pnl = (
    closed_trades['pnl_usd'].sum() if not closed_trades.empty else 0.0
)
max_dd = 0.0
profit_factor_val = 0.0
if not closed_trades.empty:
    wins_sum = closed_trades[closed_trades['outcome'] == 'WIN']['pnl_usd'].sum()
    loss_sum = abs(
        closed_trades[closed_trades['outcome'] == 'LOSS']['pnl_usd'].sum()
    )
    profit_factor_val = round(wins_sum / loss_sum, 2) if loss_sum > 0 else 0.0

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Net Realized PnL", f"${net_realized_pnl:,.2f}")
col_m2.metric("Max DD", f"{max_dd:.2f}%")
col_m3.metric("Profit Factor", f"{profit_factor_val:.2f}")
col_m4.metric(
    "Total Closed Trades",
    f"{len(closed_trades):,.0f}" if not closed_trades.empty else "0",
)

st.markdown("---")

# ==========================================
# SECTION: REAL-TIME CHART & VOLUME PROFILE
# ==========================================
col_hdr, col_tf = st.columns([3, 1])
with col_hdr:
    st.subheader("📈 Real-Time Chart & Volume Profile (POC, VAH, VAL, Gaps, TEMA)")
with col_tf:
    selected_timeframe = st.selectbox(
        "Select Timeframe:",
        ["5m", "15m", "30m", "1h", "4h", "1d"],
        index=3,
        key="chart_timeframe_select",
    )

col_chart_sel, _ = st.columns([1, 3])
with col_chart_sel:
    chart_symbol = st.selectbox(
        "Select Chart Asset", available_pairs, key="independent_chart_symbol"
    )

def fetch_market_data(symbol: str, interval_str: str):
    try:
        clean_symbol = symbol.upper().strip()
        possible_symbols = []
        if "/" in clean_symbol:
            base, quote = clean_symbol.split("/", 1)
            possible_symbols.append(f"{base}-{quote}")
            if quote == "USDT":
                possible_symbols.append(f"{base}-USD")
        else:
            possible_symbols.append(clean_symbol)
            if not clean_symbol.endswith(("-USD", "=X")):
                possible_symbols.append(f"{clean_symbol}-USD")

        period_map = {
            "5m": ("7d", "5m"),
            "15m": ("14d", "15m"),
            "30m": ("30d", "30m"),
            "1h": ("60d", "1h"),
            "4h": ("60d", "4h"),
            "1d": ("1y", "1d"),
        }
        period, yf_interval = period_map.get(interval_str, ("30d", "30m"))

        df = pd.DataFrame()
        for sym in possible_symbols:
            ticker_obj = yf.Ticker(sym)
            df = ticker_obj.history(period=period, interval=yf_interval)
            if not df.empty:
                break

        if df.empty:
            for sym in possible_symbols:
                df = yf.download(
                    sym, period=period, interval=yf_interval, progress=False
                )
                if not df.empty:
                    break

        if df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.reset_index(inplace=True)
        col_map = {}
        for col in df.columns:
            c_lower = str(col).lower()
            if 'date' in c_lower or 'time' in c_lower:
                col_map[col] = 'time'
            elif 'open' in c_lower:
                col_map[col] = 'open'
            elif 'high' in c_lower:
                col_map[col] = 'high'
            elif 'low' in c_lower:
                col_map[col] = 'low'
            elif 'close' in c_lower:
                col_map[col] = 'close'
            elif 'volume' in c_lower:
                col_map[col] = 'volume'

        df.rename(columns=col_map, inplace=True)
        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return None

        df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
        df = df.dropna(subset=['time', 'open', 'high', 'low', 'close'])
        return df[required_cols]
    except Exception as e:
        st.error(f"Error fetching market chart data for {symbol}: {e}")
        return None

df_ohlc = fetch_market_data(chart_symbol, selected_timeframe)

if df_ohlc is not None and not df_ohlc.empty:
    df_ohlc['tema_custom'] = strategy.calc_tema(
        df_ohlc['close'], period=min(tema_period, len(df_ohlc))
    )
    poc, vah, val = strategy.compute_volume_profile(df_ohlc)
    detected_gaps = strategy.calculate_volume_profile_gaps(
        df_ohlc, num_bins=100, detection_pct=vp_detection_pct
    )

    chart = StreamlitChart(width=1100, height=500)
    chart.layout(background_color='#131722', text_color='#d1d4dc')
    chart.volume_config(
        scale_margin_top=0.75, up_color='#26a69a', down_color='#ef5350'
    )
    chart.set(df_ohlc[['time', 'open', 'high', 'low', 'close', 'volume']])

    tema_df = (
        df_ohlc[['time', 'tema_custom']]
        .dropna()
        .rename(columns={'tema_custom': f'{tema_period} TEMA'})
    )
    if not tema_df.empty:
        tema_line = chart.create_line(
            name=f"{tema_period} TEMA", color="orange", width=2
        )
        tema_line.set(tema_df)

    chart.horizontal_line(vah, color="green", text=f"VAH: {vah:.2f}")
    chart.horizontal_line(val, color="green", text=f"VAL: {val:.2f}")
    chart.horizontal_line(poc, color="red", text=f"POC: {poc:.2f}")

    # Overlay Volume Profile Gap Levels
    if overlay_gaps and detected_gaps:
        for gap_price in detected_gaps:
            chart.horizontal_line(
                gap_price,
                color="purple",
                style="dashed",
                text=f"GAP: {gap_price:.2f}",
            )

    if overlay_chart and not df_all_trades.empty:
        symbol_trades = df_all_trades[
            (df_all_trades['pair'] == chart_symbol)
            & (df_all_trades['status'].isin(['EXECUTED', 'PENDING']))
        ]
        for _, tr in symbol_trades.iterrows():
            chart.horizontal_line(
                float(tr['entry_price']),
                color="blue",
                text=f"ENTRY ({tr['direction']})",
            )
            if pd.notnull(tr['stop_loss']) and tr['stop_loss'] > 0:
                chart.horizontal_line(
                    float(tr['stop_loss']), color="red", text="SL"
                )
            if pd.notnull(tr['take_profit']) and tr['take_profit'] > 0:
                chart.horizontal_line(
                    float(tr['take_profit']), color="green", text="TP"
                )

    chart.load()
else:
    st.warning(f"Could not load market chart for {chart_symbol}.")

st.markdown("---")

# ==========================================
# SECTION: TRADE SETUPS & HISTORY TABLES
# ==========================================
tab_pending, tab_history = st.tabs(
    ["⏳ Pending / Setups", "📜 Cumulative Trades History"]
)

with tab_pending:
    conn = get_db_connection()
    df_active = pd.read_sql(
        "SELECT id, pair, direction, entry_price, stop_loss, take_profit,"
        " risk_reward_ratio AS rrr, risk_pct AS risk_p, position_size, status, trade_state,"
        " created_at FROM trade_setups WHERE status IN ('EXECUTED', 'PENDING');",
        conn,
    )
    conn.close()
    if not df_active.empty:
        st.dataframe(df_active, use_container_width=True)
    else:
        st.info("No active setups found.")

with tab_history:
    conn = get_db_connection()
    df_closed_log = pd.read_sql(
        "SELECT id, pair, direction, entry_price, exit_price, pnl_usd, pnl_pct,"
        " outcome, closed_at FROM trade_setups WHERE status = 'CLOSED';",
        conn,
    )
    conn.close()
    if not df_closed_log.empty:
        st.dataframe(df_closed_log, use_container_width=True)
    else:
        st.info("No closed trade history available.")

st.markdown("---")

# ==========================================
# SECTION: ANALYTICS GRAPHS & SYSTEM LOGS
# ==========================================
col_eq, col_logs = st.columns(2)

with col_eq:
    st.subheader("📉 Cumulative Equity Curve")
    if not closed_trades.empty and 'closed_at' in closed_trades.columns:
        closed_trades['cumulative_pnl'] = closed_trades['pnl_usd'].cumsum()
        st.line_chart(closed_trades.set_index('closed_at')['cumulative_pnl'])
    else:
        st.info("No equity history available yet.")

with col_logs:
    st.subheader("📋 System & Strategy Logs")
    st.markdown("""
    > **Live Engine Status:**
    > * [INFO] Supabase PostgreSQL Database connected.
    > * [INFO] DEAP Genetic Optimizer synced with strategy engine.
    > * [INFO] Worker engine, price monitor, & webhooks online.
    """)

# ==========================================
# SECTION: LIVE AI SENTIMENT & NEWS TESTER
# ==========================================
with st.expander(
    "🧠 Test Live AI Sentiment & News Headline Evaluation", expanded=False
):
    st.markdown(
        "Enter any financial news headline below to instantly evaluate its"
        " sentiment score using the Gemini API integration from `strategy.py`."
    )

    test_headline_input = st.text_area(
        "Market Headline / Text",
        "Major regulatory approval paves way for institutional crypto adoption surge.",
    )

    if st.button("Run Sentiment Test"):
        with st.spinner("Analyzing sentiment via Gemini API..."):
            try:
                sentiment_score = strategy.get_ai_sentiment_score(test_headline_input)

                col_res1, col_res2 = st.columns(2)
                col_res1.metric("Calculated Sentiment Score", f"{sentiment_score:.2f}")

                if sentiment_score >= 0.65:
                    col_res2.success("🟢 Strong Bullish Sentiment")
                elif sentiment_score >= 0.50:
                    col_res2.info("🔵 Mildly Positive / Neutral Sentiment")
                else:
                    col_res2.warning("🔴 Bearish Sentiment")
            except Exception as e:
                st.error(f"Error evaluating sentiment: {e}")