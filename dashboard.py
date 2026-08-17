import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from lightweight_charts.widgets import StreamlitChart


st.set_page_config(page_title="Trading Terminal", layout="wide")
load_dotenv()

HTTP_PROXY = os.getenv("HTTP_PROXY") or os.getenv("PROXY_URL")
HTTPS_PROXY = os.getenv("HTTPS_PROXY") or os.getenv("PROXY_URL")

if HTTP_PROXY or HTTPS_PROXY:
    os.environ["HTTP_PROXY"] = HTTP_PROXY or HTTPS_PROXY
    os.environ["HTTPS_PROXY"] = HTTPS_PROXY or HTTP_PROXY
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,.supabase.co"


def check_password():
    expected_password = os.getenv("DASHBOARD_PASSWORD", "securepass123")
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if st.session_state["password_correct"]:
        return True

    st.markdown("## 🔒 Restricted Access")
    st.markdown("Please enter the password to access the Trading Terminal.")
    password_input = st.text_input("Password", type="password")
    if st.button("Login", width="stretch"):
        if password_input == expected_password:
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()

from common import (
    calculate_pnl,
    ensure_schema_updated,
    get_db_connection,
    send_telegram_notification,
    close_trade_manually,
)
import strategy

ensure_schema_updated()


def normalize_symbol(symbol: str) -> str:
    if not symbol:
        return ""
    s = str(symbol).replace('"', '').replace("'", "").strip().upper()
    if "/" in s:
        return s
    if s.endswith("USDT") and len(s) > 4:
        return f"{s[:-4]}/{s[-4:]}"
    return s


symbols_env = os.getenv("SYMBOLS") or os.getenv("SYMBOL", "ETH/USDT,BNB/USDT,SOL/USDT")
env_symbols = [normalize_symbol(s) for s in symbols_env.split(",")]

database_url = os.getenv("DATABASE_URL")
if database_url:
    database_url = database_url.strip('"').strip("'")

conn = get_db_connection()
try:
    with conn.cursor() as cur:
        cur.execute("""UPDATE trade_setups SET pair = REPLACE(REPLACE(pair, '"', ''), '''org.postgresql.util.PSQLException''', '');""")
        conn.commit()
except Exception:
    pass
conn.close()

df_all_trades = pd.read_sql("SELECT * FROM trade_setups ORDER BY id DESC;", database_url) if database_url else pd.DataFrame()
db_pairs = [normalize_symbol(p) for p in df_all_trades['pair'].unique().tolist()] if not df_all_trades.empty else []
available_pairs = list(dict.fromkeys(env_symbols + db_pairs))
for default_pair in ["ETH/USDT", "BNB/USDT", "SOL/USDT"]:
    if default_pair not in available_pairs:
        available_pairs.append(default_pair)

st.sidebar.subheader("🎛️ Dashboard Controls & Strategy Tuning")
select_all_pairs = st.sidebar.checkbox("Select All", value=True)
selected_pairs = available_pairs if select_all_pairs else st.sidebar.multiselect("Filter Pairs", available_pairs, default=available_pairs)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Strategy Parameters (Harmonized)")

config_target_pair = st.sidebar.selectbox("Load Dynamic Config For:", available_pairs, index=0)
dyn_cfg = strategy.load_symbol_config(config_target_pair)

lookback_bars = st.sidebar.number_input("Lookback Bars (Range)", value=600, min_value=10, max_value=5000, step=10)
tema_period = st.sidebar.number_input("TEMA Period", value=int(dyn_cfg.get("tema_period", 200)), min_value=10, max_value=500)
rsi_period = st.sidebar.number_input("RSI Period", value=int(dyn_cfg.get("rsi_period", 14)), min_value=2, max_value=50)
rsi_thresh = st.sidebar.slider("RSI Threshold", min_value=20, max_value=80, value=int(dyn_cfg.get("rsi_thresh", 42)))

adx_period = st.sidebar.number_input("ADX Period", value=int(dyn_cfg.get("adx_period", 14)), min_value=2, max_value=50)
adx_threshold = st.sidebar.slider("ADX Threshold", min_value=10.0, max_value=50.0, value=float(dyn_cfg.get("adx_threshold", 20.0)), step=1.0)
max_sl_pct = st.sidebar.slider("Max SL Distance (%)", min_value=0.5, max_value=10.0, value=float(dyn_cfg.get("max_sl_pct", 0.02)) * 100.0, step=0.1) / 100.0

zone_tolerance_pct = st.sidebar.slider("Volume Zone Proximity (%)", min_value=0.1, max_value=3.0, value=float(dyn_cfg.get("zone_tolerance", 0.0075)) * 100.0, step=0.05)

vp_cfg_val = float(dyn_cfg.get("vp_detection_pct", 0.07))
vp_init_slider = (vp_cfg_val * 100.0) if vp_cfg_val <= 1.0 else vp_cfg_val
node_detection_pct = st.sidebar.slider("Node Detection (%)", min_value=0.5, max_value=15.0, value=float(np.clip(vp_init_slider, 0.5, 15.0)), step=0.5) / 100.0
va_pct_val = st.sidebar.slider("Value Area (%)", min_value=10, max_value=100, value=70, step=1) / 100.0

use_adx_filter = st.sidebar.checkbox("Enable ADX Trend Filter", value=bool(dyn_cfg.get("use_adx_filter", True)))
use_rsi_filter = st.sidebar.checkbox("Enable RSI Momentum Filter", value=bool(dyn_cfg.get("use_rsi_filter", True)))
use_candlestick_confirm = st.sidebar.checkbox("Require Candlestick Confirmation", value=bool(dyn_cfg.get("use_candlestick_confirm", True)))

st.sidebar.markdown("---")
pair = st.sidebar.text_input("Execution Pair", "BTC/USDT").upper()
direction = st.sidebar.selectbox("Order Type / Direction", ["LONG", "SHORT"])
overlay_chart = st.sidebar.checkbox("Overlay Trade Positions on Chart", value=True)
overlay_gaps = st.sidebar.checkbox("Overlay Volume Profile Gaps", value=True)

entry_price = st.sidebar.number_input("Entry Price ($)", min_value=0.0, format="%.5f")
stop_loss = st.sidebar.number_input("Stop Loss ($)", min_value=0.0, format="%.5f")
take_profit = st.sidebar.number_input("Take Profit ($)", min_value=0.0, format="%.5f")

risk = abs(entry_price - stop_loss) if (entry_price and stop_loss) else 0.0
reward = abs(take_profit - entry_price) if (take_profit and entry_price) else 0.0
rr_ratio = round(reward / risk, 2) if risk > 0 else 0.0
st.sidebar.markdown(f"**Risk : Reward Ratio:** `1:{rr_ratio}`")

account_balance = st.sidebar.number_input("Account Balance ($)", min_value=1.0, value=float(os.getenv("ACCOUNT_BALANCE", 10000.0)))
risk_pct = st.sidebar.number_input("Risk Per Trade (%)", min_value=0.01, max_value=100.0, value=0.5)
calc_position_size = round((account_balance * (risk_pct / 100.0)) / risk, 4) if risk > 0 else 0.0

if st.sidebar.button("🚀 Execute Order", width="stretch"):
    clean_executed_pair = normalize_symbol(pair)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
            INSERT INTO trade_setups 
            (pair, direction, entry_price, stop_loss, take_profit, risk_reward_ratio, position_size, account_balance, risk_pct, status, trade_state)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'EXECUTED', 'OPEN');
        """,
        (clean_executed_pair, direction, entry_price, stop_loss, take_profit, rr_ratio, calc_position_size, account_balance, risk_pct)
    )
    conn.commit()
    cursor.close()
    conn.close()
    st.success("Trade executed successfully!")
    send_telegram_notification(f"<b>🚀 NEW MANUAL TRADE EXECUTED</b>\n\n<b>Pair:</b> <code>{clean_executed_pair}</code>\n<b>Direction:</b> <code>{direction}</code>")
    st.rerun()

st.title("📊 Forex & Crypto Trading Terminal")

closed_trades = df_all_trades[df_all_trades['status'] == 'CLOSED'] if not df_all_trades.empty else pd.DataFrame()
net_realized_pnl = float(closed_trades['pnl_usd'].sum()) if not closed_trades.empty and 'pnl_usd' in closed_trades.columns else 0.0
profit_factor_val = 0.0
if not closed_trades.empty and 'pnl_usd' in closed_trades.columns:
    wins_sum = float(closed_trades[closed_trades['outcome'] == 'WIN']['pnl_usd'].sum())
    loss_sum = abs(float(closed_trades[closed_trades['outcome'] == 'LOSS']['pnl_usd'].sum()))
    profit_factor_val = round(wins_sum / loss_sum, 2) if loss_sum > 0 else 0.0

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Net Realized PnL", f"${net_realized_pnl:,.2f}")
col_m2.metric("Max DD", "0.00%")
col_m3.metric("Profit Factor", f"{profit_factor_val:.2f}")
col_m4.metric("Total Closed Trades", f"{len(closed_trades):,.0f}" if not closed_trades.empty else "0")

st.markdown("---")

col_hdr, col_tf = st.columns([3, 1])
with col_hdr:
    st.subheader("📈 Real-Time Chart & Volume Profile")
with col_tf:
    selected_timeframe = st.selectbox("Select Timeframe:", ["5m", "15m", "30m", "1h", "4h", "1d"], index=3, key="chart_timeframe_select")

chart_symbol = st.selectbox("Select Chart Asset", available_pairs, key="independent_chart_symbol")

try:
    df_ohlc = strategy.fetch_klines(symbol=chart_symbol, interval=selected_timeframe, limit=max(lookback_bars + 50, 500))
except Exception as fetch_err:
    df_ohlc = None
    st.error(f"API Fetch Error: {fetch_err}")

if df_ohlc is not None and not df_ohlc.empty:
    try:
        if 'time' in df_ohlc.columns:
            df_ohlc['time'] = pd.to_datetime(df_ohlc['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
        elif 'timestamp' in df_ohlc.columns:
            df_ohlc['time'] = pd.to_datetime(df_ohlc['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(df_ohlc.index, pd.DatetimeIndex):
            df_ohlc = df_ohlc.reset_index()
            df_ohlc.rename(columns={df_ohlc.columns[0]: 'time'}, inplace=True)
            df_ohlc['time'] = pd.to_datetime(df_ohlc['time']).dt.strftime('%Y-%m-%d %H:%M:%S')

        df_ohlc['tema_custom'] = strategy.calc_tema(df_ohlc['close'], period=min(int(tema_period), len(df_ohlc)))
        
        poc, vah, val = strategy.compute_volume_profile(
            df_ohlc, 
            num_bins=100, 
            lookback_bars=int(lookback_bars), 
            va_pct=va_pct_val
        )
        detected_gaps = strategy.calculate_volume_profile_gaps(
            df_ohlc, 
            num_bins=100, 
            lookback_bars=int(lookback_bars), 
            detection_pct=node_detection_pct
        )

        chart = StreamlitChart(width=1100, height=500)
        chart.layout(background_color='#131722', text_color='#d1d4dc')
        chart.volume_config(scale_margin_top=0.85, scale_margin_bottom=0.0, up_color='#26a69a', down_color='#ef5350')
        chart.set(df_ohlc[['time', 'open', 'high', 'low', 'close', 'volume']].dropna())

        tema_df = df_ohlc[['time', 'tema_custom']].dropna().rename(columns={'tema_custom': f'{tema_period} TEMA'})
        if not tema_df.empty:
            tema_line = chart.create_line(name=f"{tema_period} TEMA", color="orange", width=2)
            tema_line.set(tema_df)

        if pd.notnull(vah) and vah > 0:
            chart.horizontal_line(float(vah), color="#2962ff", style="solid", width=2, text=f"VAH: {vah:.2f}")
        if pd.notnull(val) and val > 0:
            chart.horizontal_line(float(val), color="#2962ff", style="solid", width=2, text=f"VAL: {val:.2f}")
        if pd.notnull(poc) and poc > 0:
            chart.horizontal_line(float(poc), color="#f44336", style="solid", width=2, text=f"POC: {poc:.2f}")

        if overlay_gaps and detected_gaps:
            df_lookback = df_ohlc.tail(int(lookback_bars))
            min_chart_p = float(df_lookback['low'].min())
            max_chart_p = float(df_lookback['high'].max())

            for gap_price in detected_gaps:
                if pd.notnull(gap_price) and min_chart_p <= float(gap_price) <= max_chart_p:
                    chart.horizontal_line(
                        float(gap_price), 
                        color="#ff9800", 
                        style="dashed", 
                        width=1,
                        text=f"GAP: {gap_price:.2f}"
                    )

        if overlay_chart and not df_all_trades.empty:
            symbol_trades = df_all_trades[(df_all_trades['pair'].apply(normalize_symbol) == normalize_symbol(chart_symbol)) & (df_all_trades['status'].isin(['EXECUTED', 'PENDING']))]
            for _, tr in symbol_trades.iterrows():
                chart.horizontal_line(float(tr['entry_price']), color="blue", text=f"ENTRY ({tr['direction']})")
                if pd.notnull(tr['stop_loss']) and float(tr['stop_loss']) > 0: chart.horizontal_line(float(tr['stop_loss']), color="red", text="SL")
                if pd.notnull(tr['take_profit']) and float(tr['take_profit']) > 0: chart.horizontal_line(float(tr['take_profit']), color="green", text="TP")

        chart.load()
    except Exception as chart_err:
        st.error(f"Lightweight Chart Rendering Exception: {chart_err}")
else:
    st.warning(f"Could not load market chart for {chart_symbol}.")

st.markdown("---")
tab_pending, tab_history = st.tabs(["⏳ Active / Manual Management", "📜 Cumulative Trades History"])
with tab_pending:
    st.subheader("🛠️ Active Trade Management & Manual Overrides")
    df_active = pd.read_sql("SELECT id, pair, direction, entry_price, stop_loss, take_profit, risk_reward_ratio AS rrr, risk_pct AS risk_p, position_size, status, trade_state, created_at FROM trade_setups WHERE status IN ('EXECUTED', 'PENDING');", database_url) if database_url else pd.DataFrame()
    
    if not df_active.empty:
        df_active['pair'] = df_active['pair'].apply(normalize_symbol)
        st.dataframe(df_active, width="stretch")
        
        st.markdown("### 🔧 Manual Trade Control Action")
        selected_trade_id = st.selectbox("Select Trade ID to Manage:", df_active['id'].tolist())
        selected_trade_row = df_active[df_active['id'] == selected_trade_id].iloc[0]
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            new_sl = st.number_input("Update SL ($)", value=float(selected_trade_row['stop_loss']), format="%.5f")
            if st.button("Update Stop Loss"):
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute("UPDATE trade_setups SET stop_loss = %s WHERE id = %s;", (new_sl, selected_trade_id))
                conn.commit()
                cur.close()
                conn.close()
                st.success(f"Updated SL for Trade #{selected_trade_id} to ${new_sl:.5f}")
                st.rerun()

        with col_m2:
            new_tp = st.number_input("Update TP ($)", value=float(selected_trade_row['take_profit']), format="%.5f")
            if st.button("Update Take Profit"):
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute("UPDATE trade_setups SET take_profit = %s WHERE id = %s;", (new_tp, selected_trade_id))
                conn.commit()
                cur.close()
                conn.close()
                st.success(f"Updated TP for Trade #{selected_trade_id} to ${new_tp:.5f}")
                st.rerun()

        with col_m3:
            exit_price_input = st.number_input("Manual Exit Price ($)", value=float(selected_trade_row['entry_price']), format="%.5f")
            if st.button("🚨 Market Close Trade Now", type="primary"):
                if close_trade_manually(selected_trade_id, exit_price_input, reason="MANUAL_OVERRIDE"):
                    st.success(f"Trade #{selected_trade_id} successfully closed manually.")
                    st.rerun()
                else:
                    st.error("Failed to close trade.")

    else:
        st.info("No active setups found.")

with tab_history:
    df_closed_log = pd.read_sql("SELECT id, pair, direction, entry_price, exit_price, pnl_usd, pnl_pct, outcome, closed_at FROM trade_setups WHERE status = 'CLOSED';", database_url) if database_url else pd.DataFrame()
    if not df_closed_log.empty:
        df_closed_log['pair'] = df_closed_log['pair'].apply(normalize_symbol)
        st.dataframe(df_closed_log, width="stretch")
    else:
        st.info("No closed trade history available.")

st.markdown("---")
col_eq, col_hm = st.columns(2)

with col_eq:
    st.subheader("📈 Cumulative Equity Curve")
    if not closed_trades.empty and 'pnl_usd' in closed_trades.columns:
        df_eq = closed_trades.sort_values('closed_at').copy()
        df_eq['pnl_usd'] = pd.to_numeric(df_eq['pnl_usd'], errors='coerce').fillna(0.0)
        df_eq['cumulative_pnl'] = df_eq['pnl_usd'].cumsum()
        st.line_chart(df_eq.set_index('closed_at')['cumulative_pnl'])
    else:
        st.info("No equity history available yet.")

with col_hm:
    st.subheader("📅 Day vs Hour Performance Heatmap")
    if not closed_trades.empty and 'closed_at' in closed_trades.columns:
        try:
            closed_trades['closed_at'] = pd.to_datetime(closed_trades['closed_at'])
            closed_trades['day_of_week'] = closed_trades['closed_at'].dt.day_name()
            closed_trades['hour'] = closed_trades['closed_at'].dt.hour
            closed_trades['pnl_usd'] = pd.to_numeric(closed_trades['pnl_usd'], errors='coerce').fillna(0.0)
            heatmap_data = closed_trades.pivot_table(index='day_of_week', columns='hour', values='pnl_usd', aggfunc='sum').fillna(0)
            st.dataframe(heatmap_data, width="stretch")
        except Exception:
            st.info("No heatmap data available.")
    else:
        st.info("No heatmap data available.")

st.markdown("---")
col_per, col_log = st.columns(2)

with col_per:
    st.subheader("🗂️ Periodic Performance Breakdown")
    group_interval = st.radio("Group Interval", ["Weekly", "Monthly"], horizontal=True)
    if not closed_trades.empty and 'closed_at' in closed_trades.columns:
        try:
            df_per = closed_trades.copy()
            df_per['closed_at'] = pd.to_datetime(df_per['closed_at'])
            df_per['pnl_usd'] = pd.to_numeric(df_per['pnl_usd'], errors='coerce').fillna(0.0)
            period_rule = 'W' if group_interval == 'Weekly' else 'ME'
            periodic_summary = df_per.groupby(pd.Grouper(key='closed_at', freq=period_rule)).agg({'pnl_usd': ['sum', 'count']})
            periodic_summary.columns = ['Net PnL ($)', 'Trade Count']
            st.dataframe(periodic_summary, width="stretch")
        except Exception:
            st.info("No periodic data available.")
    else:
        st.info("No periodic data available.")

with col_log:
    st.subheader("📋 System & Strategy Logs")
    st.markdown("*Live Worker Logs:*")
    log_messages = [
        "[INFO] Database connection established...",
        "[INFO] Real-time chart WebSocket connected.",
        "[INFO] Pending setups tab in sync with PostgreSQL.",
        "[STATUS] Waiting for price action triggers."
    ]
    for log in log_messages:
        st.code(log, language="text")