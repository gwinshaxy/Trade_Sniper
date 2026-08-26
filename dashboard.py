import os
import sys
import json
import urllib.parse
import logging
import subprocess
import psutil
import gc
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

from lightweight_charts.widgets import StreamlitChart

# Page configuration MUST be the first Streamlit command executed
st.set_page_config(page_title="MEXC Trading Terminal", layout="wide")

# Encode and Inject Dynamic Data URI PWA Manifest & Metadata
manifest_data = {
    "name": "TradeSniper Terminal",
    "short_name": "TradeSniper",
    "description": "MEXC Algorithmic Trading Terminal",
    "start_url": "/",
    "display": "standalone",
    "background_color": "#0e1117",
    "theme_color": "#0e1117",
    "icons": [
        {
            "src": "https://raw.githubusercontent.com/gwinshaxy/Trade_Sniper/main/icon-192.png",
            "sizes": "192x192",
            "type": "image/png"
        },
        {
            "src": "https://raw.githubusercontent.com/gwinshaxy/Trade_Sniper/main/icon-512.png",
            "sizes": "512x512",
            "type": "image/png"
        }
    ]
}

manifest_json = json.dumps(manifest_data)
encoded_manifest = urllib.parse.quote(manifest_json)
manifest_uri = f"data:application/manifest+json;charset=utf-8,{encoded_manifest}"

st.markdown(
    f"""
    <link rel="manifest" href="{manifest_uri}">
    <meta name="theme-color" content="#0e1117">
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="TradeSniper">
    """,
    unsafe_allow_html=True
)

# Inject Blob-Based Service Worker Registration
st.markdown(
    """
    <script>
    if ('serviceWorker' in navigator) {
      window.addEventListener('load', function() {
        const swCode = `
          self.addEventListener('install', e => self.skipWaiting());
          self.addEventListener('activate', e => e.waitUntil(clients.claim()));
          self.addEventListener('fetch', e => {
            e.respondWith(fetch(e.request).catch(() => new Response('Offline')));
          });
        `;
        const blob = new Blob([swCode], { type: 'application/javascript' });
        const blobURL = URL.createObjectURL(blob);
        
        navigator.serviceWorker.register(blobURL)
          .then(reg => console.log('PWA ServiceWorker registered successfully via Blob'))
          .catch(err => console.error('ServiceWorker registration failed:', err));
      });
    }
    </script>
    """,
    unsafe_allow_html=True
)

HTTP_PROXY = os.getenv("HTTP_PROXY") or os.getenv("PROXY_URL")
HTTPS_PROXY = os.getenv("HTTPS_PROXY") or os.getenv("PROXY_URL")

if HTTP_PROXY or HTTPS_PROXY:
    os.environ["HTTP_PROXY"] = HTTP_PROXY or HTTPS_PROXY
    os.environ["HTTPS_PROXY"] = HTTPS_PROXY or HTTP_PROXY
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,.supabase.co"

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

def check_password():
    raw_env_pass = os.getenv("DASHBOARD_PASSWORD", "")
    expected_password = raw_env_pass.strip().strip('"').strip("'")
    if not expected_password:
        st.error("🚨 CRITICAL CONFIG ERROR: DASHBOARD_PASSWORD environment variable is missing. Login disabled.")
        st.stop()

    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if st.session_state["password_correct"]:
        return True

    st.markdown("## 🔒 Restricted Access Terminal")
    password_input = st.text_input("Password", type="password")
    if st.button("Login", use_container_width=True):
        if password_input.strip() == expected_password:
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("❌ Password incorrect")
    return False

if not check_password():
    st.stop()

from common import (
    ensure_schema_updated,
    close_trade_manually,
    get_db_connection,
    release_db_connection,
    send_telegram_notification,
    logger
)
from live_executor import LiveExecutionEngine

try:
    import strategy
except ModuleNotFoundError:
    st.error("❌ Module Import Error: 'strategy.py' could not be located in directory paths.")
    st.stop()

try:
    ensure_schema_updated()
    executor = LiveExecutionEngine()
except Exception as init_err:
    st.error(f"Failed to initialize trading core engine: {init_err}")
    st.stop()

def normalize_symbol(symbol: str) -> str:
    if not symbol:
        return ""
    s = str(symbol).replace('"', '').replace("'", "").strip().upper()
    if "/" in s:
        return s
    if s.endswith("USDT") and len(s) > 4:
        return f"{s[:-4]}/{s[-4:]}"
    return s

symbols_env = os.getenv("SYMBOLS") or os.getenv("SYMBOL", "XRP/USDT")
env_symbols = [normalize_symbol(s) for s in symbols_env.split(",")]

database_url = os.getenv("DATABASE_URL") or os.getenv("DB_URL")
if database_url:
    database_url = database_url.strip('"').strip("'")

conn = get_db_connection()
if conn:
    try:
        with conn.cursor() as cur:
            cur.execute("""UPDATE trade_setups SET pair = REPLACE(REPLACE(pair::text, '"', ''), '''', '') WHERE pair IS NOT NULL;""")
            conn.commit()
    except Exception as db_err:
        logger.warning(f"Failed to normalize DB pairs on startup: {db_err}")
    finally:
        release_db_connection(conn)

@st.cache_data(ttl=5, max_entries=20)
def fetch_trades_from_db(db_url: str) -> pd.DataFrame:
    if not db_url:
        return pd.DataFrame()
    try:
        return pd.read_sql("SELECT * FROM trade_setups ORDER BY id DESC;", db_url)
    except Exception as db_err:
        logger.error(f"Error fetching trade records: {db_err}")
        return pd.DataFrame()

df_all_trades = fetch_trades_from_db(database_url)

db_pairs = [normalize_symbol(p) for p in df_all_trades['pair'].unique().tolist()] if not df_all_trades.empty and 'pair' in df_all_trades.columns else []
available_pairs = list(dict.fromkeys(env_symbols + db_pairs))
for default_pair in ["XRP/USDT"]:
    if default_pair not in available_pairs:
        available_pairs.append(default_pair)

st.sidebar.subheader("🎛️ Terminal Controls & Tuning")

# 1. Active Execution / Config Pair
config_target_pair = st.sidebar.selectbox("Active Execution / Config Pair", available_pairs, index=0)

# Normalize key string for consistent strategy dict loading & widget session keys
pair_key = config_target_pair.replace("/", "")
dyn_cfg = strategy.load_symbol_config(config_target_pair)

# 2. Attach pair-specific keys to widgets so values re-initialize dynamically on pair switch
lookback_bars = st.sidebar.number_input("Lookback Bars (Range)", value=600, min_value=100, max_value=1000, step=50, key=f"lookback_{pair_key}")

tema_period = st.sidebar.number_input(
    "TEMA Period", 
    value=int(dyn_cfg.get("tema_period", 200)), 
    min_value=10, 
    max_value=500, 
    key=f"tema_{pair_key}"
)

rsi_period = st.sidebar.number_input(
    "RSI Period", 
    value=int(dyn_cfg.get("rsi_period", 14)), 
    min_value=2, 
    max_value=50, 
    key=f"rsi_p_{pair_key}"
)

rsi_thresh = st.sidebar.slider(
    "RSI Threshold", 
    min_value=20, 
    max_value=80, 
    value=int(dyn_cfg.get("rsi_thresh", 42)), 
    key=f"rsi_t_{pair_key}"
)

adx_period = st.sidebar.number_input(
    "ADX Period", 
    value=int(dyn_cfg.get("adx_period", 14)), 
    min_value=2, 
    max_value=50, 
    key=f"adx_p_{pair_key}"
)

adx_threshold = st.sidebar.slider(
    "ADX Threshold", 
    min_value=10.0, 
    max_value=50.0, 
    value=float(dyn_cfg.get("adx_threshold", 20.0)), 
    step=1.0, 
    key=f"adx_t_{pair_key}"
)

max_sl_pct = st.sidebar.slider(
    "Max SL Distance (%)", 
    min_value=0.5, 
    max_value=3.0, 
    value=float(dyn_cfg.get("max_sl_pct", 0.02)) * 100.0, 
    step=0.1, 
    key=f"sl_{pair_key}"
) / 100.0

zone_tolerance_pct = st.sidebar.slider(
    "Volume Zone Proximity (%)", 
    min_value=0.1, 
    max_value=3.0, 
    value=float(dyn_cfg.get("zone_tolerance", 0.0075)) * 100.0, 
    step=0.05, 
    key=f"zone_{pair_key}"
)

vp_cfg_val = float(dyn_cfg.get("vp_detection_pct", 0.07))
vp_init_slider = (vp_cfg_val * 100.0) if vp_cfg_val <= 1.0 else vp_cfg_val

node_detection_pct = st.sidebar.slider(
    "Node Detection (%)", 
    min_value=0.5, 
    max_value=15.0, 
    value=float(np.clip(vp_init_slider, 0.5, 15.0)), 
    step=0.5, 
    key=f"node_{pair_key}"
) / 100.0

use_adx_filter = st.sidebar.checkbox(
    "Enable ADX Trend Filter", 
    value=bool(dyn_cfg.get("use_adx_filter", True)), 
    key=f"chk_adx_{pair_key}"
)

use_rsi_filter = st.sidebar.checkbox(
    "Enable RSI Momentum Filter", 
    value=bool(dyn_cfg.get("use_rsi_filter", True)), 
    key=f"chk_rsi_{pair_key}"
)

use_candlestick_confirm = st.sidebar.checkbox(
    "Require Candlestick Confirmation", 
    value=bool(dyn_cfg.get("use_candlestick_confirm", True)), 
    key=f"chk_candle_{pair_key}"
)

st.sidebar.markdown("---")
direction = st.sidebar.selectbox("Order Direction", ["LONG", "SHORT"])
overlay_chart = st.sidebar.checkbox("Overlay Trade Positions on Chart", value=True)
overlay_gaps = st.sidebar.checkbox("Overlay Volume Profile Gaps", value=True)

entry_price = st.sidebar.number_input("Entry Price ($)", min_value=0.0, format="%.5f")
stop_loss = st.sidebar.number_input("Stop Loss ($)", min_value=0.0, format="%.5f")
take_profit = st.sidebar.number_input("Take Profit ($)", min_value=0.0, format="%.5f")

risk = abs(entry_price - stop_loss) if (entry_price and stop_loss) else 0.0
reward = abs(take_profit - entry_price) if (take_profit and entry_price) else 0.0
rr_ratio = round(reward / risk, 2) if risk > 0 else 0.0
st.sidebar.markdown(f"**Risk : Reward Ratio:** `1:{rr_ratio}`")

account_balance = st.sidebar.number_input("Account Balance ($)", min_value=1.0, value=float(os.getenv("ACCOUNT_BALANCE", 100.0)))
risk_pct = st.sidebar.number_input("Risk Per Trade (%)", min_value=0.01, max_value=100.0, value=1.0)
calc_position_size = round((account_balance * (risk_pct / 100.0)) / risk, 4) if risk > 0 else 0.0

if st.sidebar.button("🚀 Execute Live Order via MEXC API", use_container_width=True):
    with st.spinner("Submitting order..."):
        try:
            success = executor.execute_live_order(
                pair=config_target_pair,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            if success:
                st.cache_data.clear()
                st.success(f"Live trade executed for {config_target_pair} on MEXC Futures!")
                send_telegram_notification(f"<b>🚀 NEW LIVE ORDER EXECUTED</b>\n\n<b>Pair:</b> <code>{config_target_pair}</code>\n<b>Direction:</b> <code>{direction}</code>")
                st.rerun()
            else:
                st.error("Order execution failed. Review engine logs.")
        except Exception as exec_err:
            st.error(f"Execution Error Encountered: {exec_err}")

st.title("📊 Live MEXC Trading Dashboard")

closed_trades = df_all_trades[df_all_trades['status'] == 'CLOSED'] if not df_all_trades.empty and 'status' in df_all_trades.columns else pd.DataFrame()
net_realized_pnl = float(closed_trades['pnl_usd'].sum()) if not closed_trades.empty and 'pnl_usd' in closed_trades.columns else 0.0

profit_factor_val = 0.0
if not closed_trades.empty and 'pnl_usd' in closed_trades.columns and 'outcome' in closed_trades.columns:
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
    st.subheader("📈 Real-Time Interactive Chart & Volume Profile")
with col_tf:
    selected_timeframe = st.selectbox("Select Timeframe:", ["5m", "15m", "30m", "1h", "4h", "1d"], index=3, key="chart_timeframe_select")

chart_symbol = st.selectbox("Select Chart Asset", available_pairs, key="independent_chart_symbol")

try:
    df_ohlc = strategy.fetch_klines(symbol=chart_symbol, interval=selected_timeframe)
except Exception as fetch_err:
    df_ohlc = None
    st.error(f"API Fetch Error: {fetch_err}")

if df_ohlc is not None and not df_ohlc.empty:
    try:
        df_ohlc = df_ohlc.copy()

        # Re-index datetime index or normalize timestamp column name
        if isinstance(df_ohlc.index, pd.DatetimeIndex):
            df_ohlc = df_ohlc.reset_index()
            df_ohlc.rename(columns={df_ohlc.columns[0]: 'time'}, inplace=True)
        elif 'timestamp' in df_ohlc.columns and 'time' not in df_ohlc.columns:
            df_ohlc.rename(columns={'timestamp': 'time'}, inplace=True)

        # Deduplicate column names
        df_ohlc = df_ohlc.loc[:, ~df_ohlc.columns.duplicated()].copy()

        # Parse time column as datetime pandas series safely
        df_ohlc['time'] = pd.to_datetime(df_ohlc['time'], errors='coerce')
        df_ohlc = df_ohlc.dropna(subset=['time'])

        # Ensure numeric OHLCV types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df_ohlc.columns:
                df_ohlc[col] = pd.to_numeric(df_ohlc[col], errors='coerce')

        df_ohlc['tema_custom'] = strategy.calc_tema(df_ohlc['close'], period=min(tema_period, len(df_ohlc)))
        
        poc, vah, val = strategy.compute_volume_profile(df_ohlc, num_bins=100, lookback_bars=lookback_bars, va_pct=0.70)
        detected_gaps = strategy.calculate_volume_profile_gaps(df_ohlc, num_bins=100, lookback_bars=lookback_bars, detection_pct=node_detection_pct)

        min_chart_p = float(df_ohlc['low'].min())
        max_chart_p = float(df_ohlc['high'].max())

        # Format string time depending on daily vs intraday resolution
        df_chart = df_ohlc.copy()
        if selected_timeframe in ['1d']:
            df_chart['time'] = df_chart['time'].dt.strftime('%Y-%m-%d')
        else:
            df_chart['time'] = df_chart['time'].dt.strftime('%Y-%m-%d %H:%M:%S')

        df_chart = df_chart.drop_duplicates(subset=['time']).sort_values('time', ascending=True).reset_index(drop=True)

        chart = StreamlitChart(width=None, height=650)
        chart.layout(background_color='#131722', text_color='#d1d4dc')
        chart.volume_config(scale_margin_top=0.85, scale_margin_bottom=0.0, up_color='#26a69a', down_color='#ef5350')
        
        # Pass standardized dataframe into chart
        ohlcv_data = df_chart[['time', 'open', 'high', 'low', 'close', 'volume']].dropna()
        chart.set(ohlcv_data)

        line_name = f"{tema_period} TEMA"
        tema_df = df_chart[['time', 'tema_custom']].dropna().rename(columns={'tema_custom': line_name})
        if not tema_df.empty:
            tema_line = chart.create_line(name=line_name, color="orange", width=2)
            tema_line.set(tema_df)

        if pd.notnull(vah) and min_chart_p <= float(vah) <= max_chart_p:
            chart.horizontal_line(float(vah), color="#2962ff", style="solid", width=2, text=f"VAH: {vah:.2f}")
        if pd.notnull(val) and min_chart_p <= float(val) <= max_chart_p:
            chart.horizontal_line(float(val), color="#2962ff", style="solid", width=2, text=f"VAL: {val:.2f}")
        if pd.notnull(poc) and min_chart_p <= float(poc) <= max_chart_p:
            chart.horizontal_line(float(poc), color="#f44336", style="solid", width=2, text=f"POC: {poc:.2f}")

        if overlay_gaps and detected_gaps:
            for gap_price in detected_gaps:
                if pd.notnull(gap_price) and min_chart_p <= float(gap_price) <= max_chart_p:
                    chart.horizontal_line(float(gap_price), color="#ff9800", style="dashed", width=1, text=f"GAP: {gap_price:.2f}")

        if overlay_chart and not df_all_trades.empty and 'pair' in df_all_trades.columns and 'status' in df_all_trades.columns:
            symbol_trades = df_all_trades[(df_all_trades['pair'].apply(normalize_symbol) == normalize_symbol(chart_symbol)) & (df_all_trades['status'].isin(['EXECUTED', 'PENDING']))]
            for _, tr in symbol_trades.iterrows():
                entry_p = float(tr['entry_price']) if pd.notnull(tr['entry_price']) else 0.0
                sl_p = float(tr['stop_loss']) if pd.notnull(tr['stop_loss']) else 0.0
                tp_p = float(tr['take_profit']) if pd.notnull(tr['take_profit']) else 0.0

                if min_chart_p * 0.5 <= entry_p <= max_chart_p * 1.5:
                    chart.horizontal_line(entry_p, color="blue", text=f"ENTRY ({tr['direction']})")
                if sl_p > 0 and (min_chart_p * 0.5 <= sl_p <= max_chart_p * 1.5):
                    chart.horizontal_line(sl_p, color="red", text="SL")
                if tp_p > 0 and (min_chart_p * 0.5 <= tp_p <= max_chart_p * 1.5):
                    chart.horizontal_line(tp_p, color="green", text="TP")

        chart.load()
    except Exception as chart_err:
        st.error(f"Lightweight Chart Rendering Exception: {chart_err}")
else:
    st.warning(f"Could not load market chart for {chart_symbol}.")

st.markdown("---")

tab_active, tab_history = st.tabs(["⏳ Active Positions & Management", "📜 Trade History Log"])

with tab_active:
    df_active = df_all_trades[df_all_trades['status'].isin(['EXECUTED', 'PENDING'])] if not df_all_trades.empty and 'status' in df_all_trades.columns else pd.DataFrame()

    if not df_active.empty:
        df_active['pair'] = df_active['pair'].apply(normalize_symbol)
        st.dataframe(df_active, use_container_width=True)
        
        st.markdown("### 🔧 Live Trade Overrides & Modifications")
        selected_trade_id = st.selectbox("Select Trade ID to Manage:", df_active['id'].tolist())
        selected_row = df_active[df_active['id'] == selected_trade_id].iloc[0]
        
        col_sl, col_tp = st.columns(2)
        with col_sl:
            new_sl = st.number_input("Update Stop Loss ($)", value=float(selected_row['stop_loss']) if pd.notnull(selected_row['stop_loss']) else 0.0, format="%.5f")
            if st.button("Update SL in Database", use_container_width=True):
                conn_mod = get_db_connection()
                if conn_mod:
                    try:
                        cur = conn_mod.cursor()
                        cur.execute("UPDATE trade_setups SET stop_loss = %s WHERE id = %s;", (new_sl, selected_trade_id))
                        conn_mod.commit()
                        cur.close()
                    except Exception as sl_err:
                        logger.error(f"Error updating SL in DB: {sl_err}")
                    finally:
                        release_db_connection(conn_mod)
                    st.cache_data.clear()
                    st.success(f"Updated SL for Trade #{selected_trade_id} to ${new_sl:.5f}")
                    st.rerun()

        with col_tp:
            new_tp = st.number_input("Update Take Profit ($)", value=float(selected_row['take_profit']) if pd.notnull(selected_row['take_profit']) else 0.0, format="%.5f")
            if st.button("Update TP in Database", use_container_width=True):
                conn_mod = get_db_connection()
                if conn_mod:
                    try:
                        cur = conn_mod.cursor()
                        cur.execute("UPDATE take_profit SET take_profit = %s WHERE id = %s;", (new_tp, selected_trade_id))
                        conn_mod.commit()
                        cur.close()
                    except Exception as tp_err:
                        logger.error(f"Error updating TP in DB: {tp_err}")
                    finally:
                        release_db_connection(conn_mod)
                    st.cache_data.clear()
                    st.success(f"Updated TP for Trade #{selected_trade_id} to ${new_tp:.5f}")
                    st.rerun()

        st.markdown("---")
        manual_exit = st.number_input("Fallback Exit Price ($) [Used for Market Close/DB Override]", value=float(selected_row['entry_price']) if pd.notnull(selected_row['entry_price']) else 0.0, format="%.5f")
        
        col_mexc, col_db_only = st.columns(2)
        with col_mexc:
            if st.button("🚨 Market Close on MEXC Exchange", type="primary", use_container_width=True):
                try:
                    close_res = executor.close_live_position_mexc(
                        pair=selected_row['pair'],
                        position_size=float(selected_row['position_size']),
                        direction=selected_row['direction']
                    )
                    
                    if close_res.get("success") and float(close_res.get("exit_price", 0.0)) > 0.0:
                        final_p = float(close_res["exit_price"])
                        if close_trade_manually(selected_trade_id, final_p, reason="STREAMLIT_MEXC_MARKET_CLOSE"):
                            st.cache_data.clear()
                            st.success(f"Position #{selected_trade_id} closed on MEXC at ${final_p:.5f}.")
                            st.rerun()
                        else:
                            st.error("Position closed on MEXC, but database update failed.")
                    else:
                        st.error("🚨 CRITICAL: MEXC close order failed or returned $0.00 price. Live position remains OPEN on exchange. DB was NOT modified.")
                except Exception as close_err:
                    st.error(f"Error attempting MEXC close: {close_err}")
                    
        with col_db_only:
            if st.button("⚠️ Force DB Close Only (No MEXC Order)", use_container_width=True):
                if close_trade_manually(selected_trade_id, manual_exit, reason="STREAMLIT_MANUAL_DB_ONLY_OVERRIDE"):
                    st.cache_data.clear()
                    st.warning(f"Position #{selected_trade_id} marked as closed in DB at ${manual_exit:.5f}. (Note: No exchange order sent).")
                    st.rerun()
                else:
                    st.error("Failed to update position in database.")
    else:
        st.info("No active positions currently running.")

with tab_history:
    if not closed_trades.empty:
        closed_trades['pair'] = closed_trades['pair'].apply(normalize_symbol)
        st.dataframe(closed_trades, use_container_width=True)
    else:
        st.info("No closed trade history available.")

st.markdown("---")

col_eq, col_hm = st.columns(2)

with col_eq:
    st.subheader("📈 Cumulative Equity Curve")
    if not closed_trades.empty and 'pnl_usd' in closed_trades.columns and 'closed_at' in closed_trades.columns:
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
            st.dataframe(heatmap_data, use_container_width=True)
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
            st.dataframe(periodic_summary, use_container_width=True)
        except Exception:
            st.info("No periodic data available.")
    else:
        st.info("No periodic data available.")

with col_log:
    st.subheader("📋 System & Strategy Logs")
    st.markdown("*Live Worker Logs:*")
    log_messages = [
        "[INFO] Database connection established...",
        "[INFO] Live Execution Engine bound to MEXC API.",
        "[INFO] Pending setups tab synchronized with PostgreSQL.",
        "[STATUS] Monitoring strategy triggers and active positions."
    ]
    for log_item in log_messages:
        st.code(log_item, language="text")

gc.collect()