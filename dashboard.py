import os
import gc
import logging
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)
from lightweight_charts.widgets import StreamlitChart

st.set_page_config(page_title="Trading Terminal", layout="wide")

def check_password():
    raw_env_pass = os.getenv("DASHBOARD_PASSWORD", "securepass123")
    expected_password = raw_env_pass.strip().strip('"').strip("'")
    
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if st.session_state["password_correct"]:
        return True

    st.markdown("## 🔒 Restricted Access")
    password_input = st.text_input("Password", type="password")
    if st.button("Login"):
        if password_input.strip() == expected_password:
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("Incorrect password")
    return False

if not check_password():
    st.stop()

from common import calculate_pnl, ensure_schema_updated, get_db_connection, send_telegram_notification, close_trade_manually
import strategy

ensure_schema_updated()

def normalize_symbol(symbol: str) -> str:
    if not symbol: return ""
    s = str(symbol).replace('"', '').replace("'", "").strip().upper()
    if "/" in s: return s
    if s.endswith("USDT") and len(s) > 4: return f"{s[:-4]}/{s[-4:]}"
    return s

symbols_env = os.getenv("SYMBOLS") or os.getenv("SYMBOL", "ETH/USDT,BNB/USDT,SOL/USDT")
env_symbols = [normalize_symbol(s) for s in symbols_env.split(",")]
database_url = os.getenv("DATABASE_URL")
if database_url: database_url = database_url.strip('"').strip("'")

# --- CACHED MARKET & DB LOADERS ---
@st.cache_data(ttl=30, show_spinner=False)
def fetch_cached_klines(symbol, timeframe, limit):
    return strategy.fetch_klines(symbol=symbol, interval=timeframe, limit=limit)

@st.cache_data(ttl=15, max_entries=10)
def load_all_trades(db_url):
    if not db_url: return pd.DataFrame()
    try: return pd.read_sql("SELECT * FROM trade_setups ORDER BY id DESC;", db_url)
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=15, max_entries=10)
def load_active_trades(db_url):
    if not db_url: return pd.DataFrame()
    try: return pd.read_sql("SELECT id, pair, direction, entry_price, stop_loss, take_profit, risk_reward_ratio AS rrr, risk_pct AS risk_p, position_size, status, trade_state, created_at FROM trade_setups WHERE status IN ('EXECUTED', 'PENDING');", db_url)
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=15, max_entries=10)
def load_closed_trades_log(db_url):
    if not db_url: return pd.DataFrame()
    try: return pd.read_sql("SELECT id, pair, direction, entry_price, exit_price, pnl_usd, pnl_pct, outcome, closed_at FROM trade_setups WHERE status = 'CLOSED';", db_url)
    except Exception: return pd.DataFrame()

df_all_trades = load_all_trades(database_url)
db_pairs = [normalize_symbol(p) for p in df_all_trades['pair'].unique().tolist()] if not df_all_trades.empty and 'pair' in df_all_trades.columns else []
available_pairs = list(dict.fromkeys(env_symbols + db_pairs))

st.sidebar.subheader("🎛️ Dashboard Controls & Strategy Tuning")
config_target_pair = st.sidebar.selectbox("Load Config For:", available_pairs, index=0)
dyn_cfg = strategy.load_symbol_config(config_target_pair)

lookback_bars = st.sidebar.number_input("Lookback Bars", value=300, min_value=100, max_value=1000)
tema_period = st.sidebar.number_input("TEMA Period", value=int(dyn_cfg.get("tema_period", 200)))
node_detection_pct = st.sidebar.slider("Node Detection (%)", 0.5, 15.0, 7.0) / 100.0

overlay_chart = st.sidebar.checkbox("Overlay Positions", value=True)
overlay_gaps = st.sidebar.checkbox("Overlay Volume Gaps", value=True)

st.title("📊 Forex & Crypto Trading Terminal")

closed_trades = df_all_trades[df_all_trades['status'] == 'CLOSED'] if not df_all_trades.empty and 'status' in df_all_trades.columns else pd.DataFrame()
net_realized_pnl = float(closed_trades['pnl_usd'].sum()) if not closed_trades.empty and 'pnl_usd' in closed_trades.columns else 0.0

col1, col2, col3 = st.columns(3)
col1.metric("Net Realized PnL", f"${net_realized_pnl:,.2f}")
col2.metric("Total Trades", len(closed_trades))
col3.metric("System Status", "ONLINE")

st.markdown("---")
col_hdr, col_tf = st.columns([3, 1])
with col_hdr: st.subheader("📈 Real-Time Chart & Volume Profile")
with col_tf: selected_timeframe = st.selectbox("Timeframe:", ["5m", "15m", "30m", "1h", "4h", "1d"], index=3)

chart_symbol = st.selectbox("Asset", available_pairs)
df_ohlc = fetch_cached_klines(chart_symbol, selected_timeframe, lookback_bars)

if df_ohlc is not None and not df_ohlc.empty:
    df_ohlc['tema_custom'] = strategy.calc_tema(df_ohlc['close'], period=min(tema_period, len(df_ohlc)))
    poc, vah, val = strategy.compute_volume_profile(df_ohlc, lookback_bars=lookback_bars)
    detected_gaps = strategy.calculate_volume_profile_gaps(df_ohlc, lookback_bars=lookback_bars, detection_pct=node_detection_pct)

    chart = StreamlitChart(width=1100, height=450)
    chart.layout(background_color='#131722', text_color='#d1d4dc')
    chart.set(df_ohlc[['time', 'open', 'high', 'low', 'close', 'volume']])

    if pd.notnull(poc): chart.horizontal_line(float(poc), color="#f44336", text=f"POC: {poc:.2f}")
    if pd.notnull(vah): chart.horizontal_line(float(vah), color="#2962ff", text=f"VAH: {vah:.2f}")
    if pd.notnull(val): chart.horizontal_line(float(val), color="#2962ff", text=f"VAL: {val:.2f}")

    chart.load()
else:
    st.warning(f"Could not load market data for {chart_symbol}.")

st.markdown("---")
tab_active, tab_hist = st.tabs(["⏳ Active Positions", "📜 History"])
with tab_active:
    st.dataframe(load_active_trades(database_url), use_container_width=True)
with tab_hist:
    st.dataframe(load_closed_trades_log(database_url), use_container_width=True)

gc.collect()