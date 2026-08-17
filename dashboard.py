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
risk_pct = st.sidebar.number_input("Risk Per Trade (%)", min_value=0.01, max_value=100.0, value=1.0)
calc_position_size = round((account_balance * (risk_pct / 100.0)) / risk, 4) if risk > 0 else 0.0

st.title("📊 Forex & Crypto Trading Terminal")

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

        chart.load()
    except Exception as chart_err:
        st.error(f"Lightweight Chart Rendering Exception: {chart_err}")