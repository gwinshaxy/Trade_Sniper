import os
import json
import ccxt
import pandas as pd
import numpy as np
import pandas_ta as ta
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

# Page Setup
st.set_page_config(
    page_title="Trade Sniper Dashboard",
    page_icon="🎯",
    layout="wide"
)

load_dotenv()

# =====================================================================
# SUPABASE DATABASE CONNECTION
# =====================================================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

@st.cache_resource
def get_supabase_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.error("⚠️ Supabase credentials missing! Check environment variables.")
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.error(f"Failed to connect to Supabase: {e}")
        return None

supabase = get_supabase_client()

def load_trade_journal() -> pd.DataFrame:
    if not supabase:
        return pd.DataFrame()
    try:
        response = supabase.table("trade_journal").select("*").order("id", desc=True).execute()
        data = response.data
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        numeric_cols = [
            'entry_price', 'stop_loss', 'take_profit_1', 'take_profit_2', 
            'position_usdt', 'max_risk_usd', 'exit_price', 
            'realized_pnl_usd', 'realized_r'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error loading trade journal: {e}")
        return pd.DataFrame()

def clear_remote_journal():
    if supabase:
        try:
            supabase.table("trade_journal").delete().gt("id", 0).execute()
            st.toast("🧹 Trade journal database cleared!", icon="✅")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to clear database: {e}")

# =====================================================================
# MARKET DATA (CCXT + PANDAS_TA)
# =====================================================================
@st.cache_data(ttl=60)
def fetch_market_data(symbol: str, timeframe: str = "1h", limit: int = 300) -> pd.DataFrame:
    """Fetches OHLCV market data from Bybit via CCXT and calculates TEMA & ADX."""
    try:
        exchange = ccxt.bybit()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # TradingView Lightweight charts requires UNIX timestamp in seconds
        df['time'] = (df['timestamp'] / 1000).astype(int)
        
        # Calculate Indicators
        df['tema_200'] = ta.tema(df['close'], length=200)
        
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx_df is not None and not adx_df.empty:
            df['adx'] = adx_df['ADX_14']
        
        return df
    except Exception as e:
        st.error(f"Error fetching chart data for {symbol}: {e}")
        return pd.DataFrame()

# =====================================================================
# TRADINGVIEW LIGHTWEIGHT CHARTS HTML RENDERER
# =====================================================================
def render_tradingview_chart(df: pd.DataFrame, symbol: str, active_trades: pd.DataFrame):
    """Generates an embedded HTML5 canvas powered by TradingView Lightweight Charts v4."""
    
    # 1. Prepare Candlestick JSON Data
    candles_data = df[['time', 'open', 'high', 'low', 'close']].to_dict(orient='records')
    
    # 2. Prepare TEMA JSON Data
    df_tema = df.dropna(subset=['tema_200'])
    tema_data = [{'time': int(r['time']), 'value': float(r['tema_200'])} for _, r in df_tema.iterrows()]

    # 3. Prepare Trade Entry Markers/Lines
    entry_lines_js = ""
    if not active_trades.empty:
        symbol_trades = active_trades[active_trades['symbol'] == symbol]
        for _, trade in symbol_trades.iterrows():
            if pd.notnull(trade['entry_price']):
                entry_lines_js += f"""
                candlestickSeries.createPriceLine({{
                    price: {trade['entry_price']},
                    color: '#2962FF',
                    lineWidth: 2,
                    lineStyle: LightweightCharts.LineStyle.Dashed,
                    axisLabelVisible: true,
                    title: 'Entry #{trade['id']}',
                }});
                """
                if pd.notnull(trade['stop_loss']):
                    entry_lines_js += f"""
                    candlestickSeries.createPriceLine({{
                        price: {trade['stop_loss']},
                        color: '#FF5252',
                        lineWidth: 1,
                        lineStyle: LightweightCharts.LineStyle.Dotted,
                        axisLabelVisible: true,
                        title: 'SL',
                    }});
                    """

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
        <style>
            body {{ margin: 0; padding: 0; background-color: #131722; }}
            #chart {{ width: 100%; height: 550px; }}
        </style>
    </head>
    <body>
        <div id="chart"></div>
        <script>
            const chartContainer = document.getElementById('chart');
            const chart = LightweightCharts.createChart(chartContainer, {{
                layout: {{
                    background: {{ type: 'solid', color: '#131722' }},
                    textColor: '#d1d4dc',
                }},
                grid: {{
                    vertLines: {{ color: 'rgba(42, 46, 57, 0.5)' }},
                    horzLines: {{ color: 'rgba(42, 46, 57, 0.5)' }},
                }},
                crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
                rightPriceScale: {{ borderColor: 'rgba(197, 203, 206, 0.8)' }},
                timeScale: {{
                    borderColor: 'rgba(197, 203, 206, 0.8)',
                    timeVisible: true,
                    secondsVisible: false,
                }},
            }});

            const candlestickSeries = chart.addCandlestickSeries({{
                upColor: '#26a69a', downColor: '#ef5350', borderVisible: false,
                wickUpColor: '#26a69a', wickDownColor: '#ef5350',
            }});
            candlestickSeries.setData({json.dumps(candles_data)});

            const temaSeries = chart.addLineSeries({{
                color: '#ff9800',
                lineWidth: 2,
                title: '200 TEMA',
            }});
            temaSeries.setData({json.dumps(tema_data)});

            {entry_lines_js}

            window.addEventListener('resize', () => {{
                chart.applyOptions({{ width: chartContainer.clientWidth }});
            }});
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=560)

# =====================================================================
# CONFIG MANAGER
# =====================================================================
CONFIG_FILE = "config.json"

def load_config():
    default_config = {
        "account_balance": 1000.0,
        "risk_pct": 1.0,
        "proximity_threshold_pct": 2.0,
        "min_adx": 20.0,
        "min_atr_pct": 0.4,
        "scan_interval_minutes": 15,
        "alert_cooldown_hours": 4,
        "watchlist": ["ONDO/USDT", "PENDLE/USDT", "LINK/USDT", "TIA/USDT", "NEAR/USDT", "SYRUP/USDT"]
    }
    if not os.path.exists(CONFIG_FILE):
        return default_config
    try:
        with open(CONFIG_FILE, "r") as f:
            return {**default_config, **json.load(f)}
    except Exception:
        return default_config

def save_config(config_data):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config_data, f, indent=4)
        st.toast("Settings saved!", icon="💾")
    except Exception as e:
        st.error(f"Failed to save settings: {e}")

# =====================================================================
# UI LAYOUT
# =====================================================================
st.title("🎯 Trade Sniper Dashboard")
st.caption("Live Structural Trade Monitor & Strategy Control Center")

config = load_config()
df_journal = load_trade_journal()

# --- SIDEBAR: BOT PARAMETERS ---
with st.sidebar:
    st.header("⚙️ Bot Parameters")
    
    with st.form("config_form"):
        account_balance = st.number_input("Account Balance ($)", value=float(config.get("account_balance", 1000.0)), step=50.0)
        risk_pct = st.number_input("Risk Per Trade (%)", value=float(config.get("risk_pct", 1.0)), step=0.25)
        proximity_thresh = st.number_input("Proximity Threshold (%)", value=float(config.get("proximity_threshold_pct", 2.0)), step=0.5)
        min_adx = st.number_input("Min ADX Filter", value=float(config.get("min_adx", 20.0)), step=1.0)
        min_atr = st.number_input("Min ATR % Filter", value=float(config.get("min_atr_pct", 0.4)), step=0.1)
        scan_interval = st.number_input("Scan Interval (mins)", value=int(config.get("scan_interval_minutes", 15)), step=1)
        
        watchlist_raw = st.text_area("Watchlist (comma-separated)", value=", ".join(config.get("watchlist", [])))
        
        submitted = st.form_submit_button("Save Configuration")
        if submitted:
            new_watchlist = [symbol.strip().upper() for symbol in watchlist_raw.split(",") if symbol.strip()]
            updated_config = {
                "account_balance": account_balance,
                "risk_pct": risk_pct,
                "proximity_threshold_pct": proximity_thresh,
                "min_adx": min_adx,
                "min_atr_pct": min_atr,
                "scan_interval_minutes": scan_interval,
                "alert_cooldown_hours": config.get("alert_cooldown_hours", 4),
                "watchlist": new_watchlist
            }
            save_config(updated_config)
            st.rerun()

    st.divider()
    if st.button("Refresh Dashboard", use_container_width=True):
        st.rerun()

# --- TOP METRICS ROW ---
m1, m2, m3, m4, m5 = st.columns(5)

if not df_journal.empty:
    total_trades = len(df_journal)
    open_trades = len(df_journal[df_journal['status'].isin(['OPEN', 'TP1_HIT'])])
    closed_trades = df_journal[df_journal['status'].isin(['CLOSED_TP2', 'STOPPED_OUT'])]
    
    net_pnl = closed_trades['realized_pnl_usd'].sum() if 'realized_pnl_usd' in closed_trades.columns else 0.0
    total_r = closed_trades['realized_r'].sum() if 'realized_r' in closed_trades.columns else 0.0
    
    wins = len(closed_trades[closed_trades['realized_pnl_usd'] > 0]) if not closed_trades.empty else 0
    win_rate = (wins / len(closed_trades) * 100) if len(closed_trades) > 0 else 0.0

    m1.metric("Total Signals", total_trades)
    m2.metric("Active Trades", open_trades)
    m3.metric("Net Realized PnL", f"${net_pnl:.2f}", delta=f"{total_r:.2f}R")
    m4.metric("Win Rate", f"{win_rate:.1f}%")
    m5.metric("Database", "Supabase (Live)", delta="Online")
else:
    m1.metric("Total Signals", "0")
    m2.metric("Active Trades", "0")
    m3.metric("Net Realized PnL", "$0.00")
    m4.metric("Win Rate", "0.0%")
    m5.metric("Database", "Supabase (Live)", delta="Connected")

st.divider()

# --- MAIN TABS ---
tab_active, tab_history, tab_charts, tab_database = st.tabs([
    "🔥 Active Trades", 
    "📜 Closed History", 
    "📈 TradingView Chart", 
    "🛠️ Database Operations"
])

# TAB 1: ACTIVE TRADES
with tab_active:
    st.subheader("Currently Open Positions")
    if not df_journal.empty:
        active_df = df_journal[df_journal['status'].isin(['OPEN', 'TP1_HIT'])].copy()
        if not active_df.empty:
            display_cols = ['id', 'timestamp', 'symbol', 'status', 'entry_price', 'stop_loss', 'take_profit_1', 'take_profit_2', 'position_usdt', 'max_risk_usd', 'trigger_reason']
            st.dataframe(active_df[display_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No active signals currently open.")
    else:
        st.info("No trade data found in database.")

# TAB 2: CLOSED HISTORY
with tab_history:
    st.subheader("Closed Trade Performance")
    if not df_journal.empty:
        closed_df = df_journal[df_journal['status'].isin(['CLOSED_TP2', 'STOPPED_OUT'])].copy()
        if not closed_df.empty:
            display_cols = ['id', 'timestamp', 'closed_timestamp', 'symbol', 'status', 'entry_price', 'exit_price', 'realized_pnl_usd', 'realized_r']
            st.dataframe(
                closed_df[display_cols].style.format({
                    'entry_price': '${:.4f}',
                    'exit_price': '${:.4f}',
                    'realized_pnl_usd': '${:.2f}',
                    'realized_r': '{:.2f}R'
                }),
                use_container_width=True, 
                hide_index=True
            )
        else:
            st.info("No closed trades recorded yet.")
    else:
        st.info("No trade data found in database.")

# TAB 3: TRADINGVIEW CHART
with tab_charts:
    st.subheader("Interactive TradingView Canvas")
    
    col_sym, col_tf = st.columns([2, 1])
    with col_sym:
        selected_symbol = st.selectbox("Select Asset", config.get("watchlist", ["NEAR/USDT"]))
    with col_tf:
        selected_tf = st.selectbox("Timeframe", ["15m", "1h", "4h"], index=1)
        
    df_chart = fetch_market_data(selected_symbol, timeframe=selected_tf)
    
    if not df_chart.empty:
        render_tradingview_chart(df_chart, selected_symbol, df_journal)
    else:
        st.warning("Could not fetch market data for the selected symbol.")

# TAB 4: DATABASE OPERATIONS
with tab_database:
    st.subheader("Raw Supabase Table Viewer & Storage Control")
    if not df_journal.empty:
        st.dataframe(df_journal, use_container_width=True, hide_index=True)
        st.divider()
        st.warning("⚠️ Dangerous Operations Zone")
        confirm_clear = st.checkbox("I confirm I want to wipe all records in the remote Supabase database.")
        if st.button("Wipe Remote Journal Database", type="primary", disabled=not confirm_clear):
            clear_remote_journal()
    else:
        st.info("Journal database is currently empty.")