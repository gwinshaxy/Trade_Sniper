import os
import json
import time
import warnings
import csv
import requests
import pandas as pd
import numpy as np
import pandas_ta as ta
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Suppress minor library warnings for clean UI log outputs
warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================================
# PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="Trade Sniper Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

CONFIG_FILE = "config.json"
JOURNAL_FILE = "trade_journal.csv"

DEFAULT_CONFIG = {
    "account_balance": 1000.0,
    "risk_pct": 1.0,
    "proximity_threshold_pct": 1.5,
    "min_adx": 18.0,
    "min_atr_pct": 0.4,
    "scan_interval_minutes": 15,
    "alert_cooldown_hours": 4,
    "journal_file": JOURNAL_FILE,
    "watchlist": [
        "ONDO/USDT",
        "PENDLE/USDT",
        "LINK/USDT",
        "TIA/USDT",
        "NEAR/USDT",
        "XRP/USDT"
    ]
}

# =====================================================================
# DATA & CONFIG LOADERS
# =====================================================================
def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
                return {**DEFAULT_CONFIG, **config}
        except Exception:
            return DEFAULT_CONFIG
    return DEFAULT_CONFIG

def save_config(config_data: dict):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config_data, f, indent=4)
        st.sidebar.success("Configuration saved!")
    except Exception as e:
        st.sidebar.error(f"Failed to save configuration: {e}")

@st.cache_data(ttl=5)
def load_journal() -> pd.DataFrame:
    if os.path.exists(JOURNAL_FILE):
        try:
            df = pd.read_csv(JOURNAL_FILE, na_values=["—", "-", "N/A", "nan", "None", ""])
            
            expected_cols = [
                "Timestamp", "Symbol", "Trigger_Reason", "Entry_Price", 
                "Stop_Loss", "Take_Profit_1", "Take_Profit_2", "Position_USDT", 
                "Max_Risk_USD", "Status", "Exit_Price", "Closed_Timestamp", 
                "Realized_PnL_USD", "Realized_R"
            ]
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = np.nan

            numeric_cols = [
                "Entry_Price", "Stop_Loss", "Take_Profit_1", "Take_Profit_2", 
                "Position_USDT", "Max_Risk_USD", "Exit_Price", "Realized_PnL_USD", "Realized_R"
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            string_cols = ["Timestamp", "Symbol", "Trigger_Reason", "Status", "Closed_Timestamp"]
            for col in string_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str).replace({'nan': '', 'None': ''})

            return df
        except Exception as e:
            st.error(f"Error loading trade journal: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

# =====================================================================
# DATA FETCHING & TECHNICAL ENGINE
# =====================================================================
def sanitize_symbol(raw_symbol: str) -> str:
    clean = raw_symbol.strip().upper()
    if "/" not in clean and clean.endswith("USDT"):
        clean = clean[:-4] + "/USDT"
    return clean

def fetch_backtest_data(symbol: str, timeframe='1h', limit=1000):
    formatted_symbol = sanitize_symbol(symbol).replace('/', '').upper()
    fetch_limit = max(limit, 1000)
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }

    raw_candles = None
    provider = None

    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={formatted_symbol}&interval={timeframe}&limit={fetch_limit}"
        res = requests.get(url, headers=headers, timeout=8)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) > 0:
                raw_candles = [[c[0], c[1], c[2], c[3], c[4], c[5]] for c in data]
                provider = "Binance Global"
    except Exception:
        pass

    if not raw_candles:
        try:
            url = f"https://api.binance.us/api/v3/klines?symbol={formatted_symbol}&interval={timeframe}&limit={fetch_limit}"
            res = requests.get(url, headers=headers, timeout=8)
            if res.status_code == 200:
                data = res.json()
                if isinstance(data, list) and len(data) > 0:
                    raw_candles = [[c[0], c[1], c[2], c[3], c[4], c[5]] for c in data]
                    provider = "Binance US"
        except Exception:
            pass

    if not raw_candles:
        try:
            url = f"https://api.bybit.com/v5/market/kline?category=spot&symbol={formatted_symbol}&interval=60&limit={fetch_limit}"
            res = requests.get(url, headers=headers, timeout=8)
            if res.status_code == 200:
                data = res.json()
                if data.get('retCode') == 0 and len(data['result']['list']) > 0:
                    klist = data['result']['list'][::-1]
                    raw_candles = [[float(c[0]), c[1], c[2], c[3], c[4], c[5]] for c in klist]
                    provider = "Bybit REST"
        except Exception:
            pass

    if not raw_candles or len(raw_candles) == 0:
        st.error(f"Unable to fetch history for {symbol} across public endpoints.")
        return None

    df = pd.DataFrame(raw_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='ms')
    df.drop_duplicates(subset=['timestamp'], inplace=True)
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')

    df['tema_200'] = ta.tema(df['close'], length=200)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    if adx_df is not None and not adx_df.empty:
        adx_cols = [c for c in adx_df.columns if c.startswith('ADX_')]
        df['adx'] = adx_df[adx_cols[0]] if adx_cols else adx_df.iloc[:, 0]
    else:
        df['adx'] = 25.0

    df.attrs['source_exchange'] = provider
    return df

def run_backtest_simulation(symbol, df, risk_pct=1.0, initial_capital=1000.0, proximity_pct=1.5, min_adx=18.0, min_atr_pct=0.4, use_filters=True):
    if df is None or df.empty or 'tema_200' not in df.columns:
        return None, None, {"error": "DataFrame is uninitialized or missing TEMA."}

    clean_df = df.dropna(subset=['tema_200', 'atr']).copy().reset_index(drop=True)
    if clean_df.empty:
        return None, None, {"error": "All rows dropped during TEMA 200 warmup window."}

    trades = []
    capital = float(initial_capital)
    in_trade = False
    entry_price = 0.0
    stop_loss = 0.0
    tp1 = 0.0
    tp2 = 0.0
    risk_usd = 0.0

    min_proximities = []
    adx_values = []
    atr_pct_values = []

    for i in range(len(clean_df)):
        current_row = clean_df.iloc[i]
        close_p = float(current_row['close'])
        high_p = float(current_row['high'])
        low_p = float(current_row['low'])
        tema_p = float(current_row['tema_200'])
        atr_p = float(current_row['atr'])
        adx_p = float(current_row.get('adx', 25.0)) if not pd.isna(current_row.get('adx')) else 25.0

        if pd.isna(tema_p) or pd.isna(atr_p) or tema_p == 0:
            continue

        dist_close_pct = abs(close_p - tema_p) / tema_p * 100.0
        dist_low_pct = abs(low_p - tema_p) / tema_p * 100.0
        dist_high_pct = abs(high_p - tema_p) / tema_p * 100.0
        min_dist = min(dist_close_pct, dist_low_pct, dist_high_pct)
        atr_pct = (atr_p / close_p) * 100.0 if close_p > 0 else 0.0

        min_proximities.append(min_dist)
        adx_values.append(adx_p)
        atr_pct_values.append(atr_pct)

        if in_trade:
            if low_p <= stop_loss:
                pnl = -risk_usd
                capital += pnl
                trades.append({
                    "Timestamp": current_row['timestamp'],
                    "Symbol": symbol,
                    "Type": "LONG",
                    "Entry": round(entry_price, 4),
                    "Exit": round(stop_loss, 4),
                    "Result": "STOPPED_OUT",
                    "PnL ($)": round(pnl, 2),
                    "Capital ($)": round(capital, 2)
                })
                in_trade = False

            elif high_p >= tp2:
                pnl = risk_usd * 2.0
                capital += pnl
                trades.append({
                    "Timestamp": current_row['timestamp'],
                    "Symbol": symbol,
                    "Type": "LONG",
                    "Entry": round(entry_price, 4),
                    "Exit": round(tp2, 4),
                    "Result": "TP2_HIT",
                    "PnL ($)": round(pnl, 2),
                    "Capital ($)": round(capital, 2)
                })
                in_trade = False
        else:
            filters_passed = True
            if use_filters:
                if adx_p < min_adx or atr_pct < min_atr_pct:
                    filters_passed = False

            if min_dist <= proximity_pct and filters_passed:
                in_trade = True
                entry_price = close_p
                stop_loss = entry_price - (1.5 * atr_p)
                tp1 = entry_price + (1.5 * atr_p)
                tp2 = entry_price + (3.0 * atr_p)
                risk_usd = capital * (risk_pct / 100.0)

    if in_trade and len(clean_df) > 0:
        last_row = clean_df.iloc[-1]
        exit_p = float(last_row['close'])
        pnl = ((exit_p - entry_price) / entry_price) * risk_usd if entry_price > 0 else 0.0
        capital += pnl
        trades.append({
            "Timestamp": last_row['timestamp'],
            "Symbol": symbol,
            "Type": "LONG",
            "Entry": round(entry_price, 4),
            "Exit": round(exit_p, 4),
            "Result": "ACTIVE_AT_END",
            "PnL ($)": round(pnl, 2),
            "Capital ($)": round(capital, 2)
        })

    trades_df = pd.DataFrame(trades)
    diagnostics = {
        "source_exchange": df.attrs.get('source_exchange', 'Unknown'),
        "raw_candles_fetched": len(df),
        "valid_candles_tested": len(clean_df),
        "closest_proximity_found": round(min(min_proximities), 2) if min_proximities else None,
        "max_adx_found": round(max(adx_values), 2) if adx_values else None,
        "max_atr_pct_found": round(max(atr_pct_values), 2) if atr_pct_values else None,
    }
    return trades_df, capital, diagnostics

# =====================================================================
# DASHBOARD LAYOUT
# =====================================================================
st.title("🎯 Trade Sniper Dashboard & Controls")

# --- SIDEBAR CONFIGURATION MANAGER ---
st.sidebar.header("⚙️ Bot Parameters")
config = load_config()

account_balance = st.sidebar.number_input("Account Balance ($)", value=float(config.get("account_balance", 1000.0)), step=100.0)
risk_pct = st.sidebar.number_input("Risk Per Trade (%)", value=float(config.get("risk_pct", 1.0)), step=0.25)
proximity_threshold = st.sidebar.number_input("Proximity Threshold (%)", value=float(config.get("proximity_threshold_pct", 3.0)), step=0.1)
min_adx = st.sidebar.number_input("Min ADX Filter", value=float(config.get("min_adx", 18.0)), step=1.0)
min_atr_pct = st.sidebar.number_input("Min ATR % Filter", value=float(config.get("min_atr_pct", 0.4)), step=0.1)
scan_interval = st.sidebar.number_input("Scan Interval (Mins)", value=int(config.get("scan_interval_minutes", 15)), step=1)
cooldown_hours = st.sidebar.number_input("Alert Cooldown (Hours)", value=int(config.get("alert_cooldown_hours", 4)), step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("📌 Watchlist Manager")

custom_symbol_input = st.sidebar.text_input("Add Single Symbol (e.g., SOL/USDT)", "").strip().upper()
if st.sidebar.button("➕ Add to Watchlist"):
    if custom_symbol_input:
        formatted = sanitize_symbol(custom_symbol_input)
        current_list = config.get("watchlist", [])
        if formatted not in current_list:
            current_list.append(formatted)
            config["watchlist"] = current_list
            save_config(config)
            st.rerun()
        else:
            st.sidebar.warning(f"{formatted} is already in your watchlist.")

watchlist_str = st.sidebar.text_area("Active Watchlist (Comma Separated)", value=", ".join(config.get("watchlist", [])))

if st.sidebar.button("💾 Save All Settings"):
    updated_watchlist = [sanitize_symbol(symbol) for symbol in watchlist_str.split(",") if symbol.strip()]
    updated_config = {
        "account_balance": account_balance,
        "risk_pct": risk_pct,
        "proximity_threshold_pct": proximity_threshold,
        "min_adx": min_adx,
        "min_atr_pct": min_atr_pct,
        "scan_interval_minutes": scan_interval,
        "alert_cooldown_hours": cooldown_hours,
        "journal_file": JOURNAL_FILE,
        "watchlist": updated_watchlist
    }
    save_config(updated_config)

# =====================================================================
# MAIN DASHBOARD TABS
# =====================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Live Journal & Evaluator", 
    "📈 Charting", 
    "📋 Closed Trades Analytics", 
    "🧪 Historical Backtester"
])

# --- TAB 1: LIVE JOURNAL ---
with tab1:
    st.subheader("📋 Structural Trade Journal & Evaluator")
    
    col_title, col_reset = st.columns([0.8, 0.2])
    with col_reset:
        if st.button("🗑️ Clear Journal", use_container_width=True):
            headers = [
                "Timestamp", "Symbol", "Trigger_Reason", "Entry_Price", 
                "Stop_Loss", "Take_Profit_1", "Take_Profit_2", "Position_USDT", 
                "Max_Risk_USD", "Status", "Exit_Price", "Closed_Timestamp", 
                "Realized_PnL_USD", "Realized_R"
            ]
            with open("trade_journal.csv", mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            st.success("Journal cleared!")
            st.rerun() 

    journal_df = load_journal()

    if not journal_df.empty:
        total_signals = len(journal_df)
        open_signals = len(journal_df[journal_df["Status"].isin(["OPEN", "TP1_HIT"])])
        
        pnl_series = pd.to_numeric(journal_df["Realized_PnL_USD"], errors="coerce").fillna(0.0)
        total_realized_pnl = pnl_series.sum()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Signals Logged", total_signals)
        col2.metric("Active / Open Positions", open_signals)
        col3.metric("Net Realized PnL ($)", f"${total_realized_pnl:+.2f}")
        col4.metric("Risk Setting", f"{risk_pct}% / Trade")

        st.markdown("---")
        display_df = journal_df.sort_index(ascending=False)
        st.dataframe(display_df, hide_index=True)
    else:
        st.info("No trades logged in `trade_journal.csv` yet. Signals will display here as they trigger.")

# --- TAB 2: CHARTING ---
with tab2:
    st.subheader("📈 Interactive Strategy Charting")
    selected_symbol = st.selectbox("Select Pair to Chart", config.get("watchlist", ["NEAR/USDT"]))
    
    col_tf, col_candles = st.columns(2)
    chart_tf = col_tf.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=1)
    chart_limit = col_candles.slider("Candle Limit", min_value=50, max_value=500, value=150, step=25)

    if st.button("🔄 Refresh Chart Data", key="refresh_chart"):
        st.rerun()

    with st.spinner(f"Loading live chart for {selected_symbol}..."):
        df_full = fetch_backtest_data(selected_symbol, timeframe=chart_tf, limit=1000)

    if df_full is not None and not df_full.empty:
        df_full['tema_200'] = ta.tema(df_full['close'], length=200)

        df_chart = df_full.tail(chart_limit).copy().reset_index(drop=True)

        # Get latest values for metrics display
        latest_row = df_chart.iloc[-1]
        prev_row = df_chart.iloc[-2] if len(df_chart) > 1 else latest_row
        
        curr_price = float(latest_row['close'])
        prev_price = float(prev_row['close'])
        price_change_pct = ((curr_price - prev_price) / prev_price) * 100.0 if prev_price > 0 else 0.0
        
        curr_tema = float(latest_row['tema_200']) if not pd.isna(latest_row['tema_200']) else 0.0
        curr_adx = float(latest_row.get('adx', 0.0)) if not pd.isna(latest_row.get('adx')) else 0.0
        curr_atr = float(latest_row.get('atr', 0.0)) if not pd.isna(latest_row.get('atr')) else 0.0
        curr_atr_pct = (curr_atr / curr_price) * 100.0 if curr_price > 0 else 0.0
        
        proximity_pct = (abs(curr_price - curr_tema) / curr_tema) * 100.0 if curr_tema > 0 else 0.0

        # Live Metric Cards Above Chart
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("Current Price", f"${curr_price:.4f}", f"{price_change_pct:+.2f}%")
        m_col2.metric("200 TEMA", f"${curr_tema:.4f}")
        m_col3.metric("TEMA Proximity", f"{proximity_pct:.2f}%")
        m_col4.metric("ADX (14)", f"{curr_adx:.2f}")
        m_col5.metric("ATR %", f"{curr_atr_pct:.2f}%")

        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.08, 
            subplot_titles=(f"{selected_symbol} ({chart_tf.upper()}) Price, Volume & 200 TEMA", "ADX Trend Strength Indicator"),
            row_heights=[0.7, 0.3],
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )

        # 1. Solid, High-Visibility Candlesticks
        fig.add_trace(
            go.Candlestick(
                x=df_chart['timestamp'].dt.strftime('%Y-%m-%d %H:%M'),
                open=df_chart['open'],
                high=df_chart['high'],
                low=df_chart['low'],
                close=df_chart['close'],
                name="OHLC",
                increasing=dict(
                    fillcolor='#089981', 
                    line=dict(color='#089981', width=1.5)
                ),
                decreasing=dict(
                    fillcolor='#f23645', 
                    line=dict(color='#f23645', width=1.5)
                ),
                whiskerwidth=0.8
            ),
            row=1, col=1, secondary_y=False
        )

        # 2. Volume Bars
        volume_colors = [
            'rgba(8, 153, 129, 0.4)' if close >= open_p else 'rgba(242, 54, 69, 0.4)'
            for close, open_p in zip(df_chart['close'], df_chart['open'])
        ]
        fig.add_trace(
            go.Bar(
                x=df_chart['timestamp'].dt.strftime('%Y-%m-%d %H:%M'),
                y=df_chart['volume'],
                name="Volume",
                marker_color=volume_colors,
                showlegend=False
            ),
            row=1, col=1, secondary_y=True
        )

        # 3. 200 TEMA Line
        if 'tema_200' in df_chart.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_chart['timestamp'].dt.strftime('%Y-%m-%d %H:%M'),
                    y=df_chart['tema_200'],
                    mode='lines',
                    name='200 TEMA',
                    line=dict(color='#ff9800', width=2.5),
                    connectgaps=True
                ),
                row=1, col=1, secondary_y=False
            )

        # 4. ADX Line
        if 'adx' in df_chart.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_chart['timestamp'].dt.strftime('%Y-%m-%d %H:%M'),
                    y=df_chart['adx'],
                    mode='lines',
                    name='ADX (14)',
                    line=dict(color='#29b6f6', width=2.0),
                    connectgaps=True
                ),
                row=2, col=1
            )
            fig.add_hline(
                y=min_adx, 
                line_dash="dash", 
                line_color="rgba(255, 152, 0, 0.6)", 
                row=2, col=1, 
                annotation_text=f"Min ADX ({min_adx})"
            )

        # Pin dynamic annotations to the latest candle index
        last_x_val = df_chart['timestamp'].dt.strftime('%Y-%m-%d %H:%M').iloc[-1]
        
        fig.add_hline(
            y=curr_price, 
            line_dash="dot", 
            line_color="#089981" if curr_price >= prev_price else "#f23645", 
            row=1, col=1, 
            secondary_y=False
        )
        
        fig.add_annotation(
            x=last_x_val, y=curr_price,
            text=f" Price: ${curr_price:.4f}",
            showarrow=True, arrowhead=2, ax=50, ay=0,
            bgcolor="#089981" if curr_price >= prev_price else "#f23645",
            font=dict(color="white", size=11),
            row=1, col=1
        )
        
        if curr_tema > 0:
            fig.add_annotation(
                x=last_x_val, y=curr_tema,
                text=f" TEMA: ${curr_tema:.4f}",
                showarrow=True, arrowhead=2, ax=50, ay=25,
                bgcolor="#ff9800",
                font=dict(color="white", size=11),
                row=1, col=1
            )

        if curr_adx > 0:
            fig.add_annotation(
                x=last_x_val, y=curr_adx,
                text=f" ADX: {curr_adx:.1f}",
                showarrow=True, arrowhead=2, ax=50, ay=0,
                bgcolor="#29b6f6",
                font=dict(color="white", size=10),
                row=2, col=1
            )

        fig.update_layout(
            height=720,
            margin=dict(l=10, r=60, t=40, b=10),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=0.85
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_rangeslider_visible=False
        )

        grid_color = "rgba(128, 128, 128, 0.15)"

        # --- DYNAMIC AUTO-SCALING AXIS SETUP ---
        fig.update_xaxes(
            showgrid=True, 
            gridcolor=grid_color, 
            zeroline=False,
            fixedrange=False,
            type='category'  # Ensures rigid spacing for uniform candle bodies
        )

        fig.update_yaxes(
            autorange=True,
            fixedrange=False,
            showgrid=True, 
            gridcolor=grid_color, 
            zeroline=False, 
            secondary_y=False, 
            row=1, col=1
        )

        # Volume secondary Y-axis configuration
        max_vol = df_chart['volume'].max() if not df_chart['volume'].empty else 1.0
        fig.update_yaxes(
            range=[0, max_vol * 4], 
            showgrid=False, 
            secondary_y=True, 
            row=1, col=1
        )

        # ADX Subplot Y-axis setup
        fig.update_yaxes(
            autorange=True,
            fixedrange=False,
            showgrid=True, 
            gridcolor=grid_color, 
            zeroline=False, 
            row=2, col=1
        )

        # Render with scroll zoom and modebar enabled
        st.plotly_chart(
            fig, 
            use_container_width=True,
            config={
                'scrollZoom': True,
                'displayModeBar': True,
                'displaylogo': False
            }
        )
    else:
        st.error(f"Failed to render chart data for {selected_symbol}.")

# --- TAB 3: CLOSED TRADES ANALYTICS ---
with tab3:
    st.subheader("📋 Closed Trades & Realized Execution")
    journal_df = load_journal()
    if not journal_df.empty:
        closed_df = journal_df[journal_df["Status"].isin(["STOPPED_OUT", "CLOSED_TP2"])].dropna(subset=["Status"])
        if not closed_df.empty:
            st.write("### Realized Execution History")
            st.dataframe(closed_df.sort_index(ascending=False), hide_index=True)
        else:
            st.info("No trades have hit Stop Loss or TP2 yet.")
    else:
        st.info("Historical execution performance will populate here.")

# --- TAB 4: HISTORICAL BACKTESTER ---
with tab4:
    st.subheader("🧪 Quantitative Strategy Backtester")
    st.markdown("Simulate structural TEMA proximity strategies against historical OHLCV candles.")

    b_col1, b_col2, b_col3 = st.columns(3)
    
    active_watchlist = config.get("watchlist", ["XRP/USDT", "NEAR/USDT"])
    selected_preset = b_col1.selectbox("Select Target Pair", ["Custom Symbol..."] + active_watchlist)
    
    if selected_preset == "Custom Symbol...":
        bt_symbol = st.text_input("Enter Symbol Pair (e.g., SOL/USDT)", value="SOL/USDT").strip().upper()
    else:
        bt_symbol = selected_preset

    bt_candles = b_col2.slider("Candle Lookback (1H)", min_value=200, max_value=1000, value=1000, step=100)
    bt_capital = b_col3.number_input("Starting Capital ($)", value=1000.0, step=100.0)

    use_regime_filters = st.checkbox("Enforce ADX & ATR Filters in Backtest", value=False, help="Uncheck to test pure structural proximity without indicator restrictions.")

    if st.button("🚀 Run Strategy Simulation"):
        with st.spinner(f"Fetching historical data and backtesting {bt_symbol}..."):
            df_bt = fetch_backtest_data(bt_symbol, timeframe='1h', limit=bt_candles)
            
            if df_bt is not None and not df_bt.empty:
                results_df, final_cap, diag = run_backtest_simulation(
                    symbol=bt_symbol, 
                    df=df_bt, 
                    risk_pct=risk_pct, 
                    initial_capital=bt_capital, 
                    proximity_pct=proximity_threshold,
                    min_adx=min_adx,
                    min_atr_pct=min_atr_pct,
                    use_filters=use_regime_filters
                )
                
                if results_df is not None and not results_df.empty:
                    net_profit = final_cap - bt_capital
                    win_trades = len(results_df[results_df["Result"] == "TP2_HIT"])
                    loss_trades = len(results_df[results_df["Result"] == "STOPPED_OUT"])
                    win_rate = (win_trades / len(results_df)) * 100.0 if len(results_df) > 0 else 0.0

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Final Capital ($)", f"${final_cap:.2f}")
                    m2.metric("Net Profit ($)", f"${net_profit:+.2f}")
                    m3.metric("Total Trades", len(results_df))
                    m4.metric("Win Rate (%)", f"{win_rate:.1f}%")

                    st.markdown("---")
                    st.write("### Simulated Trade History")
                    st.dataframe(results_df.sort_index(ascending=False), hide_index=True)
                else:
                    st.warning(f"No structural triggers matched for {bt_symbol} during the selected historical window.")
                    st.info(f"🔍 **Data Diagnostics ({bt_symbol}):**\n"
                            f"- Data Provider Used: **{diag.get('source_exchange')}**\n"
                            f"- Raw Candles Fetched: **{diag.get('raw_candles_fetched')}**\n"
                            f"- Valid Candles Tested (after TEMA 200 warmup): **{diag.get('valid_candles_tested')}**\n"
                            f"- Closest Distance to 200 TEMA: **{diag.get('closest_proximity_found')}%** (Your Threshold: {proximity_threshold}%)\n"
                            f"- Max ADX Value: **{diag.get('max_adx_found')}** (Filter: {min_adx})\n"
                            f"- Max ATR %: **{diag.get('max_atr_pct_found')}%** (Filter: {min_atr_pct}%)\n")