import os
import json
import warnings
import ccxt
import pandas as pd
import pandas_ta as ta
import streamlit as st
import streamlit.components.v1 as components

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
        "NEAR/USDT"
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
        st.sidebar.success("Configuration updated successfully!")
    except Exception as e:
        st.sidebar.error(f"Failed to save configuration: {e}")

def load_journal() -> pd.DataFrame:
    if os.path.exists(JOURNAL_FILE):
        try:
            df = pd.read_csv(JOURNAL_FILE)
            expected_cols = [
                "Timestamp", "Symbol", "Trigger_Reason", "Entry_Price", 
                "Stop_Loss", "Take_Profit_1", "Take_Profit_2", "Position_USDT", 
                "Max_Risk_USD", "Status", "Exit_Price", "Closed_Timestamp", 
                "Realized_PnL_USD", "Realized_R"
            ]
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = None
            return df
        except Exception as e:
            st.error(f"Error loading trade journal: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

# =====================================================================
# REFINED BACKTESTING ENGINE
# =====================================================================
def fetch_backtest_data(symbol, timeframe='1h', limit=1000):
    try:
        exchange = ccxt.mexc({'enableRateLimit': True, 'timeout': 20000})
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Core Indicators
        df['tema_200'] = ta.tema(df['close'], length=200)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        # ADX Calculation
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx_df is not None and not adx_df.empty:
            df['adx'] = adx_df.iloc[:, 0]
        else:
            df['adx'] = 25.0
            
        return df
    except Exception as e:
        st.error(f"Failed to fetch market data for {symbol}: {e}")
        return None

def run_backtest_simulation(symbol, df, risk_pct=1.0, initial_capital=1000.0, proximity_pct=1.5, min_adx=18.0, min_atr_pct=0.4):
    if df is None or df.empty or 'tema_200' not in df.columns:
        return None, None

    # Drop early warmup rows where indicators are NaN
    clean_df = df.dropna(subset=['tema_200', 'atr']).copy().reset_index(drop=True)
    if clean_df.empty:
        return None, None

    trades = []
    capital = initial_capital
    in_trade = False
    entry_price = 0.0
    stop_loss = 0.0
    tp1 = 0.0
    tp2 = 0.0
    risk_usd = 0.0

    for i in range(len(clean_df)):
        current_row = clean_df.iloc[i]
        close_p = current_row['close']
        high_p = current_row['high']
        low_p = current_row['low']
        tema_p = current_row['tema_200']
        atr_p = current_row['atr']
        adx_p = current_row.get('adx', 20.0)

        if pd.isna(tema_p) or pd.isna(atr_p):
            continue

        if in_trade:
            # Check Stop Loss
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
            # Check Take Profit 2
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
            # Check proximity across close or low wick tests
            dist_close_pct = abs(close_p - tema_p) / tema_p * 100.0
            dist_low_pct = abs(low_p - tema_p) / tema_p * 100.0
            min_dist = min(dist_close_pct, dist_low_pct)
            
            atr_pct = (atr_p / close_p) * 100.0 if close_p > 0 else 0.0
            
            if min_dist <= proximity_pct and atr_pct >= min_atr_pct and adx_p >= min_adx:
                in_trade = True
                entry_price = close_p
                stop_loss = entry_price - (1.5 * atr_p)
                tp1 = entry_price + (1.5 * atr_p)
                tp2 = entry_price + (3.0 * atr_p)
                risk_usd = capital * (risk_pct / 100.0)

    trades_df = pd.DataFrame(trades)
    return trades_df, capital

# =====================================================================
# MAIN DASHBOARD LAYOUT
# =====================================================================
st.title("🎯 Trade Sniper Dashboard & Controls")

# --- SIDEBAR CONFIGURATION MANAGER ---
st.sidebar.header("⚙️ Bot Parameters")
config = load_config()

account_balance = st.sidebar.number_input(
    "Account Balance ($)", 
    value=float(config.get("account_balance", 1000.0)), 
    step=100.0
)
risk_pct = st.sidebar.number_input(
    "Risk Per Trade (%)", 
    value=float(config.get("risk_pct", 1.0)), 
    step=0.25
)
proximity_threshold = st.sidebar.number_input(
    "Proximity Threshold (%)", 
    value=float(config.get("proximity_threshold_pct", 1.5)), 
    step=0.1
)
min_adx = st.sidebar.number_input(
    "Min ADX Filter", 
    value=float(config.get("min_adx", 18.0)), 
    step=1.0
)
min_atr_pct = st.sidebar.number_input(
    "Min ATR % Filter", 
    value=float(config.get("min_atr_pct", 0.4)), 
    step=0.1
)
scan_interval = st.sidebar.number_input(
    "Scan Interval (Mins)", 
    value=int(config.get("scan_interval_minutes", 15)), 
    step=1
)
cooldown_hours = st.sidebar.number_input(
    "Alert Cooldown (Hours)", 
    value=int(config.get("alert_cooldown_hours", 4)), 
    step=1
)

st.sidebar.markdown("---")
st.sidebar.subheader("📌 Watchlist Manager")

# Single Custom Symbol Adder
custom_symbol_input = st.sidebar.text_input("Add Single Symbol (e.g., SOL/USDT)", "").strip().upper()
if st.sidebar.button("➕ Add to Watchlist", use_container_width=True):
    if custom_symbol_input:
        current_list = config.get("watchlist", [])
        if custom_symbol_input not in current_list:
            current_list.append(custom_symbol_input)
            config["watchlist"] = current_list
            save_config(config)
            st.rerun()
        else:
            st.sidebar.warning(f"{custom_symbol_input} is already in your watchlist.")

# Editable Text Area for Full Watchlist
watchlist_str = st.sidebar.text_area(
    "Active Watchlist (Comma Separated)", 
    value=", ".join(config.get("watchlist", []))
)

if st.sidebar.button("💾 Save All Settings", use_container_width=True):
    updated_watchlist = [symbol.strip().upper() for symbol in watchlist_str.split(",") if symbol.strip()]
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
# TABS SYSTEM
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
        display_df = journal_df.fillna("—").sort_index(ascending=False)
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("No trades logged in `trade_journal.csv` yet. Waiting for structural proximity signals.")

# --- TAB 2: CHARTING ---
with tab2:
    st.subheader("📈 TradingView Live Charting")
    selected_symbol = st.selectbox("Select Pair to Chart", config.get("watchlist", ["NEAR/USDT"]))
    tv_symbol = f"MEXC:{selected_symbol.replace('/', '')}"
    
    tradingview_html = f"""
    <div class="tradingview-widget-container" style="height:100%;width:100%">
      <div id="tradingview_chart" style="height:550px;width:100%"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget(
      {{
        "autosize": true,
        "symbol": "{tv_symbol}",
        "interval": "60",
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "enable_publishing": false,
        "hide_legend": false,
        "save_image": false,
        "container_id": "tradingview_chart"
      }}
      );
      </script>
    </div>
    """
    components.html(tradingview_html, height=560, scrolling=False)

# --- TAB 3: CLOSED TRADES ANALYTICS ---
with tab3:
    st.subheader("📋 Closed Trades & Realized Execution")
    journal_df = load_journal()
    if not journal_df.empty:
        closed_df = journal_df[journal_df["Status"].isin(["STOPPED_OUT", "CLOSED_TP2"])].dropna(subset=["Status"])
        if not closed_df.empty:
            st.write("### Realized Execution History")
            st.dataframe(closed_df.fillna("—").sort_index(ascending=False), use_container_width=True)
        else:
            st.info("No trades have hit Stop Loss or TP2 yet.")
    else:
        st.info("Historical execution performance will populate here.")

# --- TAB 4: HISTORICAL BACKTESTER ---
with tab4:
    st.subheader("🧪 Quantitative Strategy Backtester")
    st.markdown("Simulate structural TEMA proximity strategies against historical OHLCV candles.")

    b_col1, b_col2, b_col3 = st.columns(3)
    
    active_watchlist = config.get("watchlist", ["NEAR/USDT"])
    selected_preset = b_col1.selectbox("Select Target Pair", ["Custom Symbol..."] + active_watchlist)
    
    if selected_preset == "Custom Symbol...":
        bt_symbol = st.text_input("Enter Symbol Pair (e.g., SOL/USDT)", value="SOL/USDT").strip().upper()
    else:
        bt_symbol = selected_preset

    bt_candles = b_col2.slider("Candle Lookback (1H)", min_value=200, max_value=1000, value=500, step=50)
    bt_capital = b_col3.number_input("Starting Capital ($)", value=1000.0, step=100.0)

    if st.button("🚀 Run Strategy Simulation", use_container_width=True):
        with st.spinner(f"Fetching historical data and backtesting {bt_symbol}..."):
            df_bt = fetch_backtest_data(bt_symbol, timeframe='1h', limit=bt_candles)
            if df_bt is not None and not df_bt.empty:
                results_df, final_cap = run_backtest_simulation(
                    symbol=bt_symbol, 
                    df=df_bt, 
                    risk_pct=risk_pct, 
                    initial_capital=bt_capital, 
                    proximity_pct=proximity_threshold,
                    min_adx=min_adx,
                    min_atr_pct=min_atr_pct
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
                    st.dataframe(results_df.sort_index(ascending=False), use_container_width=True)
                else:
                    st.warning(f"No structural triggers matched for {bt_symbol} during the selected historical window.")