import os
import json
import asyncio
import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# --- STREAMLIT PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="Trade Sniper Dashboard & Backtester",
    page_icon="🛡️",
    layout="wide"
)

# --- AUTHENTICATION GUARD ---
def check_password():
    """Returns True if the user enters the correct password."""
    target_password = os.getenv("DASHBOARD_PASSWORD", "default_local_password")

    if st.session_state.get("password_correct", False):
        return True

    st.title("🔒 Trade Sniper Dashboard")
    st.subheader("Authentication Required")
    
    password_input = st.text_input("Enter Access Password", type="password")
    
    if st.button("Login"):
        if password_input == target_password:
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("❌ Incorrect password. Access denied.")
            
    return False

# Block execution if not authenticated
if not check_password():
    st.stop()

# --- CONFIGURATION FILE PATHS ---
CONFIG_FILE = "config.json"
JOURNAL_FILE = "trade_journal.csv"

# --- HELPER FUNCTIONS ---
def load_config():
    """Reads settings from config.json or returns default values."""
    if not os.path.exists(CONFIG_FILE):
        return {
            "account_balance": 1000.0,
            "risk_pct": 1.0,
            "proximity_threshold_pct": 1.0,
            "min_adx": 20.0,
            "min_atr_pct": 0.5,
            "scan_interval_minutes": 15,
            "alert_cooldown_hours": 4,
            "journal_file": "trade_journal.csv",
            "watchlist": ["ONDO/USDT", "PENDLE/USDT", "LINK/USDT", "TIA/USDT", "NEAR/USDT"]
        }
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_config(config_data):
    """Saves updated settings back into config.json."""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4)

config = load_config()

# --- BACKTESTING CORE LOGIC ---
def calculate_native_tema(series, length=200):
    ema1 = series.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    ema3 = ema2.ewm(span=length, adjust=False).mean()
    return 3 * (ema1 - ema2) + ema3

def fetch_historical_ohlcv(symbol, timeframe='1h', limit=1000):
    exchange = ccxt.mexc({'enableRateLimit': True})
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        st.error(f"Error fetching backtest data for {symbol}: {e}")
        return None

def run_backtest_simulation(symbol, df_1h, df_4h, initial_balance, risk_pct, proximity_threshold, sl_pct, tp1_mult, tp2_mult, cooldown_candles=4):
    # Prepare TEMAs
    try:
        df_1h['tema_200'] = ta.tema(df_1h['close'], length=200)
        if df_1h['tema_200'].dropna().empty:
            df_1h['tema_200'] = calculate_native_tema(df_1h['close'], 200)
    except Exception:
        df_1h['tema_200'] = calculate_native_tema(df_1h['close'], 200)

    try:
        df_4h['tema_200'] = ta.tema(df_4h['close'], length=200)
        if df_4h['tema_200'].dropna().empty:
            df_4h['tema_200'] = calculate_native_tema(df_4h['close'], 200)
    except Exception:
        df_4h['tema_200'] = calculate_native_tema(df_4h['close'], 200)

    # Merge 4H TEMA into 1H DataFrame via forward fill
    df_4h_sub = df_4h[['timestamp', 'tema_200']].rename(columns={'tema_200': 'tema_200_4h'})
    df = pd.merge_asof(df_1h.sort_values('timestamp'), df_4h_sub.sort_values('timestamp'), on='timestamp', direction='backward')

    trades = []
    current_balance = initial_balance
    equity_curve = [initial_balance]
    equity_timestamps = [df['timestamp'].iloc[200]]

    active_trade = None
    last_trigger_index = -cooldown_candles - 1

    for i in range(200, len(df)):
        candle = df.iloc[i]
        
        # If trade is active, evaluate exit conditions on current candle
        if active_trade:
            # Check Stop Loss
            if candle['low'] <= active_trade['stop_loss']:
                pnl = -active_trade['risk_usd']
                current_balance += pnl
                trades.append({
                    'Entry_Time': active_trade['entry_time'],
                    'Exit_Time': candle['timestamp'],
                    'Symbol': symbol,
                    'Type': active_trade['type'],
                    'Entry_Price': active_trade['entry_price'],
                    'Exit_Price': active_trade['stop_loss'],
                    'PnL_USD': pnl,
                    'Return_R': -1.0,
                    'Outcome': 'STOP_LOSS'
                })
                active_trade = None
            # Check Target Exits
            elif candle['high'] >= active_trade['tp1']:
                gain_r = abs(active_trade['tp1'] - active_trade['entry_price']) / abs(active_trade['entry_price'] - active_trade['stop_loss'])
                pnl = active_trade['risk_usd'] * gain_r
                current_balance += pnl
                trades.append({
                    'Entry_Time': active_trade['entry_time'],
                    'Exit_Time': candle['timestamp'],
                    'Symbol': symbol,
                    'Type': active_trade['type'],
                    'Entry_Price': active_trade['entry_price'],
                    'Exit_Price': active_trade['tp1'],
                    'PnL_USD': pnl,
                    'Return_R': gain_r,
                    'Outcome': 'TAKE_PROFIT'
                })
                active_trade = None

        equity_curve.append(current_balance)
        equity_timestamps.append(candle['timestamp'])

        # Skip trigger evaluation if a trade is open or on cooldown
        if active_trade is not None or (i - last_trigger_index) < cooldown_candles:
            continue

        # Evaluate Confluence Proximity Trigger
        price = candle['close']
        tema_1h = candle['tema_200']
        tema_4h = candle['tema_200_4h']

        triggered = False
        reasons = []

        if pd.notna(tema_1h):
            dist_1h = abs(price - tema_1h) / tema_1h * 100
            if dist_1h <= proximity_threshold:
                triggered = True
                reasons.append("1H TEMA")

        if pd.notna(tema_4h):
            dist_4h = abs(price - tema_4h) / tema_4h * 100
            if dist_4h <= proximity_threshold:
                triggered = True
                reasons.append("4H TEMA")

        if triggered:
            last_trigger_index = i
            entry = price
            sl = entry * (1 - (sl_pct / 100))
            tp1 = entry * (1 + ((sl_pct / 100) * tp1_mult))
            risk_usd = current_balance * (risk_pct / 100)

            active_trade = {
                'entry_time': candle['timestamp'],
                'type': 'BUY',
                'entry_price': entry,
                'stop_loss': sl,
                'tp1': tp1,
                'risk_usd': risk_usd,
                'reasons': ", ".join(reasons)
            }

    df_trades = pd.DataFrame(trades)
    df_equity = pd.DataFrame({'Timestamp': equity_timestamps, 'Equity': equity_curve})
    return df_trades, df_equity

# --- SIDEBAR: DYNAMIC CONTROL CENTER ---
st.sidebar.title("🛡️ Agent Control Panel")
st.sidebar.markdown("---")

st.sidebar.header("💰 Risk & Portfolio")
account_balance = st.sidebar.number_input(
    "Account Balance ($)", 
    value=float(config.get("account_balance", 1000.0)), 
    step=50.0,
    min_value=10.0
)

risk_pct = st.sidebar.number_input(
    "Risk Per Trade (%)", 
    value=float(config.get("risk_pct", 1.0)), 
    step=0.25,
    min_value=0.1,
    max_value=5.0
)

st.sidebar.subheader("Regime Filters")
min_adx = st.sidebar.slider(
    "Min ADX (Trend Strength)", 
    min_value=10, 
    max_value=40, 
    value=int(config.get("min_adx", 20)), 
    step=1
)
min_atr_pct = st.sidebar.number_input(
    "Min ATR % (Volatility)", 
    value=float(config.get("min_atr_pct", 0.5)), 
    step=0.1
)

st.sidebar.header("🎯 Trigger Rules")
proximity_threshold = st.sidebar.slider(
    "Proximity Threshold (%)", 
    min_value=0.1, 
    max_value=3.0, 
    value=float(config.get("proximity_threshold_pct", 1.0)),
    step=0.05
)

cooldown_hours = st.sidebar.number_input(
    "Alert Cooldown (Hours)", 
    value=int(config.get("alert_cooldown_hours", 4)), 
    step=1,
    min_value=1
)

scan_interval = st.sidebar.number_input(
    "Scan Interval (Minutes)", 
    value=int(config.get("scan_interval_minutes", 15)), 
    step=1,
    min_value=1
)

# --- MONITORED ASSETS ---
st.sidebar.header("📋 Monitored Assets")

popular_assets = [
    "ONDO/USDT", "PENDLE/USDT", "LINK/USDT", "TIA/USDT", "NEAR/USDT",
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "SUI/USDT", "APT/USDT", "XRP/USDT"
]

current_watchlist = config.get("watchlist", [])
all_options = sorted(list(set(popular_assets + current_watchlist)))

# Add Custom Symbol Box
custom_symbol = st.sidebar.text_input("Add Custom Symbol (e.g. XRP/USDT):")
if st.sidebar.button("➕ Add Ticker") and custom_symbol:
    formatted_symbol = custom_symbol.strip().upper()
    if formatted_symbol not in current_watchlist:
        current_watchlist.append(formatted_symbol)
        config["watchlist"] = current_watchlist
        save_config(config)
        st.sidebar.success(f"Added {formatted_symbol}")
        st.rerun()

# Active Watchlist Selector
selected_watchlist = st.sidebar.multiselect(
    "Active Watchlist",
    options=all_options,
    default=current_watchlist
)

st.sidebar.markdown("---")

# --- SAVE BUTTON ---
if st.sidebar.button("💾 Save Settings to Agent", type="primary", use_container_width=True):
    config["account_balance"] = account_balance
    config["risk_pct"] = risk_pct
    config["min_adx"] = float(min_adx)
    config["min_atr_pct"] = float(min_atr_pct)
    config["proximity_threshold_pct"] = proximity_threshold
    config["alert_cooldown_hours"] = cooldown_hours
    config["scan_interval_minutes"] = scan_interval
    config["watchlist"] = selected_watchlist
    
    save_config(config)
    st.sidebar.success("✅ `config.json` updated successfully!")

# --- MAIN DASHBOARD VIEW ---
st.title("📊 Multi-Asset Confluence Dashboard")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Account Balance", f"${account_balance:,.2f}")
col2.metric("Max Risk / Trade", f"${(account_balance * (risk_pct / 100)):,.2f} ({risk_pct}%)")
col3.metric("Monitored Assets", f"{len(selected_watchlist)} Pairs")
col4.metric("Proximity Filter", f"±{proximity_threshold}%")

st.markdown("---")

# --- TAB LAYOUT ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📓 Trade Journal", 
    "📈 Live Interactive Charts", 
    "🧪 Strategy Backtester",
    "⚙️ Raw Config JSON"
])

# TAB 1: TRADE JOURNAL
with tab1:
    st.subheader("Logged Signal Entries")
    journal_path = config.get("journal_file", JOURNAL_FILE)
    
    if os.path.exists(journal_path):
        df_journal = pd.read_csv(journal_path)
        
        if not df_journal.empty:
            st.dataframe(
                df_journal.sort_index(ascending=False), 
                use_container_width=True,
                height=350
            )
            
            csv_data = df_journal.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Journal CSV",
                data=csv_data,
                file_name="trade_journal_export.csv",
                mime="text/csv"
            )
        else:
            st.info("The trade journal file is currently empty.")
    else:
        st.warning(f"No journal file found at `{journal_path}` yet. Run `agent.py` to generate signals.")

# TAB 2: TRADINGVIEW CHARTS
with tab2:
    st.subheader("Interactive Price Analysis")
    
    col_sym, col_tf, col_ex = st.columns([2, 1, 1])
    
    with col_sym:
        active_assets = config.get("watchlist", ["LINK/USDT"])
        chart_symbol = st.selectbox("Select Asset to Chart", active_assets)
        
    with col_tf:
        timeframe_map = {"15 Minutes": "15", "1 Hour": "60", "4 Hours": "240", "1 Day": "D"}
        selected_tf_label = st.selectbox("Timeframe", list(timeframe_map.keys()), index=1)
        selected_tf = timeframe_map[selected_tf_label]
        
    with col_ex:
        exchange_prefix = st.selectbox("Exchange Source", ["MEXC", "BINANCE", "GATEIO", "BYBIT"], index=0)

    raw_ticker = chart_symbol.replace("/", "").replace(":", "")
    tv_symbol = f"{exchange_prefix}:{raw_ticker}"

    tradingview_html = f"""
    <div class="tradingview-widget-container" style="height:620px;width:100%;">
      <div id="tradingview_chart_element" style="height:calc(100% - 32px);width:100%;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "autosize": true,
        "symbol": "{tv_symbol}",
        "interval": "{selected_tf}",
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "container_id": "tradingview_chart_element"
      }});
      </script>
    </div>
    """
    components.html(tradingview_html, height=640)

# TAB 3: HISTORICAL BACKTESTING ENGINE
with tab3:
    st.subheader("🧪 Historical Confluence Strategy Simulation")
    st.markdown("Test the accuracy of the **1H/4H 200 TEMA Proximity Strategy** over historical candle history.")

    col_bt1, col_bt2, col_bt3, col_bt4 = st.columns(4)
    with col_bt1:
        bt_symbol = st.selectbox("Backtest Ticker", config.get("watchlist", ["LINK/USDT"]))
    with col_bt2:
        candle_limit = st.select_slider("Candle History Size (Hours)", options=[500, 1000, 2000], value=1000)
    with col_bt3:
        sl_pct_param = st.number_input("Stop Loss Distance (%)", value=2.5, step=0.5, min_value=1.0)
    with col_bt4:
        tp_mult_param = st.number_input("Target 1 Risk Multiplier (R)", value=1.5, step=0.5, min_value=1.0)

    if st.button("🚀 Run Strategy Backtest", type="primary"):
        with st.spinner(f"Fetching {candle_limit} candles and simulating trades for {bt_symbol}..."):
            df_1h_data = fetch_historical_ohlcv(bt_symbol, timeframe='1h', limit=candle_limit)
            df_4h_data = fetch_historical_ohlcv(bt_symbol, timeframe='4h', limit=int(candle_limit / 2))

            if df_1h_data is not None and df_4h_data is not None:
                df_trades, df_equity = run_backtest_simulation(
                    symbol=bt_symbol,
                    df_1h=df_1h_data,
                    df_4h=df_4h_data,
                    initial_balance=account_balance,
                    risk_pct=risk_pct,
                    proximity_threshold=proximity_threshold,
                    sl_pct=sl_pct_param,
                    tp1_mult=tp_mult_param,
                    tp2_mult=2.5,
                    cooldown_candles=cooldown_hours
                )

                st.markdown("---")
                st.subheader("📈 Backtest Results & Strategy Performance")

                if not df_trades.empty:
                    total_trades = len(df_trades)
                    winning_trades = len(df_trades[df_trades['Outcome'] == 'TAKE_PROFIT'])
                    losing_trades = len(df_trades[df_trades['Outcome'] == 'STOP_LOSS'])
                    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

                    gross_profit = df_trades[df_trades['PnL_USD'] > 0]['PnL_USD'].sum()
                    gross_loss = abs(df_trades[df_trades['PnL_USD'] < 0]['PnL_USD'].sum())
                    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else gross_profit

                    net_pnl = df_trades['PnL_USD'].sum()
                    return_pct = (net_pnl / account_balance) * 100

                    # Metrics Row
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Total Executed Trades", total_trades)
                    m2.metric("Win Rate (%)", f"{win_rate:.1f}%")
                    m3.metric("Profit Factor", f"{profit_factor:.2f}")
                    m4.metric("Net Profit ($)", f"${net_pnl:,.2f}")
                    m5.metric("Total Return (%)", f"{return_pct:+.2f}%")

                    # Equity Curve Chart
                    fig_equity = go.Figure()
                    fig_equity.add_trace(go.Scatter(
                        x=df_equity['Timestamp'],
                        y=df_equity['Equity'],
                        mode='lines',
                        name='Portfolio Value ($)',
                        line=dict(color='#00CC96', width=2)
                    ))
                    fig_equity.update_layout(
                        title=f"Portfolio Growth Curve ({bt_symbol})",
                        template="plotly_dark",
                        height=400,
                        xaxis_title="Date",
                        yaxis_title="Account Balance ($)"
                    )
                    st.plotly_chart(fig_equity, use_container_width=True)

                    # Simulated Log Table
                    st.subheader("Simulated Trade Journal Logs")
                    st.dataframe(df_trades.sort_values(by='Entry_Time', ascending=False), use_container_width=True)

                else:
                    st.info("No proximity signals were triggered during this historical timeframe given the current parameters. Try increasing the proximity threshold slider or candle count.")

# TAB 4: RAW CONFIG JSON
with tab4:
    st.subheader("Active `config.json` Contents")
    st.json(config)