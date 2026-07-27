import os
import json
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

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
            return df
        except Exception as e:
            st.error(f"Error loading trade journal: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

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

watchlist_str = st.sidebar.text_area(
    "Watchlist (Comma Separated)", 
    value=", ".join(config.get("watchlist", []))
)

if st.sidebar.button("💾 Save Settings", width="stretch"):
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
# METRICS & LIVE TRADE JOURNAL VIEW
# =====================================================================
tab1, tab2, tab3 = st.tabs(["📊 Live Journal & Signals", "📈 Charting", "🧪 Backtest Summary"])

with tab1:
    st.subheader("📋 Structural Trade Journal")
    journal_df = load_journal()

    if not journal_df.empty:
        # Display key summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Signals", len(journal_df))
        col2.metric("Open Signals", len(journal_df[journal_df["Status"] == "OPEN"]) if "Status" in journal_df.columns else 0)
        col3.metric("Last Active Pair", journal_df.iloc[-1]["Symbol"] if "Symbol" in journal_df.columns else "N/A")
        col4.metric("Risk Model", f"{risk_pct}% / Trade")

        st.markdown("---")
        
        # Updated dataframe render without deprecated container width
        st.dataframe(
            journal_df.sort_index(ascending=False), 
            width="stretch"
        )
    else:
        st.info("No trades logged in `trade_journal.csv` yet. Waiting for structural proximity signals.")

with tab2:
    st.subheader("📈 TradingView Live Charting")
    selected_symbol = st.selectbox("Select Pair to Chart", config.get("watchlist", ["NEAR/USDT"]))
    
    # Format symbol for TradingView Widget (e.g. NEAR/USDT -> MEXC:NEARUSDT)
    tv_symbol = f"MEXC:{selected_symbol.replace('/', '')}"
    
    tradingview_html = f"""
    <!-- TradingView Widget BEGIN -->
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
    <!-- TradingView Widget END -->
    """
    
    # Safe modern iframe container render
    components.html(tradingview_html, height=560, scrolling=False)

with tab3:
    st.subheader("🧪 Backtest Performance Analytics")
    st.write("Current Regime Filters: **TEMA 200 + ADX (>=18) + ATR% (>=0.4%)**")
    
    if not journal_df.empty and "Status" in journal_df.columns:
        st.write("### Recorded Execution History")
        st.dataframe(journal_df, width="stretch")
    else:
        st.info("Historical backtest data will populate here as trade states close.")