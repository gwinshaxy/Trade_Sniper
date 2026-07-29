import os
import json
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

# Page Config
st.set_page_config(
    page_title="Trade Sniper Dashboard",
    page_icon="🎯",
    layout="wide"
)

load_dotenv()

# =====================================================================
# SUPABASE CONNECTION & DATA PIPELINE
# =====================================================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

@st.cache_resource
def get_supabase_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.error("⚠️ Supabase credentials missing! Set SUPABASE_URL and SUPABASE_KEY in environment secrets.")
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.error(f"Failed to connect to Supabase: {e}")
        return None

supabase = get_supabase_client()

def load_trade_journal() -> pd.DataFrame:
    """Fetches all trade entries from Supabase PostgreSQL database."""
    if not supabase:
        return pd.DataFrame()
    
    try:
        response = supabase.table("trade_journal").select("*").order("id", desc=True).execute()
        data = response.data
        
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        
        # Column mapping & type formatting
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
        st.error(f"Error fetching trade journal: {e}")
        return pd.DataFrame()

def clear_remote_journal():
    """Deletes all logged trades from Supabase table."""
    if supabase:
        try:
            supabase.table("trade_journal").delete().gt("id", 0).execute()
            st.toast("🧹 Trade journal database cleared successfully!", icon="✅")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to clear database: {e}")

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
        st.toast("Settings saved successfully!", icon="💾")
    except Exception as e:
        st.error(f"Failed to save config: {e}")

# =====================================================================
# STREAMLIT UI LAYOUT
# =====================================================================
st.title("🎯 Trade Sniper Dashboard")
st.caption("Live Structural Trade Monitor & Strategy Control Center")

config = load_config()
df_journal = load_trade_journal()

# --- SIDEBAR: BOT CONFIGURATION ---
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
    if st.button("Refresh Dashboard Data", use_container_width=True):
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

# --- TABS SECTION ---
tab_active, tab_history, tab_database = st.tabs(["🔥 Active Trades", "📜 Closed History", "🛠️ Database Operations"])

# TAB 1: ACTIVE TRADES
with tab_active:
    st.subheader("Currently Open & Partial Targets")
    if not df_journal.empty:
        active_df = df_journal[df_journal['status'].isin(['OPEN', 'TP1_HIT'])].copy()
        if not active_df.empty:
            display_cols = ['id', 'timestamp', 'symbol', 'status', 'entry_price', 'stop_loss', 'take_profit_1', 'take_profit_2', 'position_usdt', 'max_risk_usd', 'trigger_reason']
            st.dataframe(active_df[display_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No active signals currently open.")
    else:
        st.info("No trade data found.")

# TAB 2: CLOSED HISTORY
with tab_history:
    st.subheader("Closed Trade Performance")
    if not df_journal.empty:
        closed_df = df_journal[df_journal['status'].isin(['CLOSED_TP2', 'STOPPED_OUT'])].copy()
        if not closed_df.empty:
            display_cols = ['id', 'timestamp', 'closed_timestamp', 'symbol', 'status', 'entry_price', 'exit_price', 'realized_pnl_usd', 'realized_r']
            
            # Format PnL coloring in Streamlit table
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
        st.info("No trade data found.")

# TAB 3: DATABASE MANAGEMENT
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