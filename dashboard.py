import os
import sys
import json
import logging
import gc
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv(override=True)

# Explicitly assign and enforce the Cloudflare Proxy Worker URL to route API requests
os.environ["CLOUDFLARE_WORKER_URL"] = "https://workers.dev"
CLOUDFLARE_WORKER_URL = os.environ["CLOUDFLARE_WORKER_URL"]

query_params = st.query_params
if "assetlinks" in query_params or st.context.headers.get("Path") == "/.well-known/assetlinks.json":
    try:
        with open(".well-known/assetlinks.json") as f:
            st.json(json.load(f))
    except Exception as e:
        st.error(f"Error reading assetlinks.json: {e}")
    st.stop()

from lightweight_charts.widgets import StreamlitChart

st.set_page_config(page_title="Bybit Futures 4H Breakout Terminal", layout="wide")

pwa_manifest = """
{
  "short_name": "TradeSniper",
  "name": "Trade Sniper 4H Breakout Terminal",
  "icons": [
    {
      "src": "https://githubusercontent.com",
      "sizes": "192x192",
      "type": "image/png",
      "purpose": "any maskable"
    },
    {
      "src": "https://githubusercontent.com",
      "sizes": "512x512",
      "type": "image/png"
    }
  ],
  "start_url": "/",
  "background_color": "#0e1117",
  "theme_color": "#0e1117",
  "display": "standalone",
  "orientation": "portrait"
}
"""

pwa_injector = f"""
<script>
  const manifestBlob = new Blob([{repr(pwa_manifest)}], {{type: 'application/json'}});
  const manifestUrl = URL.createObjectURL(manifestBlob);
  let manifestLink = document.createElement('link');
  manifestLink.rel = 'manifest';
  manifestLink.href = manifestUrl;
  window.parent.document.head.appendChild(manifestLink);

  let iconLink = document.createElement('link');
  iconLink.rel = 'apple-touch-icon';
  iconLink.href = 'https://githubusercontent.com';
  window.parent.document.head.appendChild(iconLink);

  let metaTheme = document.createElement('meta');
  metaTheme.name = 'theme-color';
  metaTheme.content = '#0e1117';
  window.parent.document.head.appendChild(metaTheme);

  if ('serviceWorker' in navigator) {{
    const swCode = `
      self.addEventListener('install', (e) => self.skipWaiting());
      self.addEventListener('activate', (e) => e.waitUntil(clients.claim()));
      self.addEventListener('fetch', (e) => {{
        if (e.request.url.includes('_stcore') || e.request.url.includes('stream')) {{
          return fetch(e.request);
        }}
        e.respondWith(fetch(e.request).catch(() => caches.match(e.request)));
      }});
    `;
    const swBlob = new Blob([swCode], {{type: 'application/javascript'}});
    const swUrl = URL.createObjectURL(swBlob);
    window.navigator.serviceWorker.register(swUrl);
  }}
</script>
"""

components.html(pwa_injector, height=0, width=0)

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
    normalize_symbol,
    logger
)
from live_executor import LiveExecutionEngine

try:
    from config import STRATEGY_CONFIG as _STRAT_CFG
except ImportError:
    _STRAT_CFG = {}

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

default_watchlist = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT",
    "DOGE/USDT", "BNB/USDT", "LINK/USDT", "NEAR/USDT", "HYPE/USDT"
]
cfg_watchlist = _STRAT_CFG.get("watchlist", []) if isinstance(_STRAT_CFG, dict) else []
symbols_env = os.getenv("TRADING_SYMBOLS") or os.getenv("WATCHLIST") or os.getenv("SYMBOLS") or ""
env_symbols = [s.strip() for s in symbols_env.split(",") if s.strip()]

database_url = os.getenv("DATABASE_URL") or os.getenv("DB_URL")
if database_url:
    database_url = database_url.strip('"').strip("'")


@st.cache_resource
def run_startup_db_maintenance():
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("UPDATE trade_setups SET pair = REPLACE(REPLACE(pair::text, '\"', ''), '''', '') WHERE pair IS NOT NULL;")
                conn.commit()
        except Exception as db_err:
            logger.warning(f"Failed to normalize DB pairs on startup: {db_err}")
        finally:
            release_db_connection(conn)


run_startup_db_maintenance()


@st.cache_data(ttl=5, max_entries=20)
def fetch_trades_from_db(db_url: str) -> pd.DataFrame:
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    try:
        df = pd.read_sql_query("SELECT * FROM trade_setups ORDER BY id DESC;", conn)
        return df
    except Exception as db_err:
        logger.error(f"Error fetching trade records: {db_err}")
        return pd.DataFrame()
    finally:
        release_db_connection(conn)


df_all_trades = fetch_trades_from_db(database_url)

db_pairs = [p.strip() for p in df_all_trades['pair'].unique().tolist()] if not df_all_trades.empty and 'pair' in df_all_trades.columns else []
raw_combined = cfg_watchlist + env_symbols + default_watchlist + db_pairs
seen_norm = set()
available_pairs = []
for sym in raw_combined:
    norm = normalize_symbol(sym)
    if norm and norm not in seen_norm:
        seen_norm.add(norm)
        formatted = f"{norm[:-4]}/USDT" if norm.endswith("USDT") and "/" not in sym else sym
        available_pairs.append(formatted)

st.sidebar.subheader("🎛️ 4H Breakout Controls")

is_paper_mode = os.getenv("ENABLE_PAPER_TRADING", "false").lower() == "true"
mode_badge = "🧪 KEYLESS PAPER MODE" if is_paper_mode else "🟢 LIVE BYBIT EXECUTION"
st.sidebar.info(f"**Mode:** {mode_badge}")

selected_pair = st.sidebar.selectbox("Active Chart / Execution Pair", available_pairs, index=0)
config_target_pair = selected_pair
pair_key = normalize_symbol(config_target_pair)

dyn_cfg = strategy.load_symbol_config(config_target_pair)

lookback_bars = st.sidebar.number_input("Lookback Bars (Range)", value=300, min_value=100, max_value=1000, step=50, key=f"lookback_{pair_key}")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 4H Breakout Indicator Overlays")
show_donchian = st.sidebar.checkbox("Show 48-Bar Donchian Channel", value=True, key=f"donchian_{pair_key}")
show_200_ema = st.sidebar.checkbox("Show 200 EMA (Macro Trend)", value=True, key=f"ema200_{pair_key}")
show_50_ema = st.sidebar.checkbox("Show 50 EMA (Reference)", value=False, key=f"ema50_{pair_key}")

donchian_period = st.sidebar.number_input(
    "Donchian Channel Period",
    value=int(_STRAT_CFG.get("donchian_period", strategy.DONCHIAN_PERIOD)),
    min_value=10,
    max_value=200,
    key=f"donchian_p_{pair_key}"
)

ema_200_period = st.sidebar.number_input(
    "Macro Trend EMA Period",
    value=int(_STRAT_CFG.get("ema_trend_period", strategy.EMA_TREND_PERIOD)),
    min_value=50,
    max_value=400,
    key=f"ema200_p_{pair_key}"
)

adx_threshold = st.sidebar.slider(
    "Min ADX Regime Threshold",
    min_value=10.0,
    max_value=40.0,
    value=float(_STRAT_CFG.get("min_adx", strategy.MIN_ADX_THRESHOLD)),
    step=1.0,
    key=f"adx_t_{pair_key}"
)

atr_sl_mult = st.sidebar.slider(
    "Initial Stop-Loss ATR Mult",
    min_value=1.0,
    max_value=5.0,
    value=float(dyn_cfg.get("atr_sl_multiplier", strategy.ATR_SL_MULTIPLIER)),
    step=0.1,
    key=f"atr_sl_{pair_key}"
)

atr_trail_mult = st.sidebar.slider(
    "Chandelier Trail ATR Mult",
    min_value=1.0,
    max_value=5.0,
    value=float(dyn_cfg.get("atr_trail_multiplier", strategy.ATR_TRAIL_MULTIPLIER)),
    step=0.1,
    key=f"atr_trail_{pair_key}"
)

trail_activation_r = st.sidebar.slider(
    "Trail Activation (+R)",
    min_value=0.5,
    max_value=3.0,
    value=float(dyn_cfg.get("trail_activation_r", strategy.TRAIL_ACTIVATION_R)),
    step=0.1,
    key=f"trail_act_{pair_key}"
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Manual Order Execution")
direction = st.sidebar.selectbox("Order Direction", ["BUY", "SELL"])

_default_leverage = int(_STRAT_CFG.get("leverage", 5)) if isinstance(_STRAT_CFG, dict) else 5
leverage = st.sidebar.number_input(
    "Leverage",
    min_value=1,
    max_value=50,
    value=_default_leverage,
)

overlay_chart = st.sidebar.checkbox("Overlay Trade Positions on Chart", value=True)

entry_price = st.sidebar.number_input("Reference Entry Price ($)", min_value=0.0, format="%.5f")
stop_loss = st.sidebar.number_input("Initial Stop Loss ($)", min_value=0.0, format="%.5f")
take_profit = st.sidebar.number_input("Take Profit ($) [0 = Chandelier Trail]", min_value=0.0, value=0.0, format="%.5f")

risk = abs(entry_price - stop_loss) if (entry_price and stop_loss) else 0.0
reward = abs(take_profit - entry_price) if (take_profit and entry_price) else 0.0
rr_ratio = round(reward / risk, 2) if (risk > 0 and take_profit > 0) else 0.0

st.sidebar.markdown(f"Exit Mode: {'1:' + str(rr_ratio) + ' Static TP' if take_profit > 0 else f'{atr_trail_mult}x ATR Chandelier (@ +{trail_activation_r}R)'}")
account_balance = st.sidebar.number_input("Account Balance ($)", min_value=1.0, value=float(os.getenv("ACCOUNT_BALANCE", 100.0)))
risk_pct = st.sidebar.number_input("Risk Per Trade (%)", min_value=0.01, max_value=25.0, value=1.0)

if st.sidebar.button("🚀 Execute Manual Futures Order", use_container_width=True):
    with st.spinner("Submitting Futures Order..."):
        try:
            max_pos = int(_STRAT_CFG.get("max_concurrent_positions", 5)) if isinstance(_STRAT_CFG, dict) else 5
            allocated_usd = account_balance / max(max_pos, 1)
            res = executor.order_futures_bybit(
                symbol=config_target_pair,
                direction=direction,
                amount_usd=allocated_usd,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=leverage,
                account_balance=account_balance,
                risk_pct=risk_pct
            )
            if res.get("status") == "SUCCESS":
                fill_price = res["fill_price"]
                executed_qty = res["executed_qty"]
                manual_entry_order_id = res.get("order_id")
                conn_exec = get_db_connection()
                manual_trade_id = None
                if conn_exec:
                    try:
                        with conn_exec.cursor() as cur:
                            cur.execute("""
                                INSERT INTO trade_setups
                                (pair, direction, entry_price, stop_loss, initial_stop_loss, take_profit,
                                highest_price, lowest_price, position_size, status, trade_state,
                                account_balance, risk_pct, updated_at)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'EXECUTED', 'OPEN', %s, %s, NOW())
                                RETURNING id;
                            """, (
                                config_target_pair, direction, fill_price, stop_loss, stop_loss, take_profit,
                                fill_price, fill_price, executed_qty, account_balance, risk_pct
                            ))
                            row = cur.fetchone()
                            manual_trade_id = row[0] if row else None
                            conn_exec.commit()
                        if manual_trade_id and manual_entry_order_id:
                            try:
                                from reconciler import mark_trade_entry_order
                                mark_trade_entry_order(manual_trade_id, manual_entry_order_id)
                            except Exception as reg_err:
                                logger.debug(f"[{config_target_pair}] Entry order cache registration failed: {reg_err}")
                    finally:
                        release_db_connection(conn_exec)
                st.cache_data.clear()
                st.success(f"Futures {direction} trade executed for {config_target_pair} at ${fill_price:.5f}!")
                send_telegram_notification(
                    f"🚀 MANUAL DASHBOARD ORDER EXECUTED ({direction})\n\n"
                    f"Pair: {config_target_pair}\n"
                    f"Entry: ${fill_price:.5f}\n"
                    f"Qty: {executed_qty}\n"
                    f"Initial SL: ${stop_loss:.5f}"
                )
                st.rerun()
            else:
                st.error(f"Futures order execution failed: {res.get('error') or res.get('reason')}")
        except Exception as exec_err:
            st.error(f"Execution Error Encountered: {exec_err}")

st.title("📊 Bybit Futures 4H Donchian Breakout Dashboard")
st.info("Synced with 4H Donchian-48 Breakout + 200 EMA Macro Trend + 3.0x ATR Chandelier Trailing Stop Engine.")

closed_trades = df_all_trades[df_all_trades['status'] == 'CLOSED'] if not df_all_trades.empty and 'status' in df_all_trades.columns else pd.DataFrame()
net_realized_pnl = float(closed_trades['pnl_usd'].sum()) if not closed_trades.empty and 'pnl_usd' in closed_trades.columns else 0.0
profit_factor_val = 0.0
max_dd_pct = 0.0

if not closed_trades.empty and 'pnl_usd' in closed_trades.columns:
    pnl_series = pd.to_numeric(closed_trades['pnl_usd'], errors='coerce').fillna(0.0)
    if 'outcome' in closed_trades.columns:
        wins_sum = float(closed_trades[closed_trades['outcome'] == 'WIN']['pnl_usd'].sum())
        loss_sum = abs(float(closed_trades[closed_trades['outcome'] == 'LOSS']['pnl_usd'].sum()))
        profit_factor_val = round(wins_sum / loss_sum, 2) if loss_sum > 0 else (99.99 if wins_sum > 0 else 0.0)
    if 'closed_at' in closed_trades.columns:
        df_dd = closed_trades.sort_values('closed_at').copy()
        eq_curve = account_balance + pd.to_numeric(df_dd['pnl_usd'], errors='coerce').fillna(0.0).cumsum()
        rolling_peak = eq_curve.cummax()
        dd_pct_series = ((eq_curve - rolling_peak) / rolling_peak.replace(0, np.nan)) * 100.0
        max_dd_pct = abs(float(dd_pct_series.min())) if not dd_pct_series.empty and pd.notnull(dd_pct_series.min()) else 0.0

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Net Realized PnL", f"${net_realized_pnl:,.2f}")
col_m2.metric("Max Drawdown", f"{max_dd_pct:.2f}%")
col_m3.metric("Profit Factor", f"{profit_factor_val:.2f}")
col_m4.metric("Total Closed Trades", f"{len(closed_trades):,.0f}" if not closed_trades.empty else "0")

st.markdown("---")
col_hdr, col_tf = st.columns([3, 1])
with col_hdr:
    st.subheader("📈 Interactive 4H Donchian Breakout & Chandelier Chart")
with col_tf:
    timeframe_options = ["15m", "1h", "4h", "1d"]
    default_tf_index = timeframe_options.index("4h")
    selected_timeframe = st.selectbox("Select Timeframe:", timeframe_options, index=default_tf_index, key="chart_timeframe_select")

chart_symbol = selected_pair
try:
    df_ohlc = strategy.fetch_klines(symbol=chart_symbol, interval=selected_timeframe, limit=min(lookback_bars, 500))
except Exception as fetch_err:
    df_ohlc = None
    st.error(f"API Fetch Error: {fetch_err}")

if df_ohlc is not None and not df_ohlc.empty:
    try:
        df_ohlc = df_ohlc.copy()
        if isinstance(df_ohlc.index, pd.DatetimeIndex):
            df_ohlc = df_ohlc.reset_index()
            df_ohlc.rename(columns={df_ohlc.columns[0]: 'time'}, inplace=True)
        elif 'timestamp' in df_ohlc.columns and 'time' not in df_ohlc.columns:
            df_ohlc.rename(columns={'timestamp': 'time'}, inplace=True)
        df_ohlc = df_ohlc.loc[:, ~df_ohlc.columns.duplicated()].copy()
        df_ohlc['time'] = pd.to_datetime(df_ohlc['time'], errors='coerce')
        df_ohlc = df_ohlc.dropna(subset=['time'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df_ohlc.columns:
                df_ohlc[col] = pd.to_numeric(df_ohlc[col], errors='coerce')
        
        df_ohlc['ema_200'] = strategy.calculate_ema(df_ohlc['close'], period=min(ema_200_period, len(df_ohlc)))
        df_ohlc['ema_50'] = strategy.calculate_ema(df_ohlc['close'], period=50)
        df_ohlc['donchian_high'] = df_ohlc['high'].rolling(donchian_period).max().shift(1)
        df_ohlc['donchian_low'] = df_ohlc['low'].rolling(donchian_period).min().shift(1)
        min_chart_p = float(df_ohlc['low'].min())
        max_chart_p = float(df_ohlc['high'].max())
        df_chart = df_ohlc.copy()
        df_chart['time'] = pd.to_datetime(df_chart['time'])
        if selected_timeframe in ['1d']:
            df_chart['time'] = df_chart['time'].dt.strftime('%Y-%m-%d')
        else:
            df_chart['time'] = df_chart['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_chart = df_chart.drop_duplicates(subset=['time']).sort_values('time', ascending=True).reset_index(drop=True)

        chart = StreamlitChart(width=None, height=680)
        chart.time_scale(time_visible=True, seconds_visible=False)

        chart.layout(background_color='#131722', text_color='#d1d4dc')
        chart.volume_config(scale_margin_top=0.85, scale_margin_bottom=0.0, up_color='#26a69a', down_color='#ef5350')
        ohlcv_data = df_chart[['time', 'open', 'high', 'low', 'close', 'volume']].dropna()
        chart.set(ohlcv_data)

        if show_donchian:
            dc_high_df = df_chart[['time', 'donchian_high']].dropna().rename(columns={'donchian_high': f'DC-{donchian_period} High'})
            dc_low_df = df_chart[['time', 'donchian_low']].dropna().rename(columns={'donchian_low': f'DC-{donchian_period} Low'})
            if not dc_high_df.empty:
                line_dc_h = chart.create_line(name=f'DC-{donchian_period} High', color="#26a69a", width=1)
                line_dc_h.set(dc_high_df)
            if not dc_low_df.empty:
                line_dc_l = chart.create_line(name=f'DC-{donchian_period} Low', color="#ef5350", width=1)
                line_dc_l.set(dc_low_df)

        if show_200_ema:
            ema200_df = df_chart[['time', 'ema_200']].dropna().rename(columns={'ema_200': f'{ema_200_period} EMA'})
            if not ema200_df.empty:
                line_200 = chart.create_line(name=f'{ema_200_period} EMA', color="#ab47bc", width=2)
                line_200.set(ema200_df)

        if show_50_ema:
            ema50_df = df_chart[['time', 'ema_50']].dropna().rename(columns={'ema_50': '50 EMA'})
            if not ema50_df.empty:
                line_50 = chart.create_line(name='50 EMA', color="#ff9800", width=1)
                line_50.set(ema50_df)

        if overlay_chart and not df_all_trades.empty and 'pair' in df_all_trades.columns and 'status' in df_all_trades.columns:
            symbol_trades = df_all_trades[
                (df_all_trades['pair'].apply(normalize_symbol) == normalize_symbol(selected_pair)) &
                (df_all_trades['status'].isin(['EXECUTED', 'PENDING']))
            ]
            for _, tr in symbol_trades.iterrows():
                entry_p = float(tr['entry_price']) if pd.notnull(tr['entry_price']) else 0.0
                sl_p = float(tr['stop_loss']) if pd.notnull(tr['stop_loss']) else 0.0
                tp_p = float(tr['take_profit']) if pd.notnull(tr['take_profit']) else 0.0
                t_state = str(tr.get('trade_state') or 'OPEN')
                if min_chart_p * 0.5 <= entry_p <= max_chart_p * 1.5:
                    chart.horizontal_line(entry_p, color="#2962ff", text=f"ENTRY ({tr.get('direction', 'LONG')})")
                if sl_p > 0 and (min_chart_p * 0.5 <= sl_p <= max_chart_p * 1.5):
                    sl_label = f"CHANDELIER SL ({t_state})" if t_state == "TRAILING" else "INITIAL SL"
                    chart.horizontal_line(sl_p, color="#f44336", text=sl_label)
                if tp_p > 0 and (min_chart_p * 0.5 <= tp_p <= max_chart_p * 1.5):
                    chart.horizontal_line(tp_p, color="#4caf50", text="TP")

        chart.load()
    except Exception as chart_err:
        st.error(f"Lightweight Chart Rendering Exception: {chart_err}")
else:
    st.warning(f"Could not load market chart for {chart_symbol}.")

st.markdown("---")
tab_active, tab_history = st.tabs(["⏳ Active Positions & Chandelier Management", "📜 Trade History Log"])

with tab_active:
    df_active = df_all_trades[df_all_trades['status'].isin(['EXECUTED', 'PENDING'])] if not df_all_trades.empty and 'status' in df_all_trades.columns else pd.DataFrame()
    if not df_active.empty:
        df_active = df_active.copy()
        df_active['pair'] = df_active['pair'].apply(normalize_symbol)
        st.dataframe(df_active, use_container_width=True)

        st.markdown("### 🔧 Live Trade Overrides & Modifications")
        selected_trade_id = st.selectbox("Select Trade ID to Manage:", df_active['id'].tolist())
        selected_row = df_active[df_active['id'] == selected_trade_id].iloc[0]

        col_sl, col_tp = st.columns(2)
        with col_sl:
            new_sl = st.number_input(
                "Update Stop Loss ($)",
                value=float(selected_row['stop_loss']) if pd.notnull(selected_row['stop_loss']) else 0.0,
                format="%.5f"
            )
            if st.button("Update SL in DB & Exchange", use_container_width=True):
                executor.set_position_trading_stop(
                    symbol=selected_row['pair'],
                    stop_loss=new_sl,
                    take_profit=0,
                    direction=str(selected_row['direction'])
                )
                conn_mod = get_db_connection()
                if conn_mod:
                    try:
                        cur = conn_mod.cursor()
                        cur.execute(
                            "UPDATE trade_setups SET stop_loss = %s, trailing_stop_price = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s;",
                            (new_sl, new_sl, selected_trade_id)
                        )
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
            new_tp = st.number_input(
                "Update Take Profit ($) [0 = Trailing Only]",
                value=float(selected_row['take_profit']) if pd.notnull(selected_row['take_profit']) else 0.0,
                format="%.5f"
            )
            if st.button("Update TP in Database", use_container_width=True):
                conn_mod = get_db_connection()
                if conn_mod:
                    try:
                        cur = conn_mod.cursor()
                        cur.execute(
                            "UPDATE trade_setups SET take_profit = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s;",
                            (new_tp, selected_trade_id)
                        )
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
        manual_exit = st.number_input(
            "Fallback Exit Price ($) [Used for Market Close/DB Override]",
            value=float(selected_row['entry_price']) if pd.notnull(selected_row['entry_price']) else 0.0,
            format="%.5f"
        )

        col_bybit, col_db_only = st.columns(2)
        with col_bybit:
            if st.button("🚨 Market Close on Bybit Exchange", type="primary", use_container_width=True):
                try:
                    close_res = executor.close_live_position_bybit(
                        symbol=selected_row['pair'],
                        position_size=float(selected_row['position_size']),
                        current_price=manual_exit,
                        outcome="MANUAL_CLOSE"
                    )
                    if close_res.get("status") in ["SUCCESS", "FORCE_CLOSED_DB_ONLY"] and float(close_res.get("exit_price", 0.0)) > 0.0:
                        final_p = float(close_res["exit_price"])
                        if close_trade_manually(selected_trade_id, final_p, reason="STREAMLIT_BYBIT_MARKET_CLOSE"):
                            st.cache_data.clear()
                            st.success(f"Position #{selected_trade_id} closed on Bybit at ${final_p:.5f}.")
                            st.rerun()
                        else:
                            st.error("Position closed on Bybit, but database update failed.")
                    else:
                        st.error(f"🚨 CRITICAL: Bybit close order failed: {close_res.get('error')}. DB was NOT modified.")
                except Exception as close_err:
                    st.error(f"Error attempting Bybit close: {close_err}")

        with col_db_only:
            if st.button("⚠️ Force DB Close Only (No Bybit Order)", use_container_width=True):
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
        closed_trades_display = closed_trades.copy()
        closed_trades_display['pair'] = closed_trades_display['pair'].apply(normalize_symbol)
        st.dataframe(closed_trades_display, use_container_width=True)
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
            df_hm = closed_trades.copy()
            df_hm['closed_at'] = pd.to_datetime(df_hm['closed_at'])
            df_hm['day_of_week'] = df_hm['closed_at'].dt.day_name()
            df_hm['hour'] = df_hm['closed_at'].dt.hour
            df_hm['pnl_usd'] = pd.to_numeric(df_hm['pnl_usd'], errors='coerce').fillna(0.0)
            heatmap_data = df_hm.pivot_table(index='day_of_week', columns='hour', values='pnl_usd', aggfunc='sum').fillna(0)
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
    st.markdown("Active Engine Status:")
    log_messages = [
        "[INFO] Database connection pool active (initial_stop_loss, highest_price, lowest_price synced).",
        "[INFO] Live Execution Engine bound to Bybit Futures API via Cloudflare Worker Proxy.",
        "[INFO] Active Positions tab synchronized with Chandelier Trailing Stop state machine.",
        f"[STATUS] Monitoring 4H Donchian-{donchian_period} Breakout + {ema_200_period} EMA + {atr_trail_mult}x ATR Trail (@ +{trail_activation_r}R)."
    ]
    for log_item in log_messages:
        st.code(log_item, language="text")

gc.collect()