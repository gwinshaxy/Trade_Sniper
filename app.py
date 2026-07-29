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
# MARKET DATA & TECHNICAL INDICATOR CALCULATIONS
# =====================================================================
def calculate_tema_fallback(series: pd.Series, length: int = 200) -> pd.Series:
    """Pure Pandas fallback calculation for TEMA 200."""
    ema1 = series.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    ema3 = ema2.ewm(span=length, adjust=False).mean()
    return 3 * ema1 - 3 * ema2 + ema3

def calculate_volume_profile(df: pd.DataFrame, num_bins: int = 24):
    """Calculates price volume histogram and average volume threshold line."""
    if df.empty or 'volume' not in df.columns:
        return [], 0.0

    p_min = df['low'].min()
    p_max = df['high'].max()
    
    if p_min == p_max or pd.isna(p_min) or pd.isna(p_max):
        return [], 0.0

    bins = np.linspace(p_min, p_max, num_bins + 1)
    bin_volumes = np.zeros(num_bins)

    for _, row in df.iterrows():
        c_low, c_high, vol = row['low'], row['high'], row['volume']
        if pd.isna(vol) or vol <= 0 or c_high == c_low:
            continue
        
        mask = (bins[:-1] <= c_high) & (bins[1:] >= c_low)
        overlapping_bins = np.where(mask)[0]
        if len(overlapping_bins) > 0:
            vol_per_bin = vol / len(overlapping_bins)
            for b_idx in overlapping_bins:
                bin_volumes[b_idx] += vol_per_bin

    max_vol = float(np.max(bin_volumes)) if len(bin_volumes) > 0 else 1.0
    avg_vol = float(np.mean(bin_volumes)) if len(bin_volumes) > 0 else 0.0

    vp_data = []
    for i in range(num_bins):
        vp_data.append({
            'price_low': float(bins[i]),
            'price_high': float(bins[i+1]),
            'price_mid': float((bins[i] + bins[i+1]) / 2),
            'volume': float(bin_volumes[i]),
            'vol_ratio': float(bin_volumes[i] / max_vol) if max_vol > 0 else 0.0
        })

    return vp_data, avg_vol

@st.cache_data(ttl=60)
def fetch_market_data(symbol: str, timeframe: str = "1h", limit: int = 1000) -> pd.DataFrame:
    """Fetches OHLCV market data (1000 bars for TEMA warm-up) and calculates indicators."""
    ohlcv = None
    
    try:
        exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception:
        try:
            exchange = ccxt.gate({'enableRateLimit': True})
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception as e:
            st.error(f"Error fetching chart data for {symbol}: {e}")
            return pd.DataFrame()

    if not ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    df['time'] = (df['timestamp'] / 1000).astype(int)
    
    # 1. TEMA 200 Calculation
    try:
        df['tema_200'] = ta.tema(df['close'], length=200)
        if df['tema_200'].dropna().empty:
            df['tema_200'] = calculate_tema_fallback(df['close'], length=200)
    except Exception:
        df['tema_200'] = calculate_tema_fallback(df['close'], length=200)
        
    df['tema_200'] = pd.to_numeric(df['tema_200'], errors='coerce')

    # 2. ADX (14) Calculation
    try:
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx_df is not None and not adx_df.empty:
            df['adx'] = pd.to_numeric(adx_df['ADX_14'], errors='coerce')
        else:
            df['adx'] = 0.0
    except Exception:
        df['adx'] = 0.0

    # 3. ATR % Calculation
    try:
        raw_atr = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr'] = pd.to_numeric(raw_atr, errors='coerce')
        df['atr_pct'] = (df['atr'] / df['close']) * 100
    except Exception:
        df['atr_pct'] = 0.0

    return df

# =====================================================================
# TRADINGVIEW LIGHTWEIGHT CHARTS HTML RENDERER WITH FULLSCREEN CONTROL
# =====================================================================
def render_tradingview_chart(df: pd.DataFrame, symbol: str, df_journal: pd.DataFrame):
    """Generates chart with Candlesticks, 200 TEMA, Aggregated Trade Levels, Volume Profile, and Fullscreen Mode."""
    
    candles_records = []
    for _, row in df.iterrows():
        candles_records.append({
            'time': int(row['time']),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close'])
        })
    
    tema_records = []
    if 'tema_200' in df.columns:
        valid_tema = df[['time', 'tema_200']].dropna().copy()
        valid_tema['tema_200'] = pd.to_numeric(valid_tema['tema_200'], errors='coerce')
        valid_tema = valid_tema.dropna().drop_duplicates(subset=['time']).sort_values('time')
        
        for _, r in valid_tema.iterrows():
            val = float(r['tema_200'])
            if np.isfinite(val):
                tema_records.append({
                    'time': int(r['time']),
                    'value': round(val, 6)
                })

    vp_bins, avg_vol = calculate_volume_profile(df, num_bins=28)
    max_vol = max([b['volume'] for b in vp_bins]) if vp_bins else 1.0

    # --- AGGREGATE TRADE LEVELS BY ZONE (2 DECIMALS) TO COLLAPSE AXIS BADGES ---
    price_lines_js = ""
    if not df_journal.empty and 'symbol' in df_journal.columns:
        symbol_trades = df_journal[
            (df_journal['symbol'] == symbol) & 
            (df_journal['status'].isin(['OPEN', 'TP1_HIT', 'PENDING']))
        ]
        
        entries = {}
        stop_losses = {}
        tp1_levels = {}
        tp2_levels = {}

        for _, trade in symbol_trades.iterrows():
            t_id = str(trade.get('id', ''))
            
            p_entry = trade.get('entry_price')
            if pd.notnull(p_entry) and float(p_entry) > 0:
                zone_key = round(float(p_entry), 2)
                entries.setdefault(zone_key, []).append((float(p_entry), t_id))

            p_sl = trade.get('stop_loss')
            if pd.notnull(p_sl) and float(p_sl) > 0:
                zone_key = round(float(p_sl), 2)
                stop_losses.setdefault(zone_key, []).append((float(p_sl), t_id))

            p_tp1 = trade.get('take_profit_1')
            if pd.notnull(p_tp1) and float(p_tp1) > 0:
                zone_key = round(float(p_tp1), 2)
                tp1_levels.setdefault(zone_key, []).append((float(p_tp1), t_id))

            p_tp2 = trade.get('take_profit_2')
            if pd.notnull(p_tp2) and float(p_tp2) > 0:
                zone_key = round(float(p_tp2), 2)
                tp2_levels.setdefault(zone_key, []).append((float(p_tp2), t_id))

        # Helper to output a single consolidated price line per zone
        def build_line(data_dict, color, style, prefix):
            js_out = ""
            for _, item_list in data_dict.items():
                avg_price = sum(x[0] for x in item_list) / len(item_list)
                count = len(item_list)
                label = f"{prefix} (#{item_list[0][1]})" if count == 1 else f"{prefix} ({count} Trades)"
                js_out += f"""
                candlestickSeries.createPriceLine({{
                    price: {avg_price:.4f},
                    color: '{color}',
                    lineWidth: 2,
                    lineStyle: LightweightCharts.LineStyle.{style},
                    axisLabelVisible: true,
                    title: '{label}',
                }});
                """
            return js_out

        price_lines_js += build_line(entries, '#2962FF', 'Dashed', 'ENTRY')
        price_lines_js += build_line(stop_losses, '#FF5252', 'Solid', 'SL')
        price_lines_js += build_line(tp1_levels, '#00E676', 'Dotted', 'TP1')
        price_lines_js += build_line(tp2_levels, '#00B0FF', 'Dotted', 'TP2')

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
        <style>
            html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; background-color: #131722; font-family: monospace; overflow: hidden; }}
            #chart-container {{ position: relative; width: 100%; height: 550px; background-color: #131722; }}
            #chart-container.fullscreen {{ position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: 99999; }}
            #chart {{ width: 100%; height: 100%; }}
            #vp-canvas {{ position: absolute; top: 0; left: 0; pointer-events: none; z-index: 2; }}
            #fullscreen-btn {{
                position: absolute; top: 10px; right: 80px; z-index: 100;
                background-color: rgba(42, 46, 57, 0.85); color: #d1d4dc;
                border: 1px solid rgba(197, 203, 206, 0.4); border-radius: 4px;
                padding: 4px 10px; font-size: 11px; cursor: pointer; transition: all 0.2s;
            }}
            #fullscreen-btn:hover {{ background-color: #2962FF; color: #ffffff; border-color: #2962FF; }}
            #error-overlay {{ color: #ff5252; padding: 20px; font-size: 14px; white-space: pre-wrap; display: none; }}
        </style>
    </head>
    <body>
        <div id="chart-container">
            <button id="fullscreen-btn" onclick="toggleFullscreen()">⛶ Fullscreen</button>
            <div id="chart"></div>
            <canvas id="vp-canvas"></canvas>
        </div>
        <div id="error-overlay"></div>
        <script>
            try {{
                const chartContainer = document.getElementById('chart-container');
                const chartElement = document.getElementById('chart');
                const vpCanvas = document.getElementById('vp-canvas');
                const fsBtn = document.getElementById('fullscreen-btn');
                const ctx = vpCanvas.getContext('2d');

                const chart = LightweightCharts.createChart(chartElement, {{
                    width: chartContainer.clientWidth || 800,
                    height: chartContainer.clientHeight || 550,
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

                // 1. Candlestick Series
                const candlestickSeries = chart.addCandlestickSeries({{
                    upColor: '#26a69a', downColor: '#ef5350', borderVisible: false,
                    wickUpColor: '#26a69a', wickDownColor: '#ef5350',
                }});
                const candleData = {json.dumps(candles_records)};
                candlestickSeries.setData(candleData);

                // 2. 200 TEMA Line Series
                const temaData = {json.dumps(tema_records)};
                if (temaData && temaData.length > 0) {{
                    const temaSeries = chart.addLineSeries({{
                        color: '#FF9800',
                        lineWidth: 2,
                        crosshairMarkerVisible: true,
                        priceLineVisible: false,
                        title: '200 TEMA',
                    }});
                    temaSeries.setData(temaData);
                }}

                // 3. Consolidated Trade Price Lines
                {price_lines_js}

                chart.timeScale().fitContent();

                // 4. Volume Profile Canvas Overlay
                const vpBins = {json.dumps(vp_bins)};
                const maxVol = {max_vol};
                const avgVol = {avg_vol};

                function drawVolumeProfile() {{
                    vpCanvas.width = chartContainer.clientWidth;
                    vpCanvas.height = chartContainer.clientHeight;
                    ctx.clearRect(0, 0, vpCanvas.width, vpCanvas.height);

                    if (!vpBins || vpBins.length === 0 || maxVol <= 0) return;

                    const chartWidth = vpCanvas.width - 65;
                    const maxBarWidth = chartWidth * 0.20;

                    ctx.fillStyle = 'rgba(41, 98, 255, 0.25)';
                    ctx.strokeStyle = 'rgba(41, 98, 255, 0.5)';

                    let avgLinePoints = [];

                    vpBins.forEach(bin => {{
                        const yTop = candlestickSeries.priceToCoordinate(bin.price_high);
                        const yBottom = candlestickSeries.priceToCoordinate(bin.price_low);
                        
                        if (yTop !== null && yBottom !== null && !isNaN(yTop) && !isNaN(yBottom)) {{
                            const barHeight = Math.max(Math.abs(yBottom - yTop) - 1, 1);
                            const barWidth = (bin.volume / maxVol) * maxBarWidth;
                            const yPos = Math.min(yTop, yBottom);

                            ctx.fillRect(0, yPos, barWidth, barHeight);
                            ctx.strokeRect(0, yPos, barWidth, barHeight);

                            const yMid = (yTop + yBottom) / 2;
                            const avgX = (avgVol / maxVol) * maxBarWidth;
                            avgLinePoints.push({{ x: avgX, y: yMid }});
                        }}
                    }});

                    if (avgLinePoints.length > 1) {{
                        const avgX = (avgVol / maxVol) * maxBarWidth;
                        ctx.beginPath();
                        ctx.setLineDash([4, 4]);
                        ctx.strokeStyle = '#FFEB3B';
                        ctx.lineWidth = 2;
                        
                        const yMin = Math.min(...avgLinePoints.map(p => p.y));
                        const yMax = Math.max(...avgLinePoints.map(p => p.y));

                        ctx.moveTo(avgX, yMin);
                        ctx.lineTo(avgX, yMax);
                        ctx.stroke();
                        ctx.setLineDash([]);

                        ctx.fillStyle = '#FFEB3B';
                        ctx.font = '10px monospace';
                        ctx.fillText('AVG VOL', avgX + 4, yMin + 12);
                    }}
                }}

                // --- FULLSCREEN TOGGLE FUNCTIONALITY ---
                window.toggleFullscreen = function() {{
                    const isFullscreen = chartContainer.classList.toggle('fullscreen');
                    if (isFullscreen) {{
                        fsBtn.innerText = "✕ Exit Fullscreen";
                        chart.applyOptions({{
                            width: window.innerWidth,
                            height: window.innerHeight
                        }});
                    }} else {{
                        fsBtn.innerText = "⛶ Fullscreen";
                        chart.applyOptions({{
                            width: chartContainer.parentElement.clientWidth || 800,
                            height: 550
                        }});
                    }}
                    drawVolumeProfile();
                }};

                document.addEventListener('keydown', (e) => {{
                    if (e.key === 'Escape' && chartContainer.classList.contains('fullscreen')) {{
                        toggleFullscreen();
                    }}
                }});

                chart.timeScale().subscribeVisibleLogicalRangeChange(drawVolumeProfile);
                chart.timeScale().subscribeVisibleTimeRangeChange(drawVolumeProfile);
                
                window.addEventListener('resize', () => {{
                    if (chartContainer.classList.contains('fullscreen')) {{
                        chart.applyOptions({{ width: window.innerWidth, height: window.innerHeight }});
                    }} else {{
                        chart.applyOptions({{ width: chartContainer.clientWidth, height: 550 }});
                    }}
                    drawVolumeProfile();
                }});

                setTimeout(drawVolumeProfile, 150);

            }} catch (err) {{
                document.getElementById('chart-container').style.display = 'none';
                const errDiv = document.getElementById('error-overlay');
                errDiv.style.display = 'block';
                errDiv.innerText = "JS Render Exception: " + err.message + "\\n" + err.stack;
            }}
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=570)

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
# MAIN DASHBOARD INTERFACE
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

# --- MAIN TABBED INTERFACE ---
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

# TAB 3: TRADINGVIEW CHART & LIVE MARKET METRICS
with tab_charts:
    st.subheader("Interactive TradingView Canvas")
    
    col_sym, col_tf = st.columns([2, 1])
    with col_sym:
        selected_symbol = st.selectbox("Select Asset", config.get("watchlist", ["NEAR/USDT"]))
    with col_tf:
        selected_tf = st.selectbox("Timeframe", ["15m", "1h", "4h"], index=1)
        
    df_chart = fetch_market_data(selected_symbol, timeframe=selected_tf)
    
    if not df_chart.empty:
        latest = df_chart.iloc[-1]
        cur_price = latest['close']
        cur_tema = latest.get('tema_200', np.nan)
        cur_adx = latest.get('adx', 0.0)
        cur_atr_pct = latest.get('atr_pct', 0.0)
        
        if pd.notnull(cur_tema) and float(cur_tema) > 0:
            proximity_pct = abs(cur_price - float(cur_tema)) / float(cur_tema) * 100
            proximity_str = f"{proximity_pct:.2f}%"
            bias_str = "ABOVE TEMA 📈" if cur_price >= float(cur_tema) else "BELOW TEMA 📉"
            tema_display = f"${float(cur_tema):.4f}"
        else:
            proximity_str = "N/A"
            bias_str = "Calculating..."
            tema_display = "N/A"

        stat_c1, stat_c2, stat_c3, stat_c4, stat_c5 = st.columns(5)
        stat_c1.metric("Current Price", f"${cur_price:.4f}")
        stat_c2.metric("200 TEMA Price", tema_display)
        stat_c3.metric("TEMA Proximity", proximity_str, delta=bias_str)
        stat_c4.metric("ADX (14)", f"{float(cur_adx):.1f}", delta="Strong Trend" if float(cur_adx) >= config.get("min_adx", 20) else "Weak/Ranging")
        stat_c5.metric("ATR %", f"{float(cur_atr_pct):.2f}%", delta="Volatility")

        st.divider()

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