import os
import json
import gc
import ccxt
import pandas as pd
import numpy as np
import pandas_ta as ta
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

# =====================================================================
# PAGE SETUP & STYLING
# =====================================================================
st.set_page_config(
    page_title="Trade Sniper Dashboard Pro",
    page_icon="🎯",
    layout="wide"
)

load_dotenv()

AVAILABLE_PAIRS = [
    "ONDO/USDT", "PENDLE/USDT", "LINK/USDT", "TIA/USDT", "NEAR/USDT",
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "SUI/USDT",
    "AR/USDT", "FET/USDT", "RENDER/USDT", "TAO/USDT", "SYRUP/USDT",
    "AAVE/USDT", "UNI/USDT", "APT/USDT", "INJ/USDT", "SEI/USDT"
]

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
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
        return df
    except Exception as e:
        st.error(f"Error loading trade journal: {e}")
        return pd.DataFrame()
    finally:
        gc.collect()

def clear_remote_journal():
    if supabase:
        try:
            supabase.table("trade_journal").delete().gt("id", 0).execute()
            st.toast("🧹 Trade journal database cleared!", icon="✅")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to clear database: {e}")

# =====================================================================
# MARKET DATA & INDICATOR CALCULATIONS
# =====================================================================
def calculate_tema_fallback(series: pd.Series, length: int = 200) -> pd.Series:
    """Fallback TEMA calculation using Pandas EWM."""
    ema1 = series.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    ema3 = ema2.ewm(span=length, adjust=False).mean()
    res = 3 * ema1 - 3 * ema2 + ema3
    del ema1, ema2, ema3
    return res

def calculate_volume_profile(df: pd.DataFrame, num_bins: int = 24):
    """Calculates price volume histogram and average volume threshold line."""
    if df.empty or 'volume' not in df.columns:
        return [], 0.0

    p_min = float(df['low'].min())
    p_max = float(df['high'].max())
    
    if p_min == p_max or pd.isna(p_min) or pd.isna(p_max):
        return [], 0.0

    bins = np.linspace(p_min, p_max, num_bins + 1, dtype=np.float32)
    bin_volumes = np.zeros(num_bins, dtype=np.float32)

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

    del bins, bin_volumes
    return vp_data, avg_vol

@st.cache_data(ttl=120, max_entries=5)
def fetch_market_data(symbol: str, timeframe: str = "1h", limit: int = 350, htf_period: int = 200) -> pd.DataFrame:
    """Fetches OHLCV market data and calculates 1H & 4H HTF TEMA indicators."""
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
    del ohlcv
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
        
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    df['time'] = (df['timestamp'] / 1000).astype(int)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

    # 1. Base 1H TEMA 200 Calculation
    try:
        df['tema_200'] = ta.tema(df['close'], length=200)
        if df['tema_200'].dropna().empty:
            df['tema_200'] = calculate_tema_fallback(df['close'], length=200)
    except Exception:
        df['tema_200'] = calculate_tema_fallback(df['close'], length=200)

    # 2. Upgrade A: Resample and Calculate 4H HTF TEMA
    try:
        df_indexed = df.set_index('datetime')
        df_4h = df_indexed.resample('4h').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna().reset_index()
        
        try:
            df_4h['tema_htf'] = ta.tema(df_4h['close'], length=htf_period)
            if df_4h['tema_htf'].dropna().empty:
                df_4h['tema_htf'] = calculate_tema_fallback(df_4h['close'], length=htf_period)
        except Exception:
            df_4h['tema_htf'] = calculate_tema_fallback(df_4h['close'], length=htf_period)
            
        df = pd.merge_asof(
            df.sort_values('datetime'), 
            df_4h[['datetime', 'tema_htf']].sort_values('datetime'), 
            on='datetime', 
            direction='backward'
        )
        df['tema_htf'] = df['tema_htf'].ffill()
        df['tema_htf_slope'] = df['tema_htf'] - df['tema_htf'].shift(4)
    except Exception:
        df['tema_htf'] = df['tema_200']
        df['tema_htf_slope'] = 0.0

    # 3. ADX (14) Calculation
    try:
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx_df is not None and not adx_df.empty:
            df['adx'] = pd.to_numeric(adx_df['ADX_14'], errors='coerce').astype(np.float32)
            del adx_df
        else:
            df['adx'] = np.float32(0.0)
    except Exception:
        df['adx'] = np.float32(0.0)

    # 4. ATR Calculation
    try:
        raw_atr = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr'] = pd.to_numeric(raw_atr, errors='coerce').astype(np.float32)
        df['atr_pct'] = ((df['atr'] / df['close']) * 100).astype(np.float32)
        del raw_atr
    except Exception:
        df['atr'] = np.float32(0.0)
        df['atr_pct'] = np.float32(0.0)

    gc.collect()
    return df

# =====================================================================
# UPGRADED HISTORICAL BACKTESTING ENGINE (UPGRADES A, B, & C)
# =====================================================================
def run_backtest_upgraded(df: pd.DataFrame, params: dict):
    if df.empty or len(df) < 200:
        return pd.DataFrame(), pd.Series(), {}

    initial_balance = params['initial_balance']
    base_risk_pct = params['risk_pct'] / 100.0
    target_rr = params['target_rr']
    min_adx = params['min_adx']
    min_atr_pct = params['min_atr_pct']
    proximity_pct = params['proximity_pct']
    atr_mult_sl = params.get('atr_mult_sl', 1.5)
    
    # Upgrades Toggles & Config
    use_mtf = params.get('use_mtf', True)
    htf_mode = params.get('htf_mode', 'Price Level')
    use_atr_trail = params.get('use_atr_trail', True)
    atr_trail_mult = params.get('atr_trail_mult', 1.5)
    use_equity_filter = params.get('use_equity_filter', True)
    eq_ma_period = params.get('eq_ma_period', 10)
    reduced_risk_factor = params.get('reduced_risk_factor', 0.5)

    balance = float(initial_balance)
    equity_curve = [balance]
    trades = []
    active_trade = None

    for i in range(200, len(df)):
        row = df.iloc[i]
        price = float(row['close'])
        high = float(row['high'])
        low = float(row['low'])
        tema_1h = float(row['tema_200']) if pd.notnull(row['tema_200']) else 0.0
        tema_htf = float(row['tema_htf']) if pd.notnull(row['tema_htf']) else tema_1h
        htf_slope = float(row.get('tema_htf_slope', 0.0))
        adx = float(row['adx'])
        atr_pct = float(row['atr_pct'])
        atr_val = float(row['atr'])
        timestamp = datetime.fromtimestamp(int(row['time'])).strftime('%Y-%m-%d %H:%M')

        # 1. PROCESS ACTIVE TRADE & UPGRADE B (Adaptive ATR Trailing Stop)
        if active_trade is not None:
            side = active_trade["side"]
            sl = active_trade["sl"]
            tp = active_trade["tp"]
            
            hit_sl = low <= sl if side == "LONG" else high >= sl
            hit_tp = high >= tp if side == "LONG" else low <= tp

            if hit_sl or hit_tp:
                exit_price = sl if hit_sl else tp
                pnl_r = -1.0 if hit_sl else target_rr
                pnl_usd = -active_trade["risk_usd"] if hit_sl else (active_trade["risk_usd"] * target_rr)
                balance += pnl_usd

                trades.append({
                    "Entry Time": active_trade["entry_time"],
                    "Exit Time": timestamp,
                    "Side": side,
                    "Entry ($)": round(active_trade["entry"], 4),
                    "Exit ($)": round(exit_price, 4),
                    "Result": "WIN" if pnl_usd > 0 else "LOSS",
                    "PnL ($)": round(pnl_usd, 2),
                    "Realized R": f"{pnl_r:.2f}R",
                    "Balance ($)": round(balance, 2)
                })
                active_trade = None

            elif use_atr_trail and atr_val > 0:
                if side == "LONG":
                    new_sl = price - (atr_val * atr_trail_mult)
                    active_trade["sl"] = max(active_trade["sl"], new_sl)
                else:
                    new_sl = price + (atr_val * atr_trail_mult)
                    active_trade["sl"] = min(active_trade["sl"], new_sl)

        # 2. SCAN FOR SIGNALS & APPLY UPGRADE A & UPGRADE C
        if active_trade is None and tema_1h > 0:
            if adx >= min_adx and atr_pct >= min_atr_pct:
                prox = abs(price - tema_1h) / tema_1h * 100
                if prox <= proximity_pct:
                    
                    base_long = price >= tema_1h
                    base_short = price < tema_1h
                    direction = None

                    # Upgrade A: Multi-Timeframe TEMA Alignment
                    if use_mtf:
                        if htf_mode == "Price Level":
                            htf_long = price >= tema_htf
                            htf_short = price < tema_htf
                        else:  # Slope Mode
                            htf_long = htf_slope > 0
                            htf_short = htf_slope < 0

                        if base_long and htf_long:
                            direction = "LONG"
                        elif base_short and htf_short:
                            direction = "SHORT"
                    else:
                        direction = "LONG" if base_long else "SHORT"

                    if direction:
                        # Upgrade C: Equity Curve Drawdown Risk Filter
                        effective_risk = base_risk_pct
                        if use_equity_filter and len(equity_curve) >= eq_ma_period:
                            recent_eq_ma = np.mean(equity_curve[-eq_ma_period:])
                            if balance < recent_eq_ma:
                                effective_risk *= reduced_risk_factor

                        risk_usd = balance * effective_risk
                        
                        if direction == "LONG":
                            sl = price - (atr_mult_sl * atr_val)
                            tp = price + (atr_mult_sl * atr_val * target_rr)
                        else:
                            sl = price + (atr_mult_sl * atr_val)
                            tp = price - (atr_mult_sl * atr_val * target_rr)

                        active_trade = {
                            "entry_time": timestamp,
                            "side": direction,
                            "entry": price,
                            "sl": sl,
                            "tp": tp,
                            "risk_usd": risk_usd
                        }

        equity_curve.append(balance)

    trades_df = pd.DataFrame(trades)
    wins = len(trades_df[trades_df['Result'] == 'WIN']) if not trades_df.empty else 0
    total_trades = len(trades_df)
    
    metrics = {
        "Starting Balance": f"${initial_balance:.2f}",
        "Final Balance": f"${balance:.2f}",
        "Total Trades": total_trades,
        "Win Rate": f"{(wins / total_trades * 100):.1f}%" if total_trades > 0 else "0.0%",
        "Net Return": f"{((balance - initial_balance) / initial_balance * 100):.2f}%"
    }

    gc.collect()
    return trades_df, pd.Series(equity_curve), metrics

# =====================================================================
# TRADINGVIEW LIGHTWEIGHT CHARTS HTML RENDERER
# =====================================================================
def render_tradingview_chart(df: pd.DataFrame, symbol: str, df_journal: pd.DataFrame):
    candles_records = df[['time', 'open', 'high', 'low', 'close']].to_dict(orient='records')
    
    tema_records = []
    if 'tema_200' in df.columns:
        valid_tema = df[['time', 'tema_200']].dropna().copy()
        for _, r in valid_tema.iterrows():
            val = float(r['tema_200'])
            if np.isfinite(val):
                tema_records.append({'time': int(r['time']), 'value': round(val, 6)})
        del valid_tema

    vp_bins, avg_vol = calculate_volume_profile(df, num_bins=24)
    max_vol = max([b['volume'] for b in vp_bins]) if vp_bins else 1.0

    price_lines_js = ""
    if not df_journal.empty and 'symbol' in df_journal.columns:
        symbol_trades = df_journal[
            (df_journal['symbol'] == symbol) & 
            (df_journal['status'].isin(['OPEN', 'TP1_HIT', 'PENDING']))
        ]
        
        entries, stop_losses, tp1_levels, tp2_levels = {}, {}, {}, {}

        for _, trade in symbol_trades.iterrows():
            t_id = str(trade.get('id', ''))
            
            p_entry = trade.get('entry_price')
            if pd.notnull(p_entry) and float(p_entry) > 0:
                entries.setdefault(round(float(p_entry), 2), []).append((float(p_entry), t_id))

            p_sl = trade.get('stop_loss')
            if pd.notnull(p_sl) and float(p_sl) > 0:
                stop_losses.setdefault(round(float(p_sl), 2), []).append((float(p_sl), t_id))

            p_tp1 = trade.get('take_profit_1')
            if pd.notnull(p_tp1) and float(p_tp1) > 0:
                tp1_levels.setdefault(round(float(p_tp1), 2), []).append((float(p_tp1), t_id))

            p_tp2 = trade.get('take_profit_2')
            if pd.notnull(p_tp2) and float(p_tp2) > 0:
                tp2_levels.setdefault(round(float(p_tp2), 2), []).append((float(p_tp2), t_id))

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

                const candlestickSeries = chart.addCandlestickSeries({{
                    upColor: '#26a69a', downColor: '#ef5350', borderVisible: false,
                    wickUpColor: '#26a69a', wickDownColor: '#ef5350',
                }});
                const candleData = {json.dumps(candles_records)};
                candlestickSeries.setData(candleData);

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

                {price_lines_js}

                chart.timeScale().fitContent();

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

                window.toggleFullscreen = function() {{
                    const isFullscreen = chartContainer.classList.toggle('fullscreen');
                    if (isFullscreen) {{
                        fsBtn.innerText = "✕ Exit Fullscreen";
                        chart.applyOptions({{ width: window.innerWidth, height: window.innerHeight }});
                    }} else {{
                        fsBtn.innerText = "⛶ Fullscreen";
                        chart.applyOptions({{ width: chartContainer.parentElement.clientWidth || 800, height: 550 }});
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
    del candles_records, tema_records, vp_bins
    gc.collect()

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
        "atr_mult_sl": 1.5,
        "min_rr_ratio": 1.5,
        "alert_cooldown_hours": 4,
        "scan_interval_minutes": 15,
        "use_mtf_alignment": True,
        "htf_tema_period": 200,
        "htf_alignment_mode": "Price Level",
        "use_atr_trailing": True,
        "atr_trail_multiplier": 1.5,
        "use_equity_filter": True,
        "eq_ma_period": 10,
        "reduced_risk_factor": 0.5,
        "watchlist": ["ONDO/USDT", "PENDLE/USDT", "LINK/USDT", "TIA/USDT", "NEAR/USDT"]
    }
    if not os.path.exists(CONFIG_FILE):
        return default_config
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return {**default_config, **json.load(f)}
    except Exception:
        return default_config

def save_config(config_data):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4)
        st.toast("Settings saved!", icon="💾")
    except Exception as e:
        st.error(f"Failed to save settings: {e}")

# =====================================================================
# MAIN DASHBOARD INTERFACE
# =====================================================================
st.title("🎯 Trade Sniper Dashboard Pro")
st.caption("Automated Multi-Timeframe TEMA Alignment, Dynamic ATR Trailing, & Equity Curve Risk Management")

config = load_config()
df_journal = load_trade_journal()

# --- SIDEBAR: CONFIG & STRATEGY UPGRADES CONTROL ---
with st.sidebar:
    st.header("⚙️ Bot Parameters")
    
    with st.form("config_form"):
        account_balance = st.number_input(
            "Account Balance ($)", 
            value=float(config.get("account_balance", 1000.0)), 
            step=50.0
        )
        risk_pct = st.slider(
            "Base Account Risk % per Trade", 
            0.1, 5.0, 
            float(config.get("risk_pct", 1.0)), 
            step=0.1
        )
        proximity_thresh = st.slider(
            "Proximity Threshold % (TEMA/POC)", 
            0.5, 5.0, 
            float(config.get("proximity_threshold_pct", 2.0)), 
            step=0.1
        )
        min_adx = st.slider(
            "ADX Momentum Cutoff", 
            10.0, 50.0, 
            float(config.get("min_adx", 20.0)), 
            step=1.0
        )
        min_atr = st.number_input(
            "Min ATR % Filter", 
            value=float(config.get("min_atr_pct", 0.4)), 
            step=0.05
        )
        atr_mult_sl = st.slider(
            "ATR Multiplier (Stop Loss)", 
            0.5, 4.0, 
            float(config.get("atr_mult_sl", 1.5)), 
            step=0.1
        )
        min_rr_ratio = st.slider(
            "Reward-to-Risk Ratio (R)", 
            1.0, 5.0, 
            float(config.get("min_rr_ratio", 1.5)), 
            step=0.1
        )

        st.markdown("---")
        st.subheader("🚀 Upgrades Control Panel")

        use_mtf = st.checkbox("Enable Upgrade A: 1H + HTF TEMA Trend Alignment", value=config.get("use_mtf_alignment", True))
        htf_period = st.slider("HTF TEMA Lookback Period", 10, 200, int(config.get("htf_tema_period", 200)), step=10)
        htf_mode = st.radio("HTF Alignment Mode", ["Price Level", "TEMA Slope"], index=0 if config.get("htf_alignment_mode", "Price Level") == "Price Level" else 1)

        st.markdown("---")
        use_atr_trail = st.checkbox("Enable Upgrade B: Dynamic Adaptive ATR Trailing Stop", value=config.get("use_atr_trailing", True))
        atr_trail_mult = st.slider("ATR Trailing Multiplier", 1.0, 4.0, float(config.get("atr_trail_multiplier", 1.5)), step=0.1)

        st.markdown("---")
        use_equity_filter = st.checkbox("Enable Upgrade C: Equity Curve Drawdown Filter", value=config.get("use_equity_filter", True))
        eq_ma = st.number_input("Equity MA Lookback (Trades)", value=int(config.get("eq_ma_period", 10)), step=1)
        reduced_risk = st.slider("Drawdown Risk Multiplier", 0.1, 0.9, float(config.get("reduced_risk_factor", 0.5)), step=0.05)

        st.markdown("---")
        alert_cooldown = st.number_input("Alert Cooldown (hours)", value=int(config.get("alert_cooldown_hours", 4)), min_value=1, max_value=48, step=1)
        scan_interval = st.number_input("Scan Interval (mins)", value=int(config.get("scan_interval_minutes", 15)), step=1)
        
        current_watchlist = config.get("watchlist", [])
        all_options = list(dict.fromkeys(AVAILABLE_PAIRS + current_watchlist))
        
        selected_watchlist = st.multiselect("Select Watchlist Assets:", options=all_options, default=current_watchlist)
        custom_asset = st.text_input("Add Custom Asset (e.g. SOL/USDT):", "").strip().upper()
        
        submitted = st.form_submit_button("Save Configuration")
        if submitted:
            final_watchlist = list(selected_watchlist)
            if custom_asset and custom_asset not in final_watchlist:
                final_watchlist.append(custom_asset)

            updated_config = {
                "account_balance": account_balance,
                "risk_pct": risk_pct,
                "proximity_threshold_pct": proximity_thresh,
                "min_adx": min_adx,
                "min_atr_pct": min_atr,
                "atr_mult_sl": atr_mult_sl,
                "min_rr_ratio": min_rr_ratio,
                "use_mtf_alignment": use_mtf,
                "htf_tema_period": htf_period,
                "htf_alignment_mode": htf_mode,
                "use_atr_trailing": use_atr_trail,
                "atr_trail_multiplier": atr_trail_mult,
                "use_equity_filter": use_equity_filter,
                "eq_ma_period": eq_ma,
                "reduced_risk_factor": reduced_risk,
                "alert_cooldown_hours": alert_cooldown,
                "scan_interval_minutes": scan_interval,
                "watchlist": final_watchlist
            }
            save_config(updated_config)
            st.cache_data.clear()
            gc.collect()
            st.rerun()

    st.divider()
    if st.button("Clear Dashboard RAM & Refresh", use_container_width=True):
        st.cache_data.clear()
        gc.collect()
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
    del closed_trades
else:
    m1.metric("Total Signals", "0")
    m2.metric("Active Trades", "0")
    m3.metric("Net Realized PnL", "$0.00")
    m4.metric("Win Rate", "0.0%")
    m5.metric("Database", "Supabase (Live)", delta="Connected")

st.divider()

# --- MAIN TABBED INTERFACE ---
tab_active, tab_history, tab_charts, tab_backtest, tab_database = st.tabs([
    "🔥 Active Trades", 
    "📜 Closed History", 
    "📈 TradingView Chart", 
    "🧪 Historical Backtest",
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
            del active_df
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
            del closed_df
        else:
            st.info("No closed trades recorded yet.")
    else:
        st.info("No trade data found in database.")

# TAB 3: TRADINGVIEW CHART & LIVE MARKET METRICS
with tab_charts:
    st.subheader("Interactive TradingView Canvas")
    
    col_sym, col_tf = st.columns([2, 1])
    with col_sym:
        watchlist_options = config.get("watchlist", ["NEAR/USDT"])
        selected_symbol = st.selectbox("Select Asset", watchlist_options if watchlist_options else ["NEAR/USDT"])
    with col_tf:
        selected_tf = st.selectbox("Timeframe", ["15m", "1h", "4h"], index=1)
        
    df_chart = fetch_market_data(
        selected_symbol, 
        timeframe=selected_tf, 
        limit=350, 
        htf_period=int(config.get("htf_tema_period", 200))
    )
    
    if not df_chart.empty:
        latest = df_chart.iloc[-1]
        cur_price = float(latest['close'])
        cur_tema = float(latest.get('tema_200', np.nan))
        cur_htf = float(latest.get('tema_htf', np.nan))
        cur_adx = float(latest.get('adx', 0.0))
        cur_atr_pct = float(latest.get('atr_pct', 0.0))
        
        if pd.notnull(cur_tema) and cur_tema > 0:
            proximity_pct = abs(cur_price - cur_tema) / cur_tema * 100
            proximity_str = f"{proximity_pct:.2f}%"
            bias_str = "ABOVE 1H TEMA 📈" if cur_price >= cur_tema else "BELOW 1H TEMA 📉"
            tema_display = f"${cur_tema:.4f}"
        else:
            proximity_str = "N/A"
            bias_str = "Calculating..."
            tema_display = "N/A"

        htf_display = f"${cur_htf:.4f}" if pd.notnull(cur_htf) else "N/A"

        stat_c1, stat_c2, stat_c3, stat_c4, stat_c5 = st.columns(5)
        stat_c1.metric("Current Price", f"${cur_price:.4f}")
        stat_c2.metric("1H TEMA (200)", tema_display)
        stat_c3.metric("4H HTF TEMA", htf_display, delta=f"1H Bias: {bias_str}")
        stat_c4.metric("ADX (14)", f"{cur_adx:.1f}", delta="Strong Trend" if cur_adx >= config.get("min_adx", 20) else "Ranging")
        stat_c5.metric("ATR %", f"{cur_atr_pct:.2f}%", delta="Volatility")

        st.divider()

        render_tradingview_chart(df_chart, selected_symbol, df_journal)
        del df_chart
        gc.collect()
    else:
        st.warning("Could not fetch market data for the selected symbol.")

# TAB 4: HISTORICAL BACKTEST WITH ALL UPGRADES INTEGRATED
with tab_backtest:
    st.subheader("🧪 Upgraded Historical Backtesting Engine")
    st.caption("Simulate active strategy rules with Multi-Timeframe Alignment, Dynamic ATR Trailing, & Drawdown Risk Scaling.")
    
    b_col1, b_col2, b_col3 = st.columns([2, 1, 1])
    with b_col1:
        bt_symbol = st.selectbox("Backtest Asset", config.get("watchlist", ["NEAR/USDT"]), key="bt_sym")
    with b_col2:
        bt_tf = st.selectbox("Timeframe", ["15m", "1h", "4h"], index=1, key="bt_tf")
    with b_col3:
        bt_limit = st.select_slider("Historical Bars Limit", options=[300, 350, 500, 1000], value=350)
        
    if st.button("🚀 Run Strategy Backtest", type="primary", use_container_width=True):
        with st.spinner("Fetching historical data and computing multi-timeframe strategy rules..."):
            df_bt = fetch_market_data(
                bt_symbol, 
                timeframe=bt_tf, 
                limit=bt_limit, 
                htf_period=int(config.get("htf_tema_period", 200))
            )
            
            if not df_bt.empty:
                params = {
                    'initial_balance': config.get("account_balance", 1000.0),
                    'risk_pct': config.get("risk_pct", 1.0),
                    'target_rr': config.get("min_rr_ratio", 1.5),
                    'min_adx': config.get("min_adx", 20.0),
                    'min_atr_pct': config.get("min_atr_pct", 0.4),
                    'proximity_pct': config.get("proximity_threshold_pct", 2.0),
                    'atr_mult_sl': config.get("atr_mult_sl", 1.5),
                    'use_mtf': config.get("use_mtf_alignment", True),
                    'htf_mode': config.get("htf_alignment_mode", "Price Level"),
                    'use_atr_trail': config.get("use_atr_trailing", True),
                    'atr_trail_mult': config.get("atr_trail_multiplier", 1.5),
                    'use_equity_filter': config.get("use_equity_filter", True),
                    'eq_ma_period': int(config.get("eq_ma_period", 10)),
                    'reduced_risk_factor': config.get("reduced_risk_factor", 0.5)
                }

                sim_trades, equity_series, sim_metrics = run_backtest_upgraded(df_bt, params)
                
                m_c1, m_c2, m_c3, m_c4, m_c5 = st.columns(5)
                m_c1.metric("Starting Balance", sim_metrics["Starting Balance"])
                m_c2.metric("Final Balance", sim_metrics["Final Balance"])
                m_c3.metric("Simulated Trades", sim_metrics["Total Trades"])
                m_c4.metric("Win Rate", sim_metrics["Win Rate"])
                m_c5.metric("Net Return", sim_metrics["Net Return"])
                
                st.divider()

                tab_sub1, tab_sub2 = st.tabs(["📈 Equity Curve", "📜 Simulated Trade Log"])
                with tab_sub1:
                    st.line_chart(equity_series, use_container_width=True)
                with tab_sub2:
                    if not sim_trades.empty:
                        st.dataframe(sim_trades, use_container_width=True, hide_index=True)
                    else:
                        st.info("No trades were triggered during this historical period with the active parameters.")
                
                del df_bt, sim_trades, equity_series, sim_metrics
                gc.collect()
            else:
                st.error("Failed to load historical data for backtesting.")

# TAB 5: DATABASE OPERATIONS
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

# End of Script Memory Cleanup
gc.collect()