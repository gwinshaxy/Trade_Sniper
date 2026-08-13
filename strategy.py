import json
import logging
import urllib.request
import os
import re
import pandas as pd
import numpy as np
from google import genai
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


def normalize_symbol(symbol: str) -> str:
    if not symbol:
        return ""
    s = str(symbol).replace('"', '').replace("'", "").strip().upper()
    if "/" in s:
        return s
    if s.endswith("USDT") and len(s) > 4:
        return f"{s[:-4]}/{s[-4:]}"
    return s


def load_symbol_config(symbol: str) -> dict:
    """Loads optimized strategy parameters from PostgreSQL database stripping slashes for lookup."""
    clean_symbol = normalize_symbol(symbol).replace("/", "").upper()
    db_url = os.getenv("DATABASE_URL")
    
    if db_url:
        db_url = db_url.strip('"').strip("'").strip()
    
    if PSYCOPG2_AVAILABLE and db_url:
        try:
            conn = psycopg2.connect(db_url)
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT * 
                    FROM strategy_parameters 
                    WHERE UPPER(TRIM(REPLACE(REPLACE(symbol, '"', ''), '''', ''))) = %s
                    ORDER BY updated_at DESC LIMIT 1;
                    """, 
                    (clean_symbol,)
                )
                row = cur.fetchone()
                conn.close()
                if row:
                    config = dict(row)
                    config.setdefault("rsi_thresh", 42.0)
                    config.setdefault("use_rsi_filter", True)
                    config.setdefault("use_candlestick_confirm", True)
                    config.setdefault("use_adx_filter", True)
                    config.setdefault("adx_period", 14)
                    config.setdefault("adx_threshold", 20.0)
                    config.setdefault("max_sl_pct", 0.02)
                    logging.info(f"[load_symbol_config] Loaded dynamic DEAP params for {clean_symbol}")
                    return config
                else:
                    logging.warning(f"[load_symbol_config] Parameter row missing for {clean_symbol}, using defaults.")
        except Exception as e:
            logging.error(f"[load_symbol_config] DB query failed for {clean_symbol}: {e}")

    return {
        "tema_period": 200, "rsi_period": 14, "rsi_thresh": 42.0,
        "adx_period": 14, "adx_threshold": 20.0, "use_adx_filter": True,
        "zone_tolerance": 0.0075, "min_sentiment": 0.0, "risk_pct": 1.0,
        "min_rr": 2.0, "vp_detection_pct": 0.07, "use_rsi_filter": True,
        "use_candlestick_confirm": True, "max_sl_pct": 0.02
    }


def get_ai_sentiment_score(text: str) -> float:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return 0.5
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"Analyze the financial sentiment of the following text and return ONLY a single float number between 0.0 (bearish) and 1.0 (bullish):\n\n'{text}'"
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        
        match = re.search(r"0\.\d+|1\.0|0|1", response.text.strip())
        if match:
            return max(0.0, min(1.0, float(match.group(0))))
        return 0.5
    except Exception:
        return 0.5


def fetch_cryptocompare_klines(symbol: str = "ETH/USDT", limit: int = 1000) -> pd.DataFrame:
    """Fallback: CryptoCompare Historical Hourly Data API."""
    try:
        clean = symbol.replace("/", "").upper()
        fsym = clean[:-4] if clean.endswith("USDT") else clean[:-3]
        tsym = "USDT" if clean.endswith("USDT") else "USD"

        url = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={fsym}&tsym={tsym}&limit={min(limit, 2000)}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode('utf-8'))

        if data.get("Response") == "Success" and "Data" in data and "Data" in data["Data"]:
            candles = data["Data"]["Data"]
            df = pd.DataFrame(candles)
            df.rename(columns={"time": "open_time", "volumeto": "volume"}, inplace=True)
            df['time'] = pd.to_datetime(df['open_time'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
                
            return df[['time', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        logging.debug(f"[CryptoCompare Fallback Error]: {e}")
        
    return pd.DataFrame()


def fetch_klines(symbol: str = "BNB/USDT", interval: str = "1h", limit: int = 1000) -> pd.DataFrame:
    norm_sym = normalize_symbol(symbol)
    binance_symbol = norm_sym.replace("/", "")
    
    # Bybit Interval Mapping
    bybit_interval_map = {"1m": "1", "5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "D"}
    bybit_interval = bybit_interval_map.get(interval, "60")

    # 1. Primary & Exchange Endpoints
    urls = [
        f"https://api.binance.com/api/v3/klines?symbol={binance_symbol}&interval={interval}&limit={min(limit, 1000)}",
        f"https://api.binance.us/api/v3/klines?symbol={binance_symbol}&interval={interval}&limit={min(limit, 1000)}",
        f"https://api.bybit.com/v5/market/kline?category=spot&symbol={binance_symbol}&interval={bybit_interval}&limit={min(limit, 1000)}"
    ]

    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            
            # Binance / Binance US structure
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'not', 'tbba', 'tbqa', 'ignore'])
                df['time'] = pd.to_datetime(df['open_time'], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                return df[['time', 'open', 'high', 'low', 'close', 'volume']]
            
            # Bybit structure
            elif isinstance(data, dict) and 'result' in data and 'list' in data['result']:
                raw_list = data['result']['list']
                df = pd.DataFrame(raw_list, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                df = df.iloc[::-1].reset_index(drop=True)
                df['time'] = pd.to_datetime(df['open_time'].astype(int), unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                return df[['time', 'open', 'high', 'low', 'close', 'volume']]

        except Exception as err:
            logging.debug(f"[fetch_klines] Endpoint bypassed ({url}): {err}")
            continue

    # 2. Third-Party Fallback (CryptoCompare)
    df_fallback = fetch_cryptocompare_klines(symbol=norm_sym, limit=limit)
    if not df_fallback.empty:
        logging.info(f"[fetch_klines] Successfully retrieved fallback data from CryptoCompare for {norm_sym}")
        return df_fallback

    logging.error(f"[fetch_klines] All primary and third-party data endpoints failed for symbol: {symbol}")
    return pd.DataFrame()


def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_tema(series: pd.Series, period: int = 200) -> pd.Series:
    ema1 = calc_ema(series, period)
    ema2 = calc_ema(ema1, period)
    ema3 = calc_ema(ema2, period)
    return (3 * ema1) - (3 * ema2) + ema3


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss.replace(0, np.nan))
    return (100 - (100 / (1 + rs))).fillna(50)


def calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates Average Directional Index (ADX) to quantify trend strength."""
    if df.empty or len(df) < period + 1:
        return pd.Series(0.0, index=df.index)

    df_calc = df.copy()
    df_calc['prev_close'] = df_calc['close'].shift(1)
    df_calc['tr1'] = df_calc['high'] - df_calc['low']
    df_calc['tr2'] = (df_calc['high'] - df_calc['prev_close']).abs()
    df_calc['tr3'] = (df_calc['low'] - df_calc['prev_close']).abs()
    tr = df_calc[['tr1', 'tr2', 'tr3']].max(axis=1)

    up_move = df_calc['high'] - df_calc['high'].shift(1)
    down_move = df_calc['low'].shift(1) - df_calc['low']

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Wilder's Smoothing
    tr_smooth = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / tr_smooth.replace(0, np.nan))
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / tr_smooth.replace(0, np.nan))

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.ewm(alpha=1/period, adjust=False).mean().fillna(0.0)
    return adx


def compute_volume_profile(
    df: pd.DataFrame,
    num_bins: int = 100,
    lookback_bars: int = 600,
    va_pct: float = 0.70,
):
    """Computes POC, VAH, and VAL aligned with TradingView's Fixed Range (600 bars, 70% Value Area)."""
    if (
        df.empty
        or "volume" not in df.columns
        or "high" not in df.columns
        or "low" not in df.columns
    ):
        return np.nan, np.nan, np.nan

    df_range = df.tail(lookback_bars).copy()
    if df_range.empty:
        return np.nan, np.nan, np.nan

    min_p, max_p = float(df_range["low"].min()), float(df_range["high"].max())
    if min_p >= max_p or np.isnan(min_p) or np.isnan(max_p):
        return np.nan, np.nan, np.nan

    bin_size = (max_p - min_p) / num_bins
    if bin_size <= 0:
        return np.nan, np.nan, np.nan

    vol_profile = np.zeros(num_bins)

    for _, row in df_range.iterrows():
        low_val, high_val, vol = (
            float(row["low"]),
            float(row["high"]),
            float(row["volume"]),
        )
        s_idx = max(int(np.floor((low_val - min_p) / bin_size)), 0)
        e_idx = min(int(np.floor((high_val - min_p) / bin_size)), num_bins - 1)

        num_spanned = e_idx - s_idx + 1
        vol_per_bin = vol / num_spanned if num_spanned > 0 else vol

        for idx in range(s_idx, e_idx + 1):
            vol_profile[idx] += vol_per_bin

    poc_idx = int(np.argmax(vol_profile))
    poc = min_p + (poc_idx + 0.5) * bin_size

    target_vol = np.sum(vol_profile) * va_pct
    current_vol = vol_profile[poc_idx]
    la_idx, lb_idx = poc_idx, poc_idx

    max_iter = num_bins * 2
    iter_count = 0

    while current_vol < target_vol and iter_count < max_iter:
        iter_count += 1
        if lb_idx == 0 and la_idx == num_bins - 1:
            break

        up_vol = vol_profile[la_idx + 1] if la_idx < num_bins - 1 else 0.0
        dn_vol = vol_profile[lb_idx - 1] if lb_idx > 0 else 0.0

        if up_vol >= dn_vol:
            current_vol += up_vol
            la_idx += 1
        else:
            current_vol += dn_vol
            lb_idx -= 1

    vah = min_p + (la_idx + 1.0) * bin_size
    val = min_p + (lb_idx + 0.0) * bin_size

    return float(poc), float(vah), float(val)


def calculate_volume_profile_gaps(df: pd.DataFrame, num_bins: int = 100, lookback_bars: int = 600, detection_pct: float = 0.07) -> list:
    """Identifies Volume Profile Gaps (LVNs)."""
    if df.empty or 'volume' not in df.columns or 'high' not in df.columns or 'low' not in df.columns:
        return []

    df_range = df.tail(lookback_bars).copy()
    if df_range.empty:
        return []

    pLST, pHST = float(df_range['low'].min()), float(df_range['high'].max())
    if pLST >= pHST or np.isnan(pLST) or np.isnan(pHST):
        return []

    bin_size = (pHST - pLST) / num_bins
    if bin_size <= 0:
        return []

    vol_profile = np.zeros(num_bins)

    for _, row in df_range.iterrows():
        low_val, high_val, vol = float(row['low']), float(row['high']), float(row['volume'])
        s_idx = max(int(np.floor((low_val - pLST) / bin_size)), 0)
        e_idx = min(int(np.floor((high_val - pLST) / bin_size)), num_bins - 1)
        
        num_spanned = (e_idx - s_idx + 1)
        vol_per_bin = vol / num_spanned if num_spanned > 0 else vol

        for idx in range(s_idx, e_idx + 1):
            vol_profile[idx] += vol_per_bin

    max_vol = np.max(vol_profile)
    if max_vol == 0:
        return []

    noN = max(1, int(num_bins * detection_pct))
    padded_vt = np.pad(vol_profile, (noN, noN), mode='constant', constant_values=max_vol)

    gap_prices = []

    for vn in range(2 * noN, num_bins + 2 * noN):
        uNth = True
        for cVN in range(vn - 2 * noN, vn - noN):
            if padded_vt[vn - noN] >= padded_vt[cVN]:
                uNth = False
                break

        lNth = True
        for cVN in range(vn - noN + 1, vn + 1):
            if padded_vt[vn - noN] >= padded_vt[cVN]:
                lNth = False
                break

        if uNth and lNth:
            bin_idx = vn - 2 * noN
            gap_price = float(round(pLST + (bin_idx + 0.5) * bin_size, 2))
            if pLST <= gap_price <= pHST:
                gap_prices.append(gap_price)

    return list(dict.fromkeys(gap_prices))


def evaluate_signals(df: pd.DataFrame, symbol: str = "ETH/USDT", account_balance: float = 10000.0, **kwargs) -> dict:
    config = load_symbol_config(symbol)
    tema_period = int(kwargs.get("tema_period", config.get("tema_period", 200)))
    rsi_period = int(kwargs.get("rsi_period", config.get("rsi_period", 14)))
    adx_period = int(kwargs.get("adx_period", config.get("adx_period", 14)))

    if df.empty or len(df) < max(tema_period, adx_period + 1):
        return {"action": "HOLD", "reason": "Insufficient historical data for indicator convergence"}

    # Calculate indicators on current timeframe
    df['tema'] = calc_tema(df['close'], period=tema_period)
    df['rsi'] = calc_rsi(df['close'], period=rsi_period)
    df['adx'] = calc_adx(df, period=adx_period)

    latest = df.iloc[-1]
    current_close = float(latest['close'])
    current_tema = float(latest['tema'])
    current_rsi = float(latest['rsi'])
    current_adx = float(latest['adx'])

    # Chop/Trend Guardrail: ADX Threshold Check
    use_adx_filter = bool(kwargs.get("use_adx_filter", config.get("use_adx_filter", True)))
    adx_threshold = float(kwargs.get("adx_threshold", config.get("adx_threshold", 20.0)))

    if use_adx_filter and current_adx < adx_threshold:
        return {"action": "HOLD", "reason": f"ADX filter active: Market is choppy/ranging (ADX: {current_adx:.2f} < {adx_threshold:.2f})"}

    # Higher Timeframe (4H) Trend Confirmation
    macro_trend_long = True
    macro_trend_short = True
    disable_htf = bool(kwargs.get("disable_htf", False))

    if not disable_htf:
        try:
            df_4h = fetch_klines(symbol=symbol, interval="4h", limit=max(200, tema_period + 10))
            if not df_4h.empty and len(df_4h) >= tema_period:
                df_4h['tema_4h'] = calc_tema(df_4h['close'], period=tema_period)
                latest_4h_close = float(df_4h.iloc[-1]['close'])
                latest_4h_tema = float(df_4h.iloc[-1]['tema_4h'])
                
                macro_trend_long = latest_4h_close > latest_4h_tema
                macro_trend_short = latest_4h_close < latest_4h_tema
        except Exception as e:
            logging.warning(f"[strategy.py] Could not fetch 4H trend context for {symbol}: {e}")

    # Compute Volume Profile structures
    poc, vah, val = compute_volume_profile(df, lookback_bars=600)
    vp_gaps = calculate_volume_profile_gaps(df, detection_pct=float(config.get("vp_detection_pct", 0.07)))

    rsi_thresh = float(kwargs.get("rsi_thresh", config.get("rsi_thresh", 42.0)))
    use_rsi = bool(kwargs.get("use_rsi_filter", config.get("use_rsi_filter", True)))
    use_candlestick = bool(kwargs.get("use_candlestick_confirm", config.get("use_candlestick_confirm", True)))
    zone_tolerance = float(kwargs.get("zone_tolerance", config.get("zone_tolerance", 0.0075)))
    min_sentiment = float(kwargs.get("min_sentiment", config.get("min_sentiment", 0.0)))
    min_rr = float(kwargs.get("min_rr", config.get("min_rr", 2.0)))
    risk_pct = float(kwargs.get("risk_pct", config.get("risk_pct", 1.0)))
    max_sl_pct = float(kwargs.get("max_sl_pct", config.get("max_sl_pct", 0.02)))

    sentiment_score = kwargs.get("sentiment_score", 0.5)
    if sentiment_score < min_sentiment:
        return {"action": "HOLD", "reason": "Sentiment score below minimum threshold"}

    # Define key confluence levels
    upper_tema_zone = current_tema * (1.0 + zone_tolerance)
    lower_tema_zone = current_tema * (1.0 - zone_tolerance)

    bullish_candlestick = (latest['close'] > latest['open']) if use_candlestick else True
    bearish_candlestick = (latest['close'] < latest['open']) if use_candlestick else True

    rsi_long_ok = (current_rsi >= rsi_thresh) if use_rsi else True
    rsi_short_ok = (current_rsi <= (100 - rsi_thresh)) if use_rsi else True

    # Identify dynamic Volume Profile targets & stop anchors
    overhead_gaps = sorted([g for g in vp_gaps if g > current_close])
    underneath_gaps = sorted([g for g in vp_gaps if g < current_close], reverse=True)

    # LONG Entry Confluence Check
    near_tema = lower_tema_zone <= current_close <= upper_tema_zone
    near_val = not np.isnan(val) and abs(current_close - val) / current_close <= zone_tolerance
    near_gap_support = len(underneath_gaps) > 0 and (current_close - underneath_gaps[0]) / current_close <= zone_tolerance

    if (near_tema or near_val or near_gap_support) and rsi_long_ok and bullish_candlestick and macro_trend_long:
        sl_anchor = val if (not np.isnan(val) and val < current_close) else current_tema
        raw_stop_loss = min(sl_anchor * (1.0 - zone_tolerance), current_close * 0.99)
        tightest_allowed_sl = current_close * (1.0 - max_sl_pct)
        stop_loss = round(max(raw_stop_loss, tightest_allowed_sl), 5)
        
        risk_distance = current_close - stop_loss

        if risk_distance <= 0:
            return {"action": "HOLD", "reason": "Invalid risk distance computed for LONG"}

        tp_candidates = overhead_gaps + [x for x in [poc, vah] if not np.isnan(x) and x > current_close]
        min_tp = current_close + (risk_distance * min_rr)
        
        if tp_candidates:
            take_profit = round(max(min(tp_candidates), min_tp), 5)
        else:
            take_profit = round(min_tp, 5)

        computed_rr = round((take_profit - current_close) / risk_distance, 2)
        risk_amt = account_balance * (risk_pct / 100.0)
        position_size = round(risk_amt / risk_distance, 4)

        return {
            "action": "BUY",
            "symbol": symbol,
            "direction": "LONG",
            "entry_price": current_close,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward_ratio": computed_rr,
            "risk_pct": risk_pct,
            "position_size": position_size,
            "reason": "Long signal confirmed with TEMA, ADX trend strength, 4H HTF trend, and Volume Profile support confluence"
        }

    # SHORT Entry Confluence Check
    near_vah = not np.isnan(vah) and abs(current_close - vah) / current_close <= zone_tolerance
    near_gap_resistance = len(overhead_gaps) > 0 and (overhead_gaps[0] - current_close) / current_close <= zone_tolerance

    if (near_tema or near_vah or near_gap_resistance) and rsi_short_ok and bearish_candlestick and macro_trend_short:
        sl_anchor = vah if (not np.isnan(vah) and vah > current_close) else current_tema
        raw_stop_loss = max(sl_anchor * (1.0 + zone_tolerance), current_close * 1.01)
        tightest_allowed_sl = current_close * (1.0 + max_sl_pct)
        stop_loss = round(min(raw_stop_loss, tightest_allowed_sl), 5)

        risk_distance = stop_loss - current_close

        if risk_distance <= 0:
            return {"action": "HOLD", "reason": "Invalid risk distance computed for SHORT"}

        tp_candidates = underneath_gaps + [x for x in [poc, val] if not np.isnan(x) and x < current_close]
        min_tp = current_close - (risk_distance * min_rr)

        if tp_candidates:
            take_profit = round(min(max(tp_candidates), min_tp), 5)
        else:
            take_profit = round(min_tp, 5)

        computed_rr = round((current_close - take_profit) / risk_distance, 2)
        risk_amt = account_balance * (risk_pct / 100.0)
        position_size = round(risk_amt / risk_distance, 4)

        return {
            "action": "SELL",
            "symbol": symbol,
            "direction": "SHORT",
            "entry_price": current_close,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward_ratio": computed_rr,
            "risk_pct": risk_pct,
            "position_size": position_size,
            "reason": "Short signal confirmed with TEMA, ADX trend strength, 4H HTF trend, and Volume Profile resistance confluence"
        }

    return {"action": "HOLD", "reason": "Price action outside target TEMA/VP confluence zones or filters unmet"}