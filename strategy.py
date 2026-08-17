import json
import logging
import os
import re
import urllib.request
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

try:
  import psycopg2
  from psycopg2.extras import RealDictCursor

  PSYCOPG2_AVAILABLE = True
except ImportError:
  PSYCOPG2_AVAILABLE = False


def normalize_symbol(symbol: str) -> str:
  if not symbol:
    return ""
  s = str(symbol).replace('"', "").replace("'", "").strip().upper()
  if "/" in s:
    return s
  if s.endswith("USDT") and len(s) > 4:
    return f"{s[:-4]}/{s[-4:]}"
  return s


def load_symbol_config(symbol: str) -> dict:
  """Loads optimized strategy parameters from PostgreSQL database safely closing connections in a finally block."""
  clean_symbol = normalize_symbol(symbol).replace("/", "").upper()
  db_url = os.getenv("DATABASE_URL")

  if db_url:
    db_url = db_url.strip('"').strip("'").strip()

  if PSYCOPG2_AVAILABLE and db_url:
    conn = None
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
            (clean_symbol,),
        )
        row = cur.fetchone()
        if row:
          config = dict(row)
          config.setdefault("rsi_thresh", 42.0)
          config.setdefault("use_rsi_filter", True)
          config.setdefault("use_candlestick_confirm", True)
          config.setdefault("use_adx_filter", True)
          config.setdefault("adx_period", 14)
          config.setdefault("adx_threshold", 20.0)
          config.setdefault("max_sl_pct", 0.02)
          config.setdefault("lookback_bars", 600)
          config.setdefault("vp_va_pct", 0.70)
          logging.debug(
              f"[load_symbol_config] Loaded dynamic DEAP params for {clean_symbol}"
          )
          return config
        else:
          logging.warning(
              f"[load_symbol_config] Parameter row missing for {clean_symbol},"
              " using defaults."
          )
    except Exception as e:
      logging.error(
          f"[load_symbol_config] DB query failed for {clean_symbol}: {e}"
      )
    finally:
      if conn:
        try:
          conn.close()
        except Exception:
          pass

  return {
      "tema_period": 200,
      "rsi_period": 14,
      "rsi_thresh": 42.0,
      "adx_period": 14,
      "adx_threshold": 20.0,
      "use_adx_filter": True,
      "zone_tolerance": 0.0075,
      "min_sentiment": 0.0,
      "risk_pct": 1.0,
      "min_rr": 2.0,
      "vp_detection_pct": 0.07,
      "use_rsi_filter": True,
      "use_candlestick_confirm": True,
      "max_sl_pct": 0.02,
      "lookback_bars": 600,
      "vp_va_pct": 0.70,
  }


def get_ai_sentiment_score(text: str) -> float:
  """Analyzes financial sentiment of a text string using Gemini AI API."""
  api_key = os.getenv("GEMINI_API_KEY")
  if not api_key:
    return 0.5
  try:
    client = genai.Client(api_key=api_key)
    prompt = (
        "Analyze the financial sentiment of the following text and return ONLY"
        " a single float number between 0.0 (bearish) and 1.0"
        f" (bullish):\n\n'{text}'"
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    )

    match = re.search(r"0\.\d+|1\.0|0|1", response.text.strip())
    if match:
      return max(0.0, min(1.0, float(match.group(0))))
    return 0.5
  except Exception:
    return 0.5


def fetch_cryptocompare_klines(
    symbol: str = "ETH/USDT", limit: int = 1000
) -> pd.DataFrame:
  """Fallback: CryptoCompare Historical Hourly Data API."""
  try:
    clean = symbol.replace("/", "").upper()
    fsym = clean[:-4] if clean.endswith("USDT") else clean[:-3]
    tsym = "USDT" if clean.endswith("USDT") else "USD"

    url = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={fsym}&tsym={tsym}&limit={min(limit, 2000)}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    with urllib.request.urlopen(req, timeout=8) as resp:
      data = json.loads(resp.read().decode("utf-8"))

    if (
        data.get("Response") == "Success"
        and "Data" in data
        and "Data" in data["Data"]
    ):
      candles = data["Data"]["Data"]
      df = pd.DataFrame(candles)
      df.rename(
          columns={"time": "open_time", "volumeto": "volume"}, inplace=True
      )
      df["time"] = pd.to_datetime(df["open_time"], unit="s").dt.strftime(
          "%Y-%m-%d %H:%M:%S"
      )

      for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

      return df[["time", "open", "high", "low", "close", "volume"]]
  except Exception as e:
    logging.debug(f"[CryptoCompare Fallback Error]: {e}")

  return pd.DataFrame()


def fetch_klines(
    symbol: str = "BNB/USDT", interval: str = "1h", limit: int = 1000
) -> pd.DataFrame:
  norm_sym = normalize_symbol(symbol)
  binance_symbol = norm_sym.replace("/", "")

  # Bybit Interval Mapping
  bybit_interval_map = {
      "1m": "1",
      "5m": "5",
      "15m": "15",
      "1h": "60",
      "4h": "240",
      "1d": "D",
  }
  bybit_interval = bybit_interval_map.get(interval, "60")

  # 1. Primary & Exchange Endpoints
  urls = [
      (
          "https://api.binance.com/api/v3/klines?symbol="
          f"{binance_symbol}&interval={interval}&limit={min(limit, 1000)}"
      ),
      (
          "https://api.binance.us/api/v3/klines?symbol="
          f"{binance_symbol}&interval={interval}&limit={min(limit, 1000)}"
      ),
      (
          "https://api.bybit.com/v5/market/kline?category=spot&symbol="
          f"{binance_symbol}&interval={bybit_interval}&limit={min(limit, 1000)}"
      ),
  ]

  for url in urls:
    try:
      req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
      with urllib.request.urlopen(req, timeout=6) as resp:
        data = json.loads(resp.read().decode("utf-8"))

      # Binance / Binance US structure
      if isinstance(data, list) and len(data) > 0:
        df = pd.DataFrame(
            data,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "qav",
                "not",
                "tbba",
                "tbqa",
                "ignore",
            ],
        )
        df["time"] = pd.to_datetime(df["open_time"], unit="ms").dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        for col in ["open", "high", "low", "close", "volume"]:
          df[col] = df[col].astype(float)
        return df[["time", "open", "high", "low", "close", "volume"]]

      # Bybit structure
      elif (
          isinstance(data, dict)
          and "result" in data
          and "list" in data["result"]
      ):
        raw_list = data["result"]["list"]
        df = pd.DataFrame(
            raw_list,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "turnover",
            ],
        )
        df = df.iloc[::-1].reset_index(drop=True)
        df["time"] = pd.to_datetime(
            df["open_time"].astype(int), unit="ms"
        ).dt.strftime("%Y-%m-%d %H:%M:%S")
        for col in ["open", "high", "low", "close", "volume"]:
          df[col] = df[col].astype(float)
        return df[["time", "open", "high", "low", "close", "volume"]]

    except Exception as err:
      logging.debug(f"[fetch_klines] Endpoint bypassed ({url}): {err}")
      continue

  # 2. Third-Party Fallback (CryptoCompare)
  df_fallback = fetch_cryptocompare_klines(symbol=norm_sym, limit=limit)
  if not df_fallback.empty:
    logging.info(
        "[fetch_klines] Successfully retrieved fallback data from"
        f" CryptoCompare for {norm_sym}"
    )
    return df_fallback

  logging.error(
      "[fetch_klines] All primary and third-party data endpoints failed for"
      f" symbol: {symbol}"
  )
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
  """Calculates Average Directional Index (ADX) with optimized series manipulation."""
  if df.empty or len(df) < period + 1:
    return pd.Series(0.0, index=df.index)

  high = df["high"]
  low = df["low"]
  prev_close = df["close"].shift(1)

  tr1 = high - low
  tr2 = (high - prev_close).abs()
  tr3 = (low - prev_close).abs()
  tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

  up_move = high - high.shift(1)
  down_move = low.shift(1) - low

  plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
  minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

  # Wilder's Smoothing
  tr_smooth = tr.ewm(alpha=1 / period, adjust=False).mean()
  plus_di = (
      100
      * (
          pd.Series(plus_dm, index=df.index)
          .ewm(alpha=1 / period, adjust=False)
          .mean()
          / tr_smooth.replace(0, np.nan)
      )
  )
  minus_di = (
      100
      * (
          pd.Series(minus_dm, index=df.index)
          .ewm(alpha=1 / period, adjust=False)
          .mean()
          / tr_smooth.replace(0, np.nan)
      )
  )

  dx = (
      abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
  ) * 100
  adx = dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0.0)
  return adx


def compute_volume_profile(
    df: pd.DataFrame,
    num_bins: int = 100,
    lookback_bars: int = 600,
    va_pct: float = 0.70,
):
  """Computes POC, VAH, and VAL aligned with Pine Script VP logic."""
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

  pLST, pHST = float(df_range["low"].min()), float(df_range["high"].max())
  if pLST >= pHST or np.isnan(pLST) or np.isnan(pHST):
    return np.nan, np.nan, np.nan

  pSTP = (pHST - pLST) / num_bins
  if pSTP <= 0:
    return np.nan, np.nan, np.nan

  vD_vt = np.zeros(num_bins)

  for _, row in df_range.iterrows():
    lL, lH, lV = float(row["low"]), float(row["high"]), float(row["volume"])
    lR = max(lH - lL, 1e-8)

    sSI = max(int(np.floor((lL - pLST) / pSTP)), 0)
    eSI = min(int(np.floor((lH - pLST) / pSTP)), num_bins - 1)

    for pLI in range(sSI, eSI + 1):
      pL = pLST + pLI * pSTP

      if lL >= pL and lH > pL + pSTP:
        vPOR = (pL + pSTP - lL) / lR
      elif lH <= pL + pSTP and lL < pL:
        vPOR = (lH - pL) / lR
      elif lL >= pL and lH <= pL + pSTP:
        vPOR = 1.0
      else:
        vPOR = pSTP / lR

      vD_vt[pLI] += lV * max(vPOR, 0.0)

  pcL = int(np.argmax(vD_vt))
  poc = round(pLST + (pcL + 0.5) * pSTP, 2)

  ttV = max(np.sum(vD_vt), 1e-10) * va_pct
  va = vD_vt[pcL]
  laP, lbP = pcL, pcL

  max_iter = num_bins * 2
  iter_count = 0

  while va < ttV and iter_count < max_iter:
    iter_count += 1
    if lbP == 0 and laP == num_bins - 1:
      break

    vaP = vD_vt[laP + 1] if laP < num_bins - 1 else 0.0
    vbP = vD_vt[lbP - 1] if lbP > 0 else 0.0

    if vaP >= vbP:
      va += vaP
      laP += 1
    else:
      va += vbP
      lbP -= 1

  vaH = round(pLST + (laP + 1.0) * pSTP, 2)
  vaL = round(pLST + (lbP + 0.0) * pSTP, 2)

  return float(poc), float(vaH), float(vaL)


def calculate_volume_profile_gaps(
    df: pd.DataFrame,
    num_bins: int = 100,
    lookback_bars: int = 600,
    detection_pct: float = 0.07,
) -> list:
  """Identifies Volume Profile Gaps matching Pine Script vgSH logic."""
  if (
      df.empty
      or "volume" not in df.columns
      or "high" not in df.columns
      or "low" not in df.columns
  ):
    return []

  df_range = df.tail(lookback_bars).copy()
  if df_range.empty:
    return []

  pLST, pHST = float(df_range["low"].min()), float(df_range["high"].max())
  if pLST >= pHST or np.isnan(pLST) or np.isnan(pHST):
    return []

  pSTP = (pHST - pLST) / num_bins
  if pSTP <= 0:
    return []

  vD_vt = np.zeros(num_bins)

  for _, row in df_range.iterrows():
    lL, lH, lV = float(row["low"]), float(row["high"]), float(row["volume"])
    lR = max(lH - lL, 1e-8)

    sSI = max(int(np.floor((lL - pLST) / pSTP)), 0)
    eSI = min(int(np.floor((lH - pLST) / pSTP)), num_bins - 1)

    for pLI in range(sSI, eSI + 1):
      pL = pLST + pLI * pSTP

      if lL >= pL and lH > pL + pSTP:
        vPOR = (pL + pSTP - lL) / lR
      elif lH <= pL + pSTP and lL < pL:
        vPOR = (lH - pL) / lR
      elif lL >= pL and lH <= pL + pSTP:
        vPOR = 1.0
      else:
        vPOR = pSTP / lR

      vD_vt[pLI] += lV * max(vPOR, 0.0)

  noN = int(num_bins * detection_pct)
  if noN < 1:
    noN = 1

  max_val = np.max(vD_vt)
  tVT = list(vD_vt)

  for _ in range(noN):
    tVT.insert(0, max_val)
    tVT.append(max_val)

  gap_prices = []

  for vn in range(2 * noN, num_bins + 2 * noN):
    uNth = True
    for cVN in range(vn - 2 * noN, vn - noN):
      if tVT[vn - noN] >= tVT[cVN]:
        uNth = False
        break

    lNth = True
    for cVN in range(vn - noN + 1, vn + 1):
      if tVT[vn - noN] >= tVT[cVN]:
        lNth = False
        break

    if uNth and lNth:
      bin_idx = vn - 2 * noN
      gap_price = round(pLST + (bin_idx + 0.5) * pSTP, 2)
      if pLST <= gap_price <= pHST:
        gap_prices.append(gap_price)

  return list(dict.fromkeys(gap_prices))


def evaluate_signals(
    df: pd.DataFrame,
    symbol: str = "ETH/USDT",
    account_balance: float = 10000.0,
    **kwargs,
) -> dict:
  config = load_symbol_config(symbol)
  tema_period = int(kwargs.get("tema_period", config.get("tema_period", 200)))
  rsi_period = int(kwargs.get("rsi_period", config.get("rsi_period", 14)))
  adx_period = int(kwargs.get("adx_period", config.get("adx_period", 14)))

  if df.empty or len(df) < max(tema_period, adx_period + 1):
    return {
        "action": "HOLD",
        "reason": "Insufficient historical data for indicator convergence",
    }

  # Calculate indicators on current timeframe
  df["tema"] = calc_tema(df["close"], period=tema_period)
  df["rsi"] = calc_rsi(df["close"], period=rsi_period)
  df["adx"] = calc_adx(df, period=adx_period)

  latest = df.iloc[-1]
  current_close = float(latest["close"])
  current_tema = float(latest["tema"])
  current_rsi = float(latest["rsi"])
  current_adx = float(latest["adx"])

  # Chop/Trend Guardrail: ADX Threshold Check
  use_adx_filter = bool(
      kwargs.get("use_adx_filter", config.get("use_adx_filter", True))
  )
  adx_threshold = float(
      kwargs.get("adx_threshold", config.get("adx_threshold", 20.0))
  )

  if use_adx_filter and current_adx < adx_threshold:
    return {
        "action": "HOLD",
        "reason": (
            "ADX filter active: Market is choppy/ranging (ADX:"
            f" {current_adx:.2f} < {adx_threshold:.2f})"
        ),
    }

  # Higher Timeframe (4H) Trend Confirmation
  macro_trend_long = True
  macro_trend_short = True
  disable_htf = bool(kwargs.get("disable_htf", False))

  if not disable_htf:
    try:
      df_4h = fetch_klines(
          symbol=symbol, interval="4h", limit=max(200, tema_period + 10)
      )
      if not df_4h.empty and len(df_4h) >= tema_period:
        df_4h["tema_4h"] = calc_tema(df_4h["close"], period=tema_period)
        latest_4h_close = float(df_4h.iloc[-1]["close"])
        latest_4h_tema = float(df_4h.iloc[-1]["tema_4h"])

        macro_trend_long = latest_4h_close > latest_4h_tema
        macro_trend_short = latest_4h_close < latest_4h_tema
    except Exception as e:
      logging.warning(
          f"[strategy.py] Could not fetch 4H trend context for {symbol}: {e}"
      )

  # Compute Volume Profile structures
  poc, vah, val = compute_volume_profile(df, lookback_bars=600)
  vp_gaps = calculate_volume_profile_gaps(
      df, detection_pct=float(config.get("vp_detection_pct", 0.07))
  )

  rsi_thresh = float(kwargs.get("rsi_thresh", config.get("rsi_thresh", 42.0)))
  use_rsi = bool(
      kwargs.get("use_rsi_filter", config.get("use_rsi_filter", True))
  )
  use_candlestick = bool(
      kwargs.get(
          "use_candlestick_confirm", config.get("use_candlestick_confirm", True)
      )
  )
  zone_tolerance = float(
      kwargs.get("zone_tolerance", config.get("zone_tolerance", 0.0075))
  )
  min_sentiment = float(
      kwargs.get("min_sentiment", config.get("min_sentiment", 0.0))
  )
  min_rr = float(kwargs.get("min_rr", config.get("min_rr", 2.0)))
  risk_pct = float(kwargs.get("risk_pct", config.get("risk_pct", 1.0)))
  max_sl_pct = float(
      kwargs.get("max_sl_pct", config.get("max_sl_pct", 0.02))
  )

  sentiment_score = kwargs.get("sentiment_score", 0.5)
  if sentiment_score < min_sentiment:
    return {
        "action": "HOLD",
        "reason": "Sentiment score below minimum threshold",
    }

  # Define key confluence levels
  upper_tema_zone = current_tema * (1.0 + zone_tolerance)
  lower_tema_zone = current_tema * (1.0 - zone_tolerance)

  bullish_candlestick = (
      (latest["close"] > latest["open"]) if use_candlestick else True
  )
  bearish_candlestick = (
      (latest["close"] < latest["open"]) if use_candlestick else True
  )

  rsi_long_ok = (current_rsi >= rsi_thresh) if use_rsi else True
  rsi_short_ok = (current_rsi <= (100 - rsi_thresh)) if use_rsi else True

  # Identify dynamic Volume Profile targets & stop anchors
  overhead_gaps = sorted([g for g in vp_gaps if g > current_close])
  underneath_gaps = sorted(
      [g for g in vp_gaps if g < current_close], reverse=True
  )

  # LONG Entry Confluence Check
  near_tema = lower_tema_zone <= current_close <= upper_tema_zone
  near_val = (
      not np.isnan(val)
      and abs(current_close - val) / current_close <= zone_tolerance
  )
  near_gap_support = (
      len(underneath_gaps) > 0
      and (current_close - underneath_gaps[0]) / current_close <= zone_tolerance
  )

  if (
      (near_tema or near_val or near_gap_support)
      and rsi_long_ok
      and bullish_candlestick
      and macro_trend_long
  ):
    sl_anchor = (
        val if (not np.isnan(val) and val < current_close) else current_tema
    )
    raw_stop_loss = min(
        sl_anchor * (1.0 - zone_tolerance), current_close * 0.99
    )
    tightest_allowed_sl = current_close * (1.0 - max_sl_pct)
    stop_loss = round(max(raw_stop_loss, tightest_allowed_sl), 5)

    risk_distance = current_close - stop_loss

    if risk_distance <= 0:
      return {
          "action": "HOLD",
          "reason": "Invalid risk distance computed for LONG",
      }

    tp_candidates = overhead_gaps + [
        x for x in [poc, vah] if not np.isnan(x) and x > current_close
    ]
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
        "reason": (
            "Long signal confirmed with TEMA, ADX trend strength, 4H HTF"
            " trend, and Volume Profile support confluence"
        ),
    }

  # SHORT Entry Confluence Check
  near_vah = (
      not np.isnan(vah)
      and abs(current_close - vah) / current_close <= zone_tolerance
  )
  near_gap_resistance = (
      len(overhead_gaps) > 0
      and (overhead_gaps[0] - current_close) / current_close <= zone_tolerance
  )

  if (
      (near_tema or near_vah or near_gap_resistance)
      and rsi_short_ok
      and bearish_candlestick
      and macro_trend_short
  ):
    sl_anchor = (
        vah if (not np.isnan(vah) and vah > current_close) else current_tema
    )
    raw_stop_loss = max(
        sl_anchor * (1.0 + zone_tolerance), current_close * 1.01
    )
    tightest_allowed_sl = current_close * (1.0 + max_sl_pct)
    stop_loss = round(min(raw_stop_loss, tightest_allowed_sl), 5)

    risk_distance = stop_loss - current_close

    if risk_distance <= 0:
      return {
          "action": "HOLD",
          "reason": "Invalid risk distance computed for SHORT",
      }

    tp_candidates = underneath_gaps + [
        x for x in [poc, val] if not np.isnan(x) and x < current_close
    ]
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
        "reason": (
            "Short signal confirmed with TEMA, ADX trend strength, 4H HTF"
            " trend, and Volume Profile resistance confluence"
        ),
    }

  return {
      "action": "HOLD",
      "reason": (
          "Price action outside target TEMA/VP confluence zones or filters"
          " unmet"
      ),
  }