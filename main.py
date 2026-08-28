import os
import time
import logging
import gc
import pandas as pd
from dotenv import load_dotenv

from common import (
    ensure_schema_updated,
    get_db_connection,
    release_db_connection,
    check_daily_circuit_breaker,
    send_telegram_notification,
    check_asset_cooldown,
)
from live_executor import LiveExecutionEngine
from reconciler import reconcile_open_trades
from strategy import (
    fetch_klines,
    evaluate_signals,
    load_symbol_config,
    safe_float,
    calc_tema,
    calc_rsi,
    calc_adx,
    calc_atr,
)
from trade_manager import TradeManager

load_dotenv()

raw_symbols = os.getenv("TRADING_SYMBOLS") or os.getenv("WATCHLIST") or "XRP/USDT"
WATCHLIST = [s.strip() for s in raw_symbols.split(",") if s.strip()]

TIMEFRAME = os.getenv("TIMEFRAME", "1h")
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "60"))
ACCOUNT_RISK_PCT = float(os.getenv("ACCOUNT_RISK_PCT", "1.0"))
FALLBACK_BALANCE = float(os.getenv("ACCOUNT_BALANCE", "100.0"))

MEXC_API_KEY = os.getenv("MEXC_API_KEY", "")
MEXC_SECRET_KEY = os.getenv("MEXC_SECRET_KEY", "")

MAX_CONCURRENT_TRADES = 4  
MAX_ALLOCATION_PER_TRADE = float(os.getenv("MAX_ALLOCATION_PER_TRADE", 0.25))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("trading_bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main_engine")

execution_engine = LiveExecutionEngine(api_key=MEXC_API_KEY, secret_key=MEXC_SECRET_KEY)


def save_market_state(pair: str, rsi: float, adx: float, atr: float, tema: float, sentiment_bias: float, regime: str = "NEUTRAL"):
    """Persists real-time indicator values to PostgreSQL for Streamlit monitoring."""
    conn = get_db_connection()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            query = """
            INSERT INTO market_state (pair, rsi, adx, atr, tema, sentiment_bias, regime, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (pair) DO UPDATE SET
                rsi = EXCLUDED.rsi,
                adx = EXCLUDED.adx,
                atr = EXCLUDED.atr,
                tema = EXCLUDED.tema,
                sentiment_bias = EXCLUDED.sentiment_bias,
                regime = EXCLUDED.regime,
                updated_at = NOW();
            """
            cur.execute(query, (pair, float(rsi), float(adx), float(atr), float(tema), float(sentiment_bias), regime))
            conn.commit()
    except Exception as e:
        logger.error(f"[{pair}] Error logging market state: {e}")
    finally:
        release_db_connection(conn)


def process_symbol(symbol: str, tm: TradeManager, active_usdt_balance: float):
    """Fetches market klines, loads strategy params, evaluates signals, and manages orders."""
    
    # Check 2-Hour Asset Cooldown Guard
    if check_asset_cooldown(symbol):
        logger.info(f"[{symbol}] Asset is currently in a 2-hour post-trade cooldown. Skipping signal generation.")
        return

    logger.info(f"[{symbol}] Processing market signal check...")

    df_klines = fetch_klines(symbol=symbol, interval=TIMEFRAME, limit=300)
    if df_klines.empty:
        logger.warning(f"[{symbol}] Unable to retrieve kline data. Skipping processing cycle.")
        return

    cfg = load_symbol_config(symbol)
    sentiment_score = 0.5  # Default baseline sentiment

    # Compute current indicators for DB state logging
    try:
        tema_p = int(cfg.get("tema_period", 200))
        rsi_p = int(cfg.get("rsi_period", 14))
        adx_p = int(cfg.get("adx_period", 14))
        atr_p = int(cfg.get("atr_period", 14))

        if len(df_klines) >= max(tema_p, adx_p + 1, atr_p + 1):
            s_tema = calc_tema(df_klines["close"], period=tema_p).iloc[-1]
            s_rsi = calc_rsi(df_klines["close"], period=rsi_p).iloc[-1]
            s_adx = calc_adx(df_klines, period=adx_p).iloc[-1]
            s_atr = calc_atr(df_klines, period=atr_p).iloc[-1]
            
            # Simple regime classifier based on ADX & TEMA
            current_close = df_klines["close"].iloc[-1]
            regime = "TRENDING_BULL" if current_close > s_tema and s_adx >= 20 else ("TRENDING_BEAR" if current_close < s_tema and s_adx >= 20 else "RANGING")

            save_market_state(
                pair=symbol,
                rsi=s_rsi,
                adx=s_adx,
                atr=s_atr,
                tema=s_tema,
                sentiment_bias=sentiment_score,
                regime=regime
            )
    except Exception as state_err:
        logger.error(f"[{symbol}] Failed to compute/persist market state: {state_err}")

    if tm.has_open_trade(symbol):
        logger.info(f"[{symbol}] Open position already exists in trade manager. Skipping new signal checks.")
        return

    # Fully harmonized parameter passing
    signal = evaluate_signals(
        df=df_klines,
        symbol=symbol,
        account_balance=active_usdt_balance,
        tema_period=cfg.get("tema_period", 200),
        rsi_period=cfg.get("rsi_period", 14),
        rsi_thresh=cfg.get("rsi_thresh", 42.0),
        adx_period=cfg.get("adx_period", 14),
        adx_threshold=cfg.get("adx_threshold", 20.0),
        use_adx_filter=cfg.get("use_adx_filter", True),
        use_rsi_filter=cfg.get("use_rsi_filter", True),
        use_candlestick_confirm=cfg.get("use_candlestick_confirm", True),
        zone_tolerance=cfg.get("zone_tolerance", 0.0075),
        max_sl_pct=cfg.get("max_sl_pct", 0.02),
        min_sentiment=cfg.get("min_sentiment", 0.0),
        min_rr=cfg.get("min_rr", 2.0),
        risk_pct=cfg.get("risk_pct", ACCOUNT_RISK_PCT),
        atr_period=cfg.get("atr_period", 14),
        atr_mult=cfg.get("atr_mult", 2.0),
        use_atr_sl=cfg.get("use_atr_sl", True),
        disable_htf=cfg.get("disable_htf", False),
        sentiment_score=sentiment_score
    )

    action = signal.get("action", "HOLD")
    reason = signal.get("reason", "")

    if action in ["BUY", "LONG"]:
        entry_p = safe_float(signal.get("entry_price"))
        sl_p = safe_float(signal.get("stop_loss"))
        tp_p = safe_float(signal.get("take_profit"))
        pos_size = safe_float(signal.get("position_size"))
        rr_ratio = safe_float(signal.get("risk_reward_ratio", 2.0))

        logger.info(f"[{symbol}] VALID BUY SIGNAL: Entry=${entry_p:.5f}, SL=${sl_p:.5f}, TP=${tp_p:.5f}, R:R={rr_ratio}, Reason: {reason}")
        
        trade_amount_usd = min(active_usdt_balance * MAX_ALLOCATION_PER_TRADE, active_usdt_balance * 0.98)
        
        exec_success = execution_engine.execute_live_order(
            pair=symbol,
            direction="BUY",
            entry_price=entry_p,
            stop_loss=sl_p,
            take_profit=tp_p,
            amount_usd=trade_amount_usd
        )

        if exec_success:
            tm.record_executed_trade(
                pair=symbol,
                direction="BUY",
                entry_price=entry_p,
                stop_loss=sl_p,
                take_profit=tp_p,
                position_size=pos_size,
                account_balance=active_usdt_balance,
                risk_reward_ratio=rr_ratio
            )
            send_telegram_notification(
                f"<b>🟢 LIVE SPOT ORDER EXECUTED</b>\n\n"
                f"<b>Pair:</b> <code>{symbol}</code>\n"
                f"<b>Entry:</b> ${entry_p:.5f}\n"
                f"<b>Stop Loss:</b> ${sl_p:.5f}\n"
                f"<b>Take Profit:</b> ${tp_p:.5f}\n"
                f"<b>R:R Ratio:</b> {rr_ratio}\n"
                f"<b>Reason:</b> {reason}"
            )
    elif action in ["SELL", "SHORT"]:
        logger.info(f"[{symbol}] Short signal evaluated ({reason}). Spot execution mode active: Skipping short setup.")
    else:
        logger.info(f"[{symbol}] Hold Signal ({reason})")


def main():
    ensure_schema_updated()
    tm = TradeManager()

    logger.info("Initializing Trading Engine Core Loop...")

    while True:
        try:
            live_balance = execution_engine.fetch_available_usdt_balance()
            active_usdt_balance = live_balance if live_balance > 1.0 else FALLBACK_BALANCE
            
            if check_daily_circuit_breaker(max_loss_pct=3.0, account_balance=active_usdt_balance):
                logger.warning("Daily Circuit Breaker Triggered. Max loss reached for today. Pausing entry execution.")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            reconcile_open_trades(execution_engine)

            for sym in WATCHLIST:
                process_symbol(sym, tm, active_usdt_balance)

        except Exception as loop_err:
            logger.error(f"Error in main bot execution cycle: {loop_err}")

        finally:
            # Force cleanup of memory overhead after processing all symbols
            gc.collect()

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()