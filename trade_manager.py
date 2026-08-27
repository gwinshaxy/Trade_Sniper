import pandas as pd
from common import logger, get_db_connection, release_db_connection


class TradeManager:
    """
    Manages trade lifecycles, stop loss/take profit checks, break-even locking,
    and dynamic TEMA/ATR trailing logic adhering to institutional risk rules.
    """

    def __init__(self, spread_buffer_pct: float = 0.0005, tema_offset_pct: float = 0.001, atr_trail_mult: float = 1.5):
        self.spread_buffer_pct = spread_buffer_pct
        self.tema_offset_pct = tema_offset_pct
        self.atr_trail_mult = atr_trail_mult

    def has_open_trade(self, symbol: str) -> bool:
        """Checks if there is an active OPEN or BE_LOCKED trade in the database for the given symbol."""
        conn = get_db_connection()
        if not conn:
            logger.error(f"[{symbol}] Database connection failed during open trade check.")
            return False

        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT COUNT(*) 
                FROM trade_setups 
                WHERE pair = %s AND trade_state IN ('OPEN', 'BE_LOCKED', 'TRAILING');
                """,
                (symbol,)
            )
            count = cursor.fetchone()[0]
            return count > 0
        except Exception as e:
            logger.error(f"[{symbol}] Failed to check open trade status: {e}")
            return False
        finally:
            release_db_connection(conn)

    def record_executed_trade(
        self, 
        pair: str, 
        direction: str, 
        entry_price: float, 
        stop_loss: float, 
        take_profit: float, 
        position_size: float, 
        account_balance: float
    ) -> bool:
        """Records a successfully executed live trade into the database."""
        conn = get_db_connection()
        if not conn:
            logger.error(f"[{pair}] Database connection failed when recording trade.")
            return False

        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO trade_setups 
                (pair, direction, entry_price, stop_loss, take_profit, position_size, account_balance, trade_state, status) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'OPEN', 'EXECUTED');
                """,
                (pair, direction, entry_price, stop_loss, take_profit, position_size, account_balance)
            )
            conn.commit()
            logger.info(f"[{pair}] Open trade recorded successfully in database.")
            return True
        except Exception as e:
            logger.error(f"[{pair}] Failed to record executed trade: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            release_db_connection(conn)

    def process_trade(self, trade_row: pd.Series, latest_candle: pd.Series) -> dict:
        trade_id = trade_row.get('id')
        pair = trade_row.get('pair', 'N/A')
        direction = str(trade_row.get('direction', '')).upper()
        
        entry = float(trade_row.get('entry_price', 0.0))
        curr_sl = float(trade_row.get('stop_loss', 0.0))
        curr_tp = float(trade_row.get('take_profit', 0.0))
        state = str(trade_row.get('trade_state', 'OPEN')).upper()

        curr_price = float(latest_candle.get('close', 0.0))
        tema = float(latest_candle.get('200_TEMA', latest_candle.get('tema', curr_price)))
        atr = float(latest_candle.get('atr', 0.0))

        if curr_price <= 0 or entry <= 0:
            return {"action": "NONE"}

        initial_risk = abs(entry - curr_sl)

        # 1. STOP LOSS BREACH CHECK
        sl_hit = (curr_price <= curr_sl) if direction in ['BUY', 'LONG'] else (curr_price >= curr_sl)
        if sl_hit:
            return {
                "action": "CLOSE_SL",
                "trade_id": trade_id,
                "exit_price": curr_sl,
                "msg": (
                    f"🛑 <b>STOP LOSS HIT</b>\n\n"
                    f"<b>Trade ID:</b> <code>#{trade_id}</code>\n"
                    f"<b>Pair:</b> <code>{pair}</code>\n"
                    f"<b>Direction:</b> <code>{direction}</code>\n"
                    f"<b>Exit Price:</b> ${curr_sl:.5f}"
                )
            }

        # 2. TAKE PROFIT BREACH CHECK
        tp_hit = (curr_price >= curr_tp) if direction in ['BUY', 'LONG'] else (curr_price <= curr_tp)
        if tp_hit:
            return {
                "action": "CLOSE_TP",
                "trade_id": trade_id,
                "exit_price": curr_tp,
                "msg": (
                    f"🎯 <b>TAKE PROFIT HIT</b>\n\n"
                    f"<b>Trade ID:</b> <code>#{trade_id}</code>\n"
                    f"<b>Pair:</b> <code>{pair}</code>\n"
                    f"<b>Direction:</b> <code>{direction}</code>\n"
                    f"<b>Exit Price:</b> ${curr_tp:.5f}"
                )
            }

        # 3. BREAK-EVEN CHECK (1:1 R:R Achieved)
        if state == 'OPEN':
            tp1_target = entry + initial_risk if direction in ['BUY', 'LONG'] else entry - initial_risk
            be_triggered = (curr_price >= tp1_target) if direction in ['BUY', 'LONG'] else (curr_price <= tp1_target)

            if be_triggered:
                be_price = entry * (1 + self.spread_buffer_pct) if direction in ['BUY', 'LONG'] else entry * (1 - self.spread_buffer_pct)
                new_sl = round(be_price, 5)
                
                if (direction in ['BUY', 'LONG'] and new_sl > curr_sl) or (direction in ['SELL', 'SHORT'] and new_sl < curr_sl):
                    return {
                        "action": "UPDATE_SL",
                        "trade_id": trade_id,
                        "new_sl": new_sl,
                        "new_state": "BE_LOCKED",
                        "msg": (
                            f"🔒 <b>BREAK-EVEN TRIGGERED</b>\n\n"
                            f"<b>Trade ID:</b> <code>#{trade_id}</code>\n"
                            f"<b>Pair:</b> <code>{pair}</code>\n"
                            f"<b>New Stop Loss:</b> ${new_sl:.5f} (1:1 R:R achieved)"
                        )
                    }

        # 4. DYNAMIC TRAILING STOP (Hybrid TEMA & ATR Trailing Offset)
        elif state in ['BE_LOCKED', 'TRAILING']:
            atr_offset = (atr * self.atr_trail_mult) if atr > 0 else (tema * self.tema_offset_pct)

            if direction in ['BUY', 'LONG']:
                proposed_sl = max(tema * (1 - self.tema_offset_pct), curr_price - atr_offset)
                
                if proposed_sl > curr_sl and curr_price > tema:
                    return {
                        "action": "UPDATE_SL",
                        "trade_id": trade_id,
                        "new_sl": round(proposed_sl, 5),
                        "new_state": "TRAILING",
                        "msg": (
                            f"📈 <b>TRAILING STOP LOSS UPDATED</b>\n\n"
                            f"<b>Trade ID:</b> <code>#{trade_id}</code>\n"
                            f"<b>Pair:</b> <code>{pair}</code>\n"
                            f"<b>New Trailing SL:</b> ${proposed_sl:.5f}\n"
                            f"<b>Anchor Price:</b> ${curr_price:.5f}"
                        )
                    }

            elif direction in ['SELL', 'SHORT']:
                proposed_sl = min(tema * (1 + self.tema_offset_pct), curr_price + atr_offset)
                
                if proposed_sl < curr_sl and curr_price < tema:
                    return {
                        "action": "UPDATE_SL",
                        "trade_id": trade_id,
                        "new_sl": round(proposed_sl, 5),
                        "new_state": "TRAILING",
                        "msg": (
                            f"📉 <b>TRAILING STOP LOSS UPDATED</b>\n\n"
                            f"<b>Trade ID:</b> <code>#{trade_id}</code>\n"
                            f"<b>Pair:</b> <code>{pair}</code>\n"
                            f"<b>New Trailing SL:</b> ${proposed_sl:.5f}\n"
                            f"<b>Anchor Price:</b> ${curr_price:.5f}"
                        )
                    }

        return {"action": "NONE"}