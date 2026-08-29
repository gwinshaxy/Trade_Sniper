import pandas as pd
from common import logger, get_db_connection, release_db_connection


class TradeManager:
    """
    Manages trade lifecycles, stop loss/take profit checks, break-even locking,
    and High-Water Mark dynamic trailing stop loss logic.
    """

    def __init__(self, spread_buffer_pct: float = 0.0005, tema_offset_pct: float = 0.001, atr_trail_mult: float = 1.5):
        self.spread_buffer_pct = spread_buffer_pct
        self.tema_offset_pct = tema_offset_pct
        self.atr_trail_mult = atr_trail_mult

    def has_open_trade(self, symbol: str) -> bool:
        """Checks if there is an active OPEN, BE_LOCKED, or TRAILING trade in the database for the given symbol."""
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
        account_balance: float,
        risk_reward_ratio: float = 2.0
    ) -> bool:
        """Records a successfully executed live trade with High-Water Mark fields into the database."""
        conn = get_db_connection()
        if not conn:
            logger.error(f"[{pair}] Database connection failed when recording trade.")
            return False

        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO trade_setups 
                (pair, direction, entry_price, stop_loss, take_profit, risk_reward_ratio, position_size, 
                 account_balance, trade_state, status, highest_price, trailing_stop_price) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'OPEN', 'EXECUTED', %s, %s);
                """,
                (pair, direction, entry_price, stop_loss, take_profit, risk_reward_ratio, position_size, 
                 account_balance, entry_price, stop_loss)
            )
            conn.commit()
            logger.info(f"[{pair}] Open trade recorded in DB with initial High-Water Mark ${entry_price:.5f}.")
            return True
        except Exception as e:
            logger.error(f"[{pair}] Failed to record executed trade: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            release_db_connection(conn)

    def get_all_open_trades(self) -> list:
        """Retrieves all active trades currently managed by the bot."""
        conn = get_db_connection()
        if not conn:
            return []
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, pair, direction, entry_price, stop_loss, take_profit, position_size, 
                       trade_state, highest_price, trailing_stop_price
                FROM trade_setups 
                WHERE trade_state IN ('OPEN', 'BE_LOCKED', 'TRAILING');
                """
            )
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"Failed to fetch active open trades: {e}")
            return []
        finally:
            release_db_connection(conn)

    def update_high_water_mark_and_trailing(self, trade_id: int, current_price: float, atr: float = 0.0, trail_pct: float = 0.02) -> dict:
        """
        Step 3: Updates highest_price (High-Water Mark) and adjusts trailing_stop_price upward.
        Triggers immediate exit signal if price breaches below the calculated trailing_stop_price.
        """
        conn = get_db_connection()
        if not conn:
            return {"action": "NONE"}

        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT pair, direction, entry_price, stop_loss, highest_price, trailing_stop_price, trade_state 
                FROM trade_setups WHERE id = %s;
                """,
                (trade_id,)
            )
            row = cursor.fetchone()
            if not row:
                return {"action": "NONE"}

            pair, direction, entry_price, stop_loss, highest_price, trailing_stop_price, trade_state = row
            entry_price = float(entry_price or 0.0)
            highest_price = float(highest_price or entry_price)
            trailing_stop_price = float(trailing_stop_price or stop_loss or (entry_price * (1 - trail_pct)))

            # 1. Update High-Water Mark if new high is printed
            if current_price > highest_price:
                highest_price = current_price
                
                # Dynamic Trailing Calculation (Percentage or ATR Offset)
                offset = (atr * self.atr_trail_mult) if atr > 0 else (highest_price * trail_pct)
                new_trailing_sl = max(trailing_stop_price, highest_price - offset)

                cursor.execute(
                    """
                    UPDATE trade_setups 
                    SET highest_price = %s, trailing_stop_price = %s, trade_state = 'TRAILING' 
                    WHERE id = %s;
                    """,
                    (highest_price, new_trailing_sl, trade_id)
                )
                conn.commit()
                logger.info(f"[{pair}] High-Water Mark updated: ${highest_price:.5f} | Trailing SL: ${new_trailing_sl:.5f}")
                trailing_stop_price = new_trailing_sl

            # 2. Check for Trailing Stop Loss Breach
            if current_price <= trailing_stop_price:
                return {
                    "action": "CLOSE_TRAILING_SL",
                    "trade_id": trade_id,
                    "pair": pair,
                    "exit_price": current_price,
                    "msg": (
                        f"🚨 <b>TRAILING STOP LOSS BREACHED</b>\n\n"
                        f"<b>Trade ID:</b> <code>#{trade_id}</code>\n"
                        f"<b>Pair:</b> <code>{pair}</code>\n"
                        f"<b>High-Water Mark:</b> ${highest_price:.5f}\n"
                        f"<b>Trailing SL Target:</b> ${trailing_stop_price:.5f}\n"
                        f"<b>Exit Market Price:</b> ${current_price:.5f}"
                    )
                }

        except Exception as e:
            logger.error(f"Error in trailing stop calculation for trade #{trade_id}: {e}")
            if conn:
                conn.rollback()
        finally:
            release_db_connection(conn)

        return {"action": "NONE"}

    def close_trade_in_db(self, trade_id: int, exit_price: float, reason: str = "CLOSED"):
        """Marks active trade record as CLOSED in the database."""
        conn = get_db_connection()
        if not conn:
            return
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE trade_setups 
                SET trade_state = 'CLOSED', status = %s 
                WHERE id = %s;
                """,
                (reason, trade_id)
            )
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to close trade #{trade_id} in DB: {e}")
        finally:
            release_db_connection(conn)