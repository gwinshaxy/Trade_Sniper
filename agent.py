import pandas as pd

class DynamicTradeManager:
    def __init__(self, spread_buffer_pct=0.0005, tema_offset_pct=0.001):
        self.spread_buffer_pct = spread_buffer_pct
        self.tema_offset_pct = tema_offset_pct

    def process_trade(self, trade_row: pd.Series, latest_candle: pd.Series) -> dict:
        trade_id = trade_row['id']
        direction = trade_row['direction']
        entry = float(trade_row['entry_price'])
        curr_sl = float(trade_row['stop_loss'])
        curr_tp = float(trade_row['take_profit'])
        state = trade_row.get('trade_state', 'OPEN')
        
        curr_price = float(latest_candle['close'])
        tema = float(latest_candle.get('200_TEMA', latest_candle.get('tema', curr_price)))
        initial_risk = abs(entry - curr_sl)

        # -------------------------------------------------------------
        # 1. STOP LOSS BREACH CHECK (HIGHEST PRIORITY)
        # -------------------------------------------------------------
        sl_hit = (curr_price <= curr_sl) if direction == 'LONG' else (curr_price >= curr_sl)
        if sl_hit:
            return {
                "action": "CLOSE_SL",
                "trade_id": trade_id,
                "exit_price": curr_sl,
                "msg": f"🔴 <b>STOP LOSS HIT</b>\n\nTrade #{trade_id} hit Stop Loss at ${curr_sl:.5f}!"
            }

        # -------------------------------------------------------------
        # 2. TAKE PROFIT BREACH CHECK
        # -------------------------------------------------------------
        tp_hit = (curr_price >= curr_tp) if direction == 'LONG' else (curr_price <= curr_tp)
        if tp_hit:
            return {
                "action": "CLOSE_TP",
                "trade_id": trade_id,
                "exit_price": curr_tp,
                "msg": f"🟢 <b>TAKE PROFIT HIT</b>\n\nTrade #{trade_id} hit Take Profit at ${curr_tp:.5f}!"
            }

        # -------------------------------------------------------------
        # 3. BREAK-EVEN CHECK (At 1:1 R:R)
        # -------------------------------------------------------------
        if state == 'OPEN':
            tp1_target = entry + initial_risk if direction == 'LONG' else entry - initial_risk
            be_triggered = (curr_price >= tp1_target) if direction == 'LONG' else (curr_price <= tp1_target)

            if be_triggered:
                be_price = entry * (1 + self.spread_buffer_pct) if direction == 'LONG' else entry * (1 - self.spread_buffer_pct)
                return {
                    "action": "UPDATE_SL",
                    "trade_id": trade_id,
                    "new_sl": round(be_price, 5),
                    "new_state": "BE_LOCKED",
                    "msg": f"🎯 Trade #{trade_id} hit 1:1 R:R. Moving Stop Loss to Break-Even (${be_price:.5f})."
                }

        # -------------------------------------------------------------
        # 4. DYNAMIC TRAILING STOP ALONG TEMA
        # -------------------------------------------------------------
        elif state in ['BE_LOCKED', 'TRAILING']:
            if direction == 'LONG':
                proposed_sl = tema * (1 - self.tema_offset_pct)
                if proposed_sl > curr_sl and curr_price > tema:
                    return {
                        "action": "UPDATE_SL",
                        "trade_id": trade_id,
                        "new_sl": round(proposed_sl, 5),
                        "new_state": "TRAILING",
                        "msg": f"📈 Trade #{trade_id} Trailing SL updated to ${proposed_sl:.5f} along TEMA (${tema:.5f})."
                    }
            elif direction == 'SHORT':
                proposed_sl = tema * (1 + self.tema_offset_pct)
                if proposed_sl < curr_sl and curr_price < tema:
                    return {
                        "action": "UPDATE_SL",
                        "trade_id": trade_id,
                        "new_sl": round(proposed_sl, 5),
                        "new_state": "TRAILING",
                        "msg": f"📉 Trade #{trade_id} Trailing SL updated to ${proposed_sl:.5f} along TEMA (${tema:.5f})."
                    }

        return {"action": "NONE"}