import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from common import (
    calculate_pnl,
    ensure_schema_updated,
    get_db_connection,
    send_telegram_notification,
)

load_dotenv()
app = FastAPI(title="Trade Settlement Webhook Engine")

ensure_schema_updated()

class SettlementPayload(BaseModel):
    trade_id: int
    exit_price: float

@app.post("/settle-trade")
def settle_trade(payload: SettlementPayload):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
                SELECT id, pair, direction, entry_price, position_size, account_balance, status 
                FROM trade_setups 
                WHERE id = %s;
            """,
            (payload.trade_id,),
        )
        trade = cursor.fetchone()

        if not trade:
            raise HTTPException(status_code=404, detail="Trade not found.")

        (
            trade_id,
            pair,
            direction,
            entry_price,
            position_size,
            account_balance,
            status,
        ) = trade

        if status == "CLOSED":
            return {
                "status": "ignored",
                "message": f"Trade {trade_id} is already closed.",
            }

        default_balance = float(os.getenv("ACCOUNT_BALANCE", 100.0))
        balance = float(account_balance) if account_balance else default_balance

        pnl_usd, pnl_pct, outcome = calculate_pnl(
            direction,
            float(entry_price),
            payload.exit_price,
            float(position_size),
            balance,
        )

        cursor.execute(
            """
                UPDATE trade_setups 
                SET exit_price = %s, pnl_usd = %s, pnl_pct = %s, outcome = %s, status = 'CLOSED', trade_state = 'CLOSED', closed_at = CURRENT_TIMESTAMP
                WHERE id = %s;
            """,
            (payload.exit_price, pnl_usd, pnl_pct, outcome, trade_id),
        )
        conn.commit()

        emoji = "🟢" if outcome == "WIN" else "🔴"
        msg = (
            f"<b>{emoji} TRADE SETTLED VIA WEBHOOK</b>\n\n"
            f"<b>Trade ID:</b> <code>#{trade_id}</code>\n"
            f"<b>Pair:</b> <code>{pair}</code>\n"
            f"<b>Exit Price:</b> ${payload.exit_price:,.2f}\n"
            f"<b>PnL ($):</b> ${pnl_usd:,.2f}\n"
            f"<b>PnL (%):</b> {pnl_pct:.2f}%\n"
            f"<b>Outcome:</b> {outcome}"
        )
        send_telegram_notification(msg)

        return {
            "status": "success",
            "trade_id": trade_id,
            "pnl_usd": pnl_usd,
            "pnl_pct": pnl_pct,
            "outcome": outcome,
        }
    finally:
        cursor.close()
        conn.close()