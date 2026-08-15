import os
import uvicorn
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


# Render Health Check Endpoints
@app.get("/")
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "webhook_engine"}


@app.post("/settle-trade")
def settle_trade(payload: SettlementPayload):
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed.")
    
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT id, pair, direction, entry_price, position_size, account_balance, status FROM trade_setups WHERE id = %s;",
            (payload.trade_id,)
        )
        trade = cursor.fetchone()
        if not trade:
            raise HTTPException(status_code=404, detail="Trade not found.")

        trade_id, pair, direction, entry_price, position_size, account_balance, status = trade
        if status == "CLOSED":
            return {"status": "ignored", "message": f"Trade {trade_id} is already closed."}

        pnl_usd, pnl_pct, outcome = calculate_pnl(
            direction, float(entry_price), payload.exit_price, float(position_size), float(account_balance or 10000.0)
        )

        cursor.execute(
            "UPDATE trade_setups SET exit_price = %s, pnl_usd = %s, pnl_pct = %s, outcome = %s, status = 'CLOSED', trade_state = 'CLOSED', closed_at = CURRENT_TIMESTAMP WHERE id = %s;",
            (payload.exit_price, pnl_usd, pnl_pct, outcome, trade_id)
        )
        conn.commit()
        
        emoji = "🟢" if outcome == "WIN" else "🔴"
        send_telegram_notification(
            f"<b>{emoji} TRADE SETTLED VIA WEBHOOK</b>\n\n"
            f"<b>Trade ID:</b> <code>#{trade_id}</code>\n"
            f"<b>Pair:</b> <code>{pair}</code>\n"
            f"<b>Exit Price:</b> ${payload.exit_price:.5f}\n"
            f"<b>PnL:</b> ${pnl_usd:,.2f} ({outcome})"
        )
        return {"status": "success", "trade_id": trade_id, "pnl_usd": pnl_usd, "pnl_pct": pnl_pct, "outcome": outcome}
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    # Standardize port resolution to prevent port collisions on Render
    webhook_port = int(os.getenv("WEBHOOK_PORT") or os.getenv("PORT") or 8080)
    uvicorn.run("webhook_engine:app", host="0.0.0.0", port=webhook_port, reload=False)