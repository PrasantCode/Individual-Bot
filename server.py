"""
server.py  — Stock + AI News Telegram Bot (Render-ready)
=========================================================
Routes
  POST /telegram       → Telegram webhook — AI News bot
  POST /telegram-stock → Telegram webhook — Stock lookup bot
  POST /run-digest     → Internal cron endpoint (Render Cron Job)

Architecture
────────────
• Two separate Telegram bots (two BotFather tokens):
    TELEGRAM_BOT_TOKEN       — AI News bot
    TELEGRAM_STOCK_BOT_TOKEN — Stock lookup bot
• Default digest target: TELEGRAM_CHAT_ID (your personal/group chat id)
• Webhook setup: configure each bot's webhook URL in BotFather or via
  the Telegram Bot API's setWebhook endpoint.

Required .env
─────────────
# Telegram Bot 1 — AI News (used by ainews.py)
TELEGRAM_BOT_TOKEN=<token from @BotFather>
TELEGRAM_CHAT_ID=<your chat_id for scheduled digests>

# Telegram Bot 2 — Stock bot
TELEGRAM_STOCK_BOT_TOKEN=<token from @BotFather>

# OpenRouter (used by ainews.py)
OPENROUTER_API_KEY=

How to get your chat_id
───────────────────────
1. Message your bot
2. Open: https://api.telegram.org/bot<TOKEN>/getUpdates
3. Copy the "id" field from message.chat
"""

import os
import time
import logging
import threading

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from fastapi import FastAPI, Request, Response
from zoneinfo import ZoneInfo
from datetime import datetime
from dotenv import load_dotenv

import ainews
from ainews import run_ainews_digest, AINEWS_TRIGGERS, _an_get_clients
from symbol_finder import find_symbol

load_dotenv()
app = FastAPI()

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("stockbot")

# ---------------------------------------------------------------------------
# TELEGRAM — Stock bot (Bot 2)
# ---------------------------------------------------------------------------
_STOCK_BOT_TOKEN: str = os.getenv("TELEGRAM_STOCK_BOT_TOKEN", "")


def _stock_reply(chat_id: int | str, text: str) -> None:
    """Send a Telegram reply via the stock bot."""
    url = f"https://api.telegram.org/bot{_STOCK_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id":    chat_id,
        "text":       text[:4096],
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=15)
        if not r.ok:
            logger.error(f"Stock bot Telegram error: {r.status_code} {r.text[:200]}")
    except Exception as e:
        logger.error(f"Stock bot Telegram request failed: {e}")


# ---------------------------------------------------------------------------
# RATE LIMITER
# Key is the Telegram user_id (always a stable integer).
# ---------------------------------------------------------------------------
STOCK_COOLDOWN   = 10   # seconds
_lookup_last: dict[int, float] = {}
_lookup_lock = threading.Lock()


def _can_lookup(user_id: int) -> bool:
    with _lookup_lock:
        if time.time() - _lookup_last.get(user_id, 0) < STOCK_COOLDOWN:
            return False
        _lookup_last[user_id] = time.time()
        return True


# ---------------------------------------------------------------------------
# MARKET HOURS CHECK
# ---------------------------------------------------------------------------
def _is_market_open() -> bool:
    IST = ZoneInfo("Asia/Kolkata")
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    t = now.time()
    from datetime import time as dtime
    return dtime(9, 15) <= t <= dtime(15, 30)


# ---------------------------------------------------------------------------
# TECHNICAL ANALYSIS  (yfinance, on-demand per request)
# ---------------------------------------------------------------------------
MIN_BARS      = 60
RSI_OVERSOLD  = 25
VOL_REV_SPIKE = 30
VOL_BRK_SPIKE = 50


def _compute_stock_data(symbol: str) -> dict | None:
    """
    Download 1-year daily OHLCV via yfinance and compute indicators.
    RSI division-by-zero is handled with np.where.
    """
    try:
        clean  = symbol.replace(".NS", "").replace(".BO", "").replace("-EQ", "").upper()
        ticker = f"{clean}.NS"

        df = yf.download(ticker, period="1y", interval="1d",
                         progress=False, auto_adjust=True)
        if df is None or len(df) < MIN_BARS:
            return None

        close  = df["Close"].squeeze()
        volume = df["Volume"].squeeze()
        high   = df["High"].squeeze()
        low    = df["Low"].squeeze()

        price       = float(close.iloc[-1])
        current_vol = int(volume.iloc[-1])

        # RSI — 14-period Wilder smoothing via EWM
        delta     = close.diff()
        gain      = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
        loss      = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
        loss_arr  = loss.to_numpy()
        gain_arr  = gain.to_numpy()
        rs        = np.where(loss_arr == 0, np.inf, gain_arr / loss_arr)
        rsi_arr   = 100.0 - (100.0 / (1.0 + rs))
        rsi       = float(rsi_arr[-1])

        # Volume spike: 5-day vs 20-day average
        v5        = float(volume.rolling(5).mean().iloc[-1])
        v20       = float(volume.rolling(20).mean().iloc[-1])
        vol_ratio = ((v5 / v20) - 1) * 100 if v20 else 0.0

        # EMAs
        ema20  = float(close.ewm(span=20,  adjust=False).mean().iloc[-1])
        ema50  = float(close.ewm(span=50,  adjust=False).mean().iloc[-1])
        ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])

        # Support / Resistance — last 20 sessions, excluding today
        recent_resistance = float(high.iloc[-21:-1].max())
        recent_support    = float(low.iloc[-21:-1].min())
        is_breakout       = price > recent_resistance

        # Signal classification
        if rsi < RSI_OVERSOLD and vol_ratio > VOL_REV_SPIKE:
            signal = "REVERSAL"
        elif is_breakout and vol_ratio > VOL_BRK_SPIKE:
            signal = "BREAKOUT"
        else:
            signal = "NO SIGNAL"

        return {
            "ticker":     clean,
            "signal":     signal,
            "price":      round(price, 2),
            "volume":     current_vol,
            "rsi":        round(rsi, 2),
            "vol_spike":  round(vol_ratio, 1),
            "ema20":      round(ema20, 2),
            "ema50":      round(ema50, 2),
            "ema200":     round(ema200, 2),
            "vs_ema20":   "Above" if price > ema20  else "Below",
            "vs_ema50":   "Above" if price > ema50  else "Below",
            "vs_ema200":  "Above" if price > ema200 else "Below",
            "resistance": round(recent_resistance, 2),
            "support":    round(recent_support, 2),
            "vs_resist":  "Above" if is_breakout else "Below",
        }
    except Exception as e:
        logger.error(f"_compute_stock_data error for {symbol}: {e}")
        return None


def _format_stock_message(data: dict) -> str:
    sig = data["signal"]
    if sig == "REVERSAL":
        status = "🔥 *REVERSAL SIGNAL*"
    elif sig == "BREAKOUT":
        status = "🚀 *BREAKOUT DETECTED*"
    else:
        status = "⬜ *NO SIGNAL*"

    ticker      = data["ticker"]
    is_breakout = data["vs_resist"] == "Above"
    return (
        f"{status}: {ticker}\n"
        f"━━━━━━━━━━━━━━━\n"
        f"💰 Price:      ₹{data['price']:.2f}\n"
        f"🔊 Volume:     {data['volume']:,}\n"
        f"📉 RSI \\(14\\):   {data['rsi']:.2f}\n"
        f"📊 Vol Spike:  {data['vol_spike']:+.1f}%\n"
        f"━━━━━━━━━━━━━━━\n"
        f"📈 EMA 20:     ₹{data['ema20']:.2f}  ({data['vs_ema20'].lower()})\n"
        f"📈 EMA 50:     ₹{data['ema50']:.2f}  ({data['vs_ema50'].lower()})\n"
        f"📈 EMA 200:    ₹{data['ema200']:.2f}  ({data['vs_ema200'].lower()})\n"
        f"━━━━━━━━━━━━━━━\n"
        f"🧱 Resistance: ₹{data['resistance']:.2f}\n"
        f"🛡️ Support:    ₹{data['support']:.2f}\n"
        f"{'✅ Above Resistance' if is_breakout else '⚠️ Below Resistance'}\n"
        f"━━━━━━━━━━━━━━━"
    )


# ---------------------------------------------------------------------------
# HELP MESSAGE
# ---------------------------------------------------------------------------
STOCK_HELP = (
    "📈 *Stock Bot*\n"
    "━━━━━━━━━━━━━━━\n"
    "Send any NSE stock symbol or company name\\.\n\n"
    "Examples: *RELIANCE*, *Tata Motors*, *INFY*\n\n"
    "You'll get:\n"
    "• Current price & RSI \\(14\\)\n"
    "• Volume spike vs 20\\-day avg\n"
    "• EMA 20 / 50 / 200\n"
    "• Support & Resistance\n"
    "• Signal: 🔥 Reversal | 🚀 Breakout | ⬜ None\n\n"
    "Send *ainews* for latest AI news digest\\."
)

HELP_TRIGGERS = {"hi", "hello", "help", "menu", "start", "/start"}


# ---------------------------------------------------------------------------
# TELEGRAM WEBHOOK HELPERS
# ---------------------------------------------------------------------------
def _extract_message(update: dict) -> tuple[str, int] | tuple[None, None]:
    """
    Extract (text, chat_id) from a Telegram Update object.
    Handles regular messages and edited messages.
    """
    msg = update.get("message") or update.get("edited_message")
    if not msg:
        return None, None
    text    = msg.get("text", "").strip()
    chat_id = msg.get("chat", {}).get("id")
    return text, chat_id


def _extract_user_id(update: dict) -> int | None:
    msg = update.get("message") or update.get("edited_message")
    if not msg:
        return None
    return msg.get("from", {}).get("id")


# ---------------------------------------------------------------------------
# CORE STOCK LOOKUP LOGIC
# ---------------------------------------------------------------------------
def _handle_stock_query(text: str, chat_id: int) -> None:
    """
    Resolve symbol → fetch data → format → send reply.
    Runs in a background thread so webhook returns instantly.
    """
    symbol = find_symbol(text)
    if not symbol:
        _stock_reply(
            chat_id,
            f"❌ Could not find a ticker for *{text}*\\.\n"
            "Try the NSE symbol directly, e\\.g\\. *RELIANCE*\\."
        )
        return

    _stock_reply(chat_id, f"🔍 Analyzing *{symbol}*\\.\\.\\.")
    data = _compute_stock_data(symbol)
    if data:
        _stock_reply(chat_id, _format_stock_message(data))
    else:
        _stock_reply(
            chat_id,
            f"⚠️ *{symbol}*: Not enough historical data \\(needs {MIN_BARS}\\+ trading days\\)\\.\n"
            "Try a different symbol or check back later\\."
        )


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

# ── Telegram Bot 1 — AI News bot ────────────────────────────────────────────
@app.post("/telegram")
async def telegram_ainews(request: Request) -> Response:
    """
    Telegram Bot 1 webhook — AI News bot.

    Triggers:
      - 'ainews' / 'news' / 'ai news' / 'latest ai'  → manual digest (6h window)
      - Scheduled hourly digests are fired by Render Cron Job hitting /run-digest
    """
    try:
        update = await request.json()
    except Exception:
        return Response(content="ok")

    text, chat_id = _extract_message(update)
    if not text or not chat_id:
        return Response(content="ok")

    if text.lower() in AINEWS_TRIGGERS:
        _an_get_clients()
        threading.Thread(
            target=run_ainews_digest, args=(chat_id, 6), daemon=True
        ).start()

    return Response(content="ok")


# ── Internal cron endpoint — called by Render Cron Job every hour ─────────
@app.post("/run-digest")
async def run_digest_cron() -> dict:
    """
    Hit this endpoint from a Render Cron Job every hour.
    Runs the digest with a 1h window and sends to the default TELEGRAM_CHAT_ID.
    """
    threading.Thread(
        target=run_ainews_digest, args=(None, 1), daemon=True
    ).start()
    return {"status": "digest started", "window_hours": 1}


# ── Telegram Bot 2 — Stock bot ───────────────────────────────────────────────
@app.post("/telegram-stock")
async def telegram_stock(request: Request) -> Response:
    """
    Telegram Bot 2 webhook — Individual stock lookup bot.
    Any text → treated as a stock query (symbol or company name).
    """
    try:
        update = await request.json()
    except Exception:
        return Response(content="ok")

    text, chat_id = _extract_message(update)
    user_id       = _extract_user_id(update)

    if not text or not chat_id:
        return Response(content="ok")

    tl = text.lower()

    # Help / greet
    if tl in HELP_TRIGGERS:
        _stock_reply(chat_id, STOCK_HELP)
        return Response(content="ok")

    # AI news command — redirect user to the correct bot
    if tl in AINEWS_TRIGGERS:
        _stock_reply(chat_id, "ℹ️ For AI news, please use the AI News bot\\.")
        return Response(content="ok")

    # Rate limit by Telegram user_id
    uid = user_id or chat_id
    if not _can_lookup(uid):
        _stock_reply(chat_id, "⏳ Please wait a moment before the next lookup\\.")
        return Response(content="ok")

    # Stock lookup in background thread (must return quickly)
    threading.Thread(
        target=_handle_stock_query, args=(text, chat_id), daemon=True
    ).start()

    return Response(content="ok")


# ---------------------------------------------------------------------------
# WEBHOOK REGISTRATION HELPER (run once at deploy time)
# ---------------------------------------------------------------------------
@app.post("/set-webhooks")
async def set_webhooks(request: Request) -> dict:
    """
    Convenience endpoint to register both bot webhooks with Telegram.
    POST with JSON body: {"base_url": "https://your-app.onrender.com"}
    Call this once after deploying.
    """
    body     = await request.json()
    base_url = body.get("base_url", "").rstrip("/")
    results  = {}

    for token, path in [
        (os.getenv("TELEGRAM_BOT_TOKEN", ""),       "/telegram"),
        (os.getenv("TELEGRAM_STOCK_BOT_TOKEN", ""), "/telegram-stock"),
    ]:
        if not token:
            results[path] = "SKIPPED (no token)"
            continue
        url = f"https://api.telegram.org/bot{token}/setWebhook"
        r   = requests.post(url, json={"url": f"{base_url}{path}"})
        results[path] = r.json()

    return results


# ---------------------------------------------------------------------------
# STARTUP
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup() -> None:
    logger.info(
        "Bot started  |  "
        "/telegram (AI News)  |  "
        "/telegram-stock (Stock)  |  "
        "/run-digest (Cron)  |  "
        "/set-webhooks (Setup)"
    )


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,
    )
