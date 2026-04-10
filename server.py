"""
server.py  — Stock + AI News WhatsApp Bot (Twilio only, Render-ready)
=====================================================================
Routes
  POST /twilio       → Twilio Account 1  — AI News bot
  POST /twilio-stock → Twilio Account 2  — Stock lookup bot

Changes from server_new.py
──────────────────────────
• Meta WhatsApp routes removed entirely
• Scheduler removed (stock bot has no need; news bot uses Render cron job
  or call /run-digest from a cron if needed — see CRON section below)
• Rate-limiter key normalised: strips "whatsapp:" prefix so the same phone
  number is always one bucket regardless of Twilio formatting
• RSI division-by-zero fixed: uses np.where instead of pd.Series.replace
• symbol_finder now validates tickers before accepting short alpha inputs

Required .env
─────────────
# Twilio Account 1 — AI News (used by ainews.py)
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_WHATSAPP_FROM=+14155238886
WHATSAPP_TO_NUMBER=919xxxxxxxxx

# Twilio Account 2 — Stock bot
TWILIO_STOCK_ACCOUNT_SID=
TWILIO_STOCK_AUTH_TOKEN=
TWILIO_STOCK_WHATSAPP_FROM=+14155238887

# OpenRouter (used by ainews.py)
OPENROUTER_API_KEY=
"""

import os
import time
import logging
import threading

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from fastapi import FastAPI, Request, BackgroundTasks, Response, Form
from twilio.rest import Client
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
# TWILIO ACCOUNT 2 — Stock bot
# ---------------------------------------------------------------------------
_STOCK_SID   = os.getenv("TWILIO_STOCK_ACCOUNT_SID", "")
_STOCK_TOKEN = os.getenv("TWILIO_STOCK_AUTH_TOKEN", "")
_STOCK_FROM  = os.getenv("TWILIO_STOCK_WHATSAPP_FROM", "")

_stock_client: Client | None = None
_stock_client_lock = threading.Lock()


def _get_stock_client() -> Client:
    global _stock_client
    with _stock_client_lock:
        if _stock_client is None:
            _stock_client = Client(_STOCK_SID, _STOCK_TOKEN)
    return _stock_client


def _stock_reply(to: str, text: str) -> None:
    """Send a WhatsApp reply via the stock bot's Twilio account."""
    try:
        _get_stock_client().messages.create(
            body=text[:1600],
            from_=f"whatsapp:{_STOCK_FROM}",
            to=to if to.startswith("whatsapp:") else f"whatsapp:{to}",
        )
    except Exception as e:
        logger.error(f"Twilio stock reply error: {e}")


# ---------------------------------------------------------------------------
# RATE LIMITER
# Key is always the bare E.164 number (strips "whatsapp:" if present) so the
# same physical phone is ONE bucket regardless of which Twilio account sends.
# ---------------------------------------------------------------------------
STOCK_COOLDOWN   = 10   # seconds
_lookup_last: dict[str, float] = {}
_lookup_lock = threading.Lock()


def _normalise_phone(phone: str) -> str:
    """Strip 'whatsapp:' prefix so '+919...' and 'whatsapp:+919...' are the same key."""
    return phone.removeprefix("whatsapp:")


def _can_lookup(phone: str) -> bool:
    key = _normalise_phone(phone)
    with _lookup_lock:
        if time.time() - _lookup_last.get(key, 0) < STOCK_COOLDOWN:
            return False
        _lookup_last[key] = time.time()
        return True


# ---------------------------------------------------------------------------
# MARKET HOURS CHECK
# ---------------------------------------------------------------------------
def _is_market_open() -> bool:
    IST = ZoneInfo("Asia/Kolkata")
    now = datetime.now(IST)
    if now.weekday() >= 5:   # Sat / Sun
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
    RSI division-by-zero is handled with np.where (no pandas .replace fragility).
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
        # Use np.where for division guard: avoids pd.Series.replace fragility
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
        f"📉 RSI (14):   {data['rsi']:.2f}\n"
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
    "Send any NSE stock symbol or company name.\n\n"
    "Examples: *RELIANCE*, *Tata Motors*, *INFY*\n\n"
    "You'll get:\n"
    "• Current price & RSI (14)\n"
    "• Volume spike vs 20-day avg\n"
    "• EMA 20 / 50 / 200\n"
    "• Support & Resistance\n"
    "• Signal: 🔥 Reversal | 🚀 Breakout | ⬜ None\n\n"
    "Send *ainews* for latest AI news digest."
)

HELP_TRIGGERS = {"hi", "hello", "help", "menu", "start", "/start"}


# ---------------------------------------------------------------------------
# CORE LOOKUP LOGIC
# ---------------------------------------------------------------------------
def _handle_stock_query(text: str, reply_fn, phone: str) -> None:
    """
    Resolve symbol → fetch data → format → send reply.
    reply_fn(msg) sends the message to the user.
    Runs in a background thread so Twilio gets an instant 200.
    """
    symbol = find_symbol(text)
    if not symbol:
        reply_fn(
            f"❌ Could not find a ticker for *{text}*.\n"
            "Try the NSE symbol directly, e.g. *RELIANCE*."
        )
        return

    reply_fn(f"🔍 Analyzing *{symbol}*...")
    data = _compute_stock_data(symbol)
    if data:
        reply_fn(_format_stock_message(data))
    else:
        reply_fn(
            f"⚠️ *{symbol}*: Not enough historical data (needs {MIN_BARS}+ trading days).\n"
            "Try a different symbol or check back later."
        )


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

# ── Twilio Account 1 — AI News bot ──────────────────────────────────────────
@app.post("/twilio")
async def twilio_ainews(Body: str = Form(...), From: str = Form(...)) -> Response:
    """
    Twilio Account 1 — AI News bot.

    Triggers:
      - 'ainews' / 'news' / 'ai news' / 'latest ai'  → manual digest (6h window)
      - Scheduled hourly digests are fired by Render Cron Job hitting /run-digest
    """
    tl = Body.strip().lower()
    if tl in AINEWS_TRIGGERS:
        _an_get_clients()
        # Manual trigger: use a 6h window so the user always gets something
        threading.Thread(
            target=run_ainews_digest, args=(From, 6), daemon=True
        ).start()
    return Response(content="<Response/>", media_type="text/xml")


# ── Internal cron endpoint — called by Render Cron Job every hour ────────────
@app.post("/run-digest")
async def run_digest_cron() -> dict:
    """
    Hit this endpoint from a Render Cron Job (or any external scheduler)
    every hour.  Runs the digest with a 1h window so only the latest
    stories are sent to the default WHATSAPP_TO_NUMBER.
    """
    threading.Thread(
        target=run_ainews_digest, args=(None, 1), daemon=True
    ).start()
    return {"status": "digest started", "window_hours": 1}


# ── Twilio Account 2 — Stock bot ────────────────────────────────────────────
@app.post("/twilio-stock")
async def twilio_stock(Body: str = Form(...), From: str = Form(...)) -> Response:
    """
    Twilio Account 2 — Individual stock lookup bot.
    Any text → treated as a stock query (symbol or company name).
    """
    text = Body.strip()
    tl   = text.lower()

    # Help / greet
    if tl in HELP_TRIGGERS:
        _stock_reply(From, STOCK_HELP)
        return Response(content="<Response/>", media_type="text/xml")

    # AI news command — redirect user to the correct number
    if tl in AINEWS_TRIGGERS:
        _stock_reply(From, "ℹ️ For AI news, please use the AI News WhatsApp number.")
        return Response(content="<Response/>", media_type="text/xml")

    # Rate limit (normalised phone key so same user can't bypass via channel-switch)
    if not _can_lookup(From):
        _stock_reply(From, "⏳ Please wait a moment before the next lookup.")
        return Response(content="<Response/>", media_type="text/xml")

    # Stock lookup in background thread (Twilio times out at 15 s)
    def _reply(msg_text: str):
        _stock_reply(From, msg_text)

    threading.Thread(
        target=_handle_stock_query, args=(text, _reply, From), daemon=True
    ).start()

    return Response(content="<Response/>", media_type="text/xml")


# ---------------------------------------------------------------------------
# STARTUP
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup() -> None:
    logger.info(
        "Bot started  |  "
        "/twilio (AI News)  |  "
        "/twilio-stock (Stock)  |  "
        "/run-digest (Cron)"
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
