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
    Fetch live quote + 1-year daily OHLCV and compute indicators.

    Live price / day OHLC come from yf.Ticker.fast_info (real-time).
    EOD technicals (RSI, EMAs, vol spike, support/resistance) are computed
    from the daily history — always based on the previous session's close
    so they are stable throughout the trading day.

    Returns a dict with a 'mode' key: "LIVE" or "POST".
    """
    try:
        clean  = symbol.replace(".NS", "").replace(".BO", "").replace("-EQ", "").upper()
        ticker_str = f"{clean}.NS"
        tk = yf.Ticker(ticker_str)

        # ── 1. Live / real-time quote ────────────────────────────────────────
        fi          = tk.fast_info          # lightweight, no full download
        ltp         = float(fi.get("last_price") or fi.get("lastPrice") or 0)
        prev_close  = float(fi.get("previous_close") or fi.get("previousClose") or 0)
        day_open    = float(fi.get("open") or 0)
        day_high    = float(fi.get("day_high") or fi.get("dayHigh") or 0)
        day_low     = float(fi.get("day_low")  or fi.get("dayLow")  or 0)

        # Sector from info (slower but cached after first call)
        try:
            info   = tk.info
            sector = info.get("sector") or info.get("sectorDisp") or ""
        except Exception:
            sector = ""

        # Timestamp of last trade
        IST = ZoneInfo("Asia/Kolkata")
        now_ist = datetime.now(IST)
        quote_time = now_ist.strftime("%-d %b %Y  %H:%M IST")

        # Determine mode
        is_live = _is_market_open()
        mode    = "LIVE" if is_live else "POST"

        if ltp == 0:
            return None   # ticker returned nothing useful

        # Day change vs prev close
        day_change     = ltp - prev_close
        day_change_pct = (day_change / prev_close * 100) if prev_close else 0.0

        # ── 2. Historical daily data for EOD technicals ──────────────────────
        df = yf.download(ticker_str, period="1y", interval="1d",
                         progress=False, auto_adjust=True)
        if df is None or len(df) < MIN_BARS:
            return None

        close  = df["Close"].squeeze()
        volume = df["Volume"].squeeze()
        high   = df["High"].squeeze()
        low    = df["Low"].squeeze()

        # Use previous session (iloc[-2] if market open, iloc[-1] if closed)
        eod_idx        = -2 if is_live else -1
        eod_close      = float(close.iloc[eod_idx])
        eod_volume     = int(volume.iloc[eod_idx])

        # RSI on full history (14-period Wilder via EWM)
        delta    = close.diff()
        gain     = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
        loss     = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
        rs       = np.where(loss.to_numpy() == 0, np.inf, gain.to_numpy() / loss.to_numpy())
        rsi      = float((100.0 - 100.0 / (1.0 + rs))[eod_idx])

        # Volume spike: 5-day vs 20-day (at EOD row)
        v5       = float(volume.rolling(5).mean().iloc[eod_idx])
        v20      = float(volume.rolling(20).mean().iloc[eod_idx])
        vol_ratio = ((v5 / v20) - 1) * 100 if v20 else 0.0

        # EMAs at EOD row
        ema20  = float(close.ewm(span=20,  adjust=False).mean().iloc[eod_idx])
        ema50  = float(close.ewm(span=50,  adjust=False).mean().iloc[eod_idx])
        ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[eod_idx])

        # Support / Resistance — 20 sessions before EOD row
        res_slice = high.iloc[eod_idx - 20 : eod_idx]
        sup_slice = low.iloc[eod_idx - 20 : eod_idx]
        recent_resistance = float(res_slice.max())
        recent_support    = float(sup_slice.min())
        is_breakout       = eod_close > recent_resistance

        # Signal classification (uses EOD close for consistency)
        if rsi < RSI_OVERSOLD and vol_ratio > VOL_REV_SPIKE:
            signal = "REVERSAL"
        elif is_breakout and vol_ratio > VOL_BRK_SPIKE:
            signal = "BREAKOUT"
        else:
            signal = "NO SIGNAL"

        return {
            "mode":           mode,
            "ticker":         clean,
            "signal":         signal,
            "quote_time":     quote_time,
            # Live price fields
            "ltp":            round(ltp, 2),
            "prev_close":     round(prev_close, 2),
            "day_change":     round(day_change, 2),
            "day_change_pct": round(day_change_pct, 2),
            "day_open":       round(day_open, 2),
            "day_high":       round(day_high, 2),
            "day_low":        round(day_low, 2),
            # EOD technicals
            "sector":         sector,
            "eod_volume":     eod_volume,
            "rsi":            round(rsi, 2),
            "vol_spike":      round(vol_ratio, 1),
            "ema20":          round(ema20, 2),
            "ema50":          round(ema50, 2),
            "ema200":         round(ema200, 2),
            "vs_ema20":       "above" if eod_close > ema20  else "below",
            "vs_ema50":       "above" if eod_close > ema50  else "below",
            "vs_ema200":      "above" if eod_close > ema200 else "below",
            "resistance":     round(recent_resistance, 2),
            "support":        round(recent_support, 2),
        }
    except Exception as e:
        logger.error(f"_compute_stock_data error for {symbol}: {e}")
        return None


def _fmt_chg(change: float, pct: float) -> str:
    """Format ₹change (+pct%) with sign."""
    sign = "+" if change >= 0 else ""
    return f"₹{sign}{change:.2f}  ({sign}{pct:.2f}%)"


def _format_stock_message(data: dict) -> str:
    """
    Render the Telegram message to match the LIVE / Post-market style
    shown in the screenshots.
    """
    mode    = data["mode"]
    ticker  = data["ticker"]
    signal  = data["signal"]

    # ── Header dot + mode label ──────────────────────────────────────────────
    if mode == "LIVE":
        dot   = "🟢"
        label = "LIVE"
    else:
        dot   = "🔵"
        label = "Post\\-market"

    # ── Signal badge (appended after header) ────────────────────────────────
    if signal == "REVERSAL":
        sig_line = "🔥 *REVERSAL SIGNAL*\n"
    elif signal == "BREAKOUT":
        sig_line = "🚀 *BREAKOUT DETECTED*\n"
    else:
        sig_line = ""

    # ── Change formatting ────────────────────────────────────────────────────
    chg     = data["day_change"]
    chg_pct = data["day_change_pct"]
    chg_str = _fmt_chg(chg, chg_pct)
    chg_ico = "🔺" if chg >= 0 else "🔻"

    # ── Live price block ─────────────────────────────────────────────────────
    price_block = (
        f"⏰  {data['quote_time']}\n"
        f"💰  LTP:          ₹{data['ltp']:.2f}\n"
        f"{chg_ico}  Change:     {chg_str}\n"
        f"📊  Prev Close: ₹{data['prev_close']:.2f}\n"
    )
    if mode == "LIVE":
        price_block += (
            f"🔓  Open:        ₹{data['day_open']:.2f}\n"
            f"📈  Day High:   ₹{data['day_high']:.2f}\n"
            f"📉  Day Low:    ₹{data['day_low']:.2f}\n"
        )

    # ── EOD technicals block ─────────────────────────────────────────────────
    eod_label = "EOD Technicals" if mode == "LIVE" else "EOD Technicals"
    sector_line = f"🏷️  Sector:      {data['sector']}\n" if data.get("sector") else ""
    tech_block = (
        f"📐  *{eod_label}*\n"
        f"{sector_line}"
        f"🌐  EOD Volume: {data['eod_volume']:,}\n"
        f"📉  RSI (14):   {data['rsi']:.2f}\n"
        f"📊  Vol Spike:  {data['vol_spike']:+.1f}%\n"
    )

    # ── EMA block ────────────────────────────────────────────────────────────
    ema_block = (
        f"📗  EMA 20:     ₹{data['ema20']:.2f}  ({data['vs_ema20']})\n"
        f"📗  EMA 50:     ₹{data['ema50']:.2f}  ({data['vs_ema50']})\n"
        f"📗  EMA 200:   ₹{data['ema200']:.2f}  ({data['vs_ema200']})\n"
    )

    # ── Support / Resistance block ───────────────────────────────────────────
    sr_block = (
        f"🧱  Resistance: ₹{data['resistance']:.2f}\n"
        f"🛡️  Support:    ₹{data['support']:.2f}\n"
    )

    SEP = "━━━━━━━━━━━━━━━\n"

    return (
        f"{dot} *{label}: {ticker}*\n"
        f"{sig_line}"
        f"{SEP}"
        f"{price_block}"
        f"{SEP}"
        f"{tech_block}"
        f"{SEP}"
        f"{ema_block}"
        f"{SEP}"
        f"{sr_block}"
        f"━━━━━━━━━━━━━━━"
    )


# ---------------------------------------------------------------------------
# HELP MESSAGE
# ---------------------------------------------------------------------------
STOCK_HELP = (
    "📈 *Stock Bot*\n"
    "━━━━━━━━━━━━━━━\n"
    "Send any NSE stock symbol or company name\\.\n\n"
    "Examples: *RELIANCE*, *Tata Motors*, *INFY*, *Ola*, *BSE*\n\n"
    "You'll get:\n"
    "• Current price & day change %\n"
    "• RSI \\(14\\) & Volume spike vs 20\\-day avg\n"
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
