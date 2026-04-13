import os
import time
import json
import logging
import re
import hashlib
import feedparser
import requests
from openai import OpenAI
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta, timezone
from pathlib import Path

_AINEWS_TZ         = ZoneInfo("Asia/Kolkata")
_AINEWS_SENT_STORE = Path("sent_news.json")
_AINEWS_MAX_DAYS   = 3
AINEWS_TRIGGERS    = {"ainews", "news", "ai news", "latest ai", "ailatest"}

_ainews_client    = None   # OpenRouter / OpenAI client
_tg_bot_token     = None   # Telegram Bot token (Account 1 — AI News)
_tg_chat_id       = None   # Default chat_id to send scheduled digests to

def _an_get_clients():
    """Lazy-init AI news clients — only runs when first digest is triggered."""
    global _ainews_client, _tg_bot_token, _tg_chat_id
    if _ainews_client is None:
        _ainews_client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    if _tg_bot_token is None:
        _tg_bot_token = os.environ["TELEGRAM_BOT_TOKEN"]
        _tg_chat_id   = os.environ["TELEGRAM_CHAT_ID"]


def _tg_send_message(chat_id: str | int, text: str, parse_mode: str = "Markdown") -> bool:
    """
    Send a Telegram message via the Bot API (Account 1 — AI News bot).
    Returns True on success, False on failure.
    """
    _an_get_clients()
    url = f"https://api.telegram.org/bot{_tg_bot_token}/sendMessage"
    payload = {
        "chat_id":    chat_id,
        "text":       text[:4096],          # Telegram hard limit
        "parse_mode": parse_mode,
        "disable_web_page_preview": False,
    }
    try:
        r = requests.post(url, json=payload, timeout=15)
        if not r.ok:
            ainews_log.error(f"Telegram send error: {r.status_code} {r.text[:200]}")
        return r.ok
    except Exception as e:
        ainews_log.error(f"Telegram request failed: {e}")
        return False


# ---------------------------------------------------------------------------
# RSS FEEDS
# ---------------------------------------------------------------------------
_AINEWS_PRIMARY_FEEDS = [
    # --- Tech media (AI-specific feeds) ---
    "https://techcrunch.com/category/artificial-intelligence/feed/",
    "https://venturebeat.com/category/ai/feed/",
    "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
    "https://www.artificialintelligence-news.com/feed/",
    "https://thenextweb.com/neural/feed/",      
    "https://www.marktechpost.com/feed/",
    # --- Official lab blogs ---
    "https://openai.com/blog/rss.xml",
    "https://raw.githubusercontent.com/taobojlen/anthropic-rss-feed/main/anthropic_news_rss.xml",
    "https://deepmind.google/blog/rss.xml",
    "https://research.facebook.com/feed/",
    "https://engineering.fb.com/category/ml-applications/feed/",
    "https://blogs.microsoft.com/ai/feed/",
    "https://huggingface.co/blog/feed.xml",    
    # --- Research / academia ---
    "https://bair.berkeley.edu/blog/feed.xml",
    "https://research.google/blog/rss",
    "https://www.technologyreview.com/feed/",
    "https://www.oreilly.com/radar/topics/ai-ml/feed/index.xml",
    # --- News wire ---
    "https://news.google.com/rss/search?q=when:24h+allinurl:reuters.com+AI&ceid=US:en&hl=en-US&gl=US",
    "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "https://www.theguardian.com/technology/artificialintelligenceai/rss",
    # --- Community / aggregators ---
    "https://news.ycombinator.com/rss",
    "https://tldr.tech/rss/ai",
    "https://aiweekly.co/issues.rss",
    "https://feeds.arstechnica.com/arstechnica/technology-lab",
    "https://www.oreilly.com/radar/feed/index.xml",
    "https://www.wired.com/feed/tag/ai/latest/rss",
    "https://www.zdnet.com/topic/artificial-intelligence/rss.xml",
    "https://arxiv.org/rss/cs.AI",
    "https://aws.amazon.com/blogs/machine-learning/feed/",
    "https://developer.nvidia.com/blog/feed",
    "https://news.mit.edu/rss/topic/artificial-intelligence2",
]

_AINEWS_VERIFY_FEEDS = [
    "https://feeds.arstechnica.com/arstechnica/technology-lab",
    "https://www.wired.com/feed/category/ai/latest/rss",
    "https://www.technologyreview.com/feed/",
    "https://www.zdnet.com/topic/artificial-intelligence/rss.xml",
    "https://news.ycombinator.com/rss",
    "https://feeds.reuters.com/reuters/technology",
    "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "https://thenextweb.com/neural/feed/",
    "https://www.marktechpost.com/feed/",
]

_AINEWS_KEYWORDS = [
    # Core terms
    "ai","artificial intelligence","machine learning","deep learning","large language model",
    "llm","gpt","claude","gemini","llama","transformer","neural network","generative ai",
    "foundation model","language model",
    # Companies / labs
    "openai","anthropic","deepmind","google deepmind","meta ai","mistral","xai","grok",
    "cohere","hugging face","stability ai","inflection","perplexity","groq","together ai",
    "nvidia ai","microsoft ai","amazon bedrock","apple intelligence",
    # Products
    "copilot","chatgpt","sora","dall-e","stable diffusion","midjourney",
    "cursor","devin","github copilot",
    # Technical
    "ai platform","ai model","ai launch","ai release","ai update","ai tool","ai agent",
    "ai benchmark","ai safety","ai regulation","inference","fine-tuning","rag",
    "multimodal","ai chip","gpu","tpu","ai infrastructure","ai startup","ai funding",
    "ai investment","ai research","reasoning model","agentic","autonomous ai",
    "ai assistant","ai search","ai coding","ai image","ai video","text to image",
    "text to video","voice ai","speech recognition","computer vision",
]

ainews_log = logging.getLogger("ainews")


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def _an_sanitize(text: str) -> str:
    """Strip control characters that break JSON parsing."""
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"[\x00-\x1f\x7f]", "", text)
    return text.strip()

def _an_story_key(title: str) -> str:
    return hashlib.md5(title.lower().strip().encode()).hexdigest()[:12]

def _an_load_sent() -> dict:
    if _AINEWS_SENT_STORE.exists():
        try:
            return json.loads(_AINEWS_SENT_STORE.read_text())
        except:
            return {}
    return {}

def _an_save_sent(store: dict):
    cutoff = (datetime.now() - timedelta(days=_AINEWS_MAX_DAYS)).isoformat()
    pruned = {k: v for k, v in store.items() if v.get("sent_at", "") > cutoff}
    _AINEWS_SENT_STORE.write_text(json.dumps(pruned, indent=2))

def _an_fetch_feed(url: str, max_age_hours: int = 1) -> list[dict]:
    try:
        feed = feedparser.parse(url, request_headers={"User-Agent": "AINewsBot/1.0"})
        cutoff_utc = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        entries = []
        for entry in feed.entries:
            title   = getattr(entry, "title", "").strip()
            summary = getattr(entry, "summary", "")[:500]
            link    = getattr(entry, "link", "")
            published = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                published = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
            if not title or not link: continue
            if published and published < cutoff_utc: continue
            text_lower = (title + " " + summary).lower()
            if not any(kw in text_lower for kw in _AINEWS_KEYWORDS): continue
            entries.append({
                "title": title, "summary": summary, "link": link,
                "source": feed.feed.get("title", url),
                "published": published.astimezone(_AINEWS_TZ).isoformat() if published else None,
            })
        return entries
    except Exception as e:
        ainews_log.warning(f"Feed error ({url}): {e}")
        return []

def _an_fetch_all(feeds: list[str], max_age_hours: int = 1) -> list[dict]:
    all_entries = []
    seen_links  = set()

    def _collect(hours: int):
        for url in feeds:
            for entry in _an_fetch_feed(url, hours):
                if entry["link"] not in seen_links:
                    seen_links.add(entry["link"])
                    all_entries.append(entry)

    _collect(max_age_hours)
    if not all_entries:
        ainews_log.info(f"Nothing in {max_age_hours}h — expanding to 6h...")
        _collect(6)
    if not all_entries:
        ainews_log.info("Nothing in 6h — expanding to 24h...")
        _collect(24)

    return all_entries

def _an_parse_model_response(raw: str) -> list | None:
    raw = raw.strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?", "", raw).rstrip("`").strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) and parsed else None
    except json.JSONDecodeError:
        # Try extracting first JSON array
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group())
                return parsed if isinstance(parsed, list) and parsed else None
            except:
                pass
    return None

def _an_verify(primary: list[dict], verification: list[dict], target_count: int = 5) -> list[dict]:
    _an_get_clients()
    primary_text = "\n".join(
        f"[P{i+1}] {_an_sanitize(e['title'])} | {_an_sanitize(e['source'])}"
        for i, e in enumerate(primary)
    )
    verify_text = "\n".join(
        f"[V{i+1}] {_an_sanitize(e['title'])}"
        for i, e in enumerate(verification[:40])
    )

    prompt = f"""You are a senior AI journalist.
PRIMARY: {primary_text}
VERIFY: {verify_text}

Rules:
1. ONLY keep major AI launches/research.
2. JSON SAFETY: Escape internal double quotes with backslash.
3. EVENT SLUG: Create a unique 'slug' (e.g. 'openai-sora-launch') for the event.
4. FORMAT: Return ONLY a raw JSON array. No markdown. The "index" field must be a plain integer (e.g. 1, not "P1").

Format:
[
  {{
    "index": 1,
    "verified": true,
    "slug": "<unique-slug>",
    "headline": "<headline>",
    "summary": "<para1>\\n\\n<para2>",
    "platform_url": "<direct-url>",
    "score": 1-10
  }}
]"""

    MAX_RETRIES = 3
    items = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _ainews_client.chat.completions.create(
                model="openrouter/free",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=4000,
            )
            raw = resp.choices[0].message.content.strip()
            parsed = _an_parse_model_response(raw)

            if parsed is None:
                ainews_log.warning(f"Empty response attempt {attempt}: {raw[:120]}")
                if attempt < MAX_RETRIES:
                    time.sleep(5)
                continue

            items = parsed
            break

        except json.JSONDecodeError as e:
            ainews_log.warning(f"JSON parse failed attempt {attempt}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(5)
        except Exception as e:
            ainews_log.error(f"API error attempt {attempt}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(5)

    if not items:
        return []

    try:
        verified = [i for i in items if i.get("verified")]
        verified.sort(key=lambda x: int(str(x.get("score", 0))), reverse=True)

        result = []
        for item in verified[:target_count]:
            idx = int(str(item["index"]).lstrip("PpVv")) - 1
            if 0 <= idx < len(primary):
                entry = primary[idx].copy()
                entry["headline"]     = item["headline"]
                entry["summary"]      = item.get("summary", "")
                entry["slug"]         = item.get("slug", _an_story_key(item["headline"]))
                entry["platform_url"] = item.get("platform_url", entry["link"])
                result.append(entry)
        return result
    except Exception as e:
        ainews_log.error(f"Result assembly failed: {e}")
        return []


def _an_format(stories: list[dict], run_time: datetime) -> list[str]:
    """
    Format stories as Telegram-friendly Markdown messages.
    Telegram uses a simpler Markdown: *bold*, _italic_, [text](url).
    """
    date_str = run_time.strftime("%d %b %Y")
    hour_str = run_time.strftime("%I:%M %p")
    messages = []
    for i, s in enumerate(stories, 1):
        msg = (
            f"🤖 *AI NEWS DIGEST* \\[{i}/{len(stories)}\\]\n"
            f"📅 {date_str}  🕐 {hour_str} IST\n"
            f"{'━' * 32}\n\n"
            f"*{_an_sanitize(s['headline']).upper()}*\n\n"
            f"{s['summary']}\n\n"
            f"🔗 {s.get('platform_url', s['link'])}\n"
            f"{'━' * 32}"
        )
        messages.append(msg)
    return messages


def _an_send(messages: list[str], chat_id: str | int, max_retries: int = 3) -> list[str]:
    """
    Send Telegram messages with per-message retry/backoff.
    Returns the list of successfully sent message bodies.
    """
    sent_bodies: list[str] = []
    for msg in messages:
        for attempt in range(1, max_retries + 1):
            success = _tg_send_message(chat_id, msg)
            if success:
                sent_bodies.append(msg)
                time.sleep(1)   # avoid Telegram flood limits
                break
            else:
                ainews_log.error(f"Telegram send failed (attempt {attempt}/{max_retries})")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                else:
                    ainews_log.error(f"Giving up on message: {msg[:80]}...")
    ainews_log.info(f"AI News: {len(sent_bodies)}/{len(messages)} messages sent → {chat_id}")
    return sent_bodies


def run_ainews_digest(chat_id: str | int = None, max_age_hours: int = 1):
    """
    Run an AI news digest.

    max_age_hours controls how far back to look for stories:
      - Scheduled (hourly) calls pass max_age_hours=1  → only last hour's news
      - Manual triggers (user sends 'ainews') pass max_age_hours=6 → richer results

    chat_id: Telegram chat_id to send to. Falls back to TELEGRAM_CHAT_ID env var.
    """
    _an_get_clients()
    now    = datetime.now(_AINEWS_TZ)
    target = chat_id or _tg_chat_id
    ainews_log.info(
        f"=== AI News digest {now.strftime('%Y-%m-%d %H:%M')} IST  "
        f"(window={max_age_hours}h) ==="
    )

    primary = _an_fetch_all(_AINEWS_PRIMARY_FEEDS, max_age_hours)
    verify  = _an_fetch_all(_AINEWS_VERIFY_FEEDS,  max_age_hours)
    ainews_log.info(f"Fetched {len(primary)} primary / {len(verify)} verify stories")

    if not primary:
        ainews_log.info("No primary stories found, skipping digest.")
        if chat_id:   # Only notify if user manually triggered
            _tg_send_message(
                target,
                "📭 No new AI stories found in the last few hours\\. Try again later\\!"
            )
        return

    ainews_log.info(f"Verifying {len(primary)} stories via OpenRouter...")
    stories = _an_verify(primary, verify)
    if not stories:
        ainews_log.info("No stories passed verification.")
        return

    sent_store  = _an_load_sent()
    new_stories = [s for s in stories if s["slug"] not in sent_store]

    if not new_stories:
        ainews_log.info("No genuinely new stories (all slugs in store).")
        if chat_id:
            _tg_send_message(
                target,
                "✅ You're up to date — no new AI stories since your last digest\\!"
            )
        return

    formatted     = _an_format(new_stories, now)
    sent_messages = _an_send(formatted, target)

    sent_count = len(sent_messages)
    for story in new_stories[:sent_count]:
        sent_store[story["slug"]] = {"title": story["headline"], "sent_at": now.isoformat()}
    _an_save_sent(sent_store)
    ainews_log.info(f"AI News digest complete: {sent_count}/{len(new_stories)} stories sent.")
