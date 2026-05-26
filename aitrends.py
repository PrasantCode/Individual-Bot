"""
aitrends.py — AI Trends Module
================================
Companion to ainews.py. Instead of RSS news feeds, this module pulls REAL
signals from the internet: what developers and individuals are BUILDING with AI.

Sources:
  • Hacker News  — "Show HN" posts (builders announcing projects)
  • GitHub       — Trending AI repositories (what's getting starred)
  • Hugging Face — Trending models & Spaces (what people are making/using)
  • Reddit       — r/LocalLLaMA, r/SideProject, r/MachineLearning (no auth needed)
  • Product Hunt — Newly launched AI tools

Trigger keywords (add to your bot dispatcher):
  AITRENDS_TRIGGERS = {"aitrends", "trends", "ai trends", "what are people building"}

Scheduled digest: call run_aitrends_digest() on a cron / APScheduler job.
Manual digest:    call run_aitrends_digest(chat_id=<id>) from your message handler.
Detail view:      call run_aitrends_story_detail(number, chat_id) after user sends "trends 3".

Required env vars (same as ainews.py):
  OPENROUTER_API_KEY   — for LLM summarisation
  TELEGRAM_BOT_TOKEN   — your Telegram bot token
  TELEGRAM_CHAT_ID     — default chat to send scheduled digests to

Optional env vars:
  REDDIT_CLIENT_ID     — for authenticated Reddit API (higher rate limits)
  REDDIT_CLIENT_SECRET — paired with REDDIT_CLIENT_ID
  PRODUCT_HUNT_TOKEN   — for Product Hunt API (get free token at producthunt.com/v2/oauth)
"""

import os
import re
import time
import json
import hashlib
import logging
import requests
from openai import OpenAI
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
_AITRENDS_TZ                  = ZoneInfo("Asia/Kolkata")
_AITRENDS_SENT_STORE          = Path("sent_trends.json")
_AITRENDS_LAST_DIGEST_STORE   = Path("last_trends.json")
_AITRENDS_MAX_DAYS            = 3
_AITRENDS_MAX_AGE_HOURS       = 24   # Trends move slower than news; 24h is good

AITRENDS_TRIGGERS             = {"aitrends", "trends", "ai trends", "what are people building"}
AITRENDS_DETAIL_PATTERN       = re.compile(r"^(?:trends|aitrends)\s+(\d+)$", re.IGNORECASE)

_aitrends_client  = None
_tg_bot_token_t   = None
_tg_chat_id_t     = None

aitrends_log = logging.getLogger("aitrends")

# Reddit subreddits to watch (JSON API, no auth needed for basic access)
_REDDIT_SUBS = [
    "LocalLLaMA",
    "MachineLearning",
    "SideProject",
    "artificial",
    "learnmachinelearning",
    "ChatGPT",
    "singularity",
]

# GitHub topics/languages to scrape for trending repos
_GITHUB_TRENDING_URLS = [
    "https://github.com/trending/python?since=daily",
    "https://github.com/trending/javascript?since=daily",
    "https://github.com/trending?since=daily",   # all languages
]

# Hugging Face Trending API endpoints
_HF_MODELS_URL = "https://huggingface.co/api/models?sort=trending&limit=20&direction=-1"
_HF_SPACES_URL = "https://huggingface.co/api/spaces?sort=trending&limit=20&direction=-1"

# AI-related keywords used to filter GitHub repos and Reddit posts
_AITRENDS_KEYWORDS = [
    "llm", "ai", "gpt", "claude", "gemini", "llama", "mistral", "openai",
    "langchain", "langgraph", "agent", "rag", "chatbot", "diffusion",
    "stable diffusion", "hugging face", "transformers", "neural", "ml",
    "machine learning", "deep learning", "inference", "fine-tun", "lora",
    "quantiz", "ollama", "open source model", "multimodal", "voice ai",
    "text to", "image generation", "code generation", "copilot", "cursor",
    "automated", "automation", "ai tool", "ai app", "built with ai",
    "powered by ai", "ai assistant", "embedding", "vector", "semantic search",
]

_SHOW_HN_PATTERN = re.compile(r"^show hn[:\s]", re.IGNORECASE)


# ---------------------------------------------------------------------------
# CLIENT INIT
# ---------------------------------------------------------------------------
def _at_get_clients():
    global _aitrends_client, _tg_bot_token_t, _tg_chat_id_t
    if _aitrends_client is None:
        _aitrends_client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    if _tg_bot_token_t is None:
        _tg_bot_token_t = os.environ["TELEGRAM_BOT_TOKEN"]
        _tg_chat_id_t   = os.environ["TELEGRAM_CHAT_ID"]


# ---------------------------------------------------------------------------
# TELEGRAM SENDER (self-contained, independent of ainews.py)
# ---------------------------------------------------------------------------
def _tg_send_trends(chat_id, text: str, parse_mode: str = "Markdown") -> bool:
    _at_get_clients()
    url     = f"https://api.telegram.org/bot{_tg_bot_token_t}/sendMessage"
    payload = {
        "chat_id":                  chat_id,
        "text":                     text[:4096],
        "parse_mode":               parse_mode,
        "disable_web_page_preview": False,
    }
    try:
        r = requests.post(url, json=payload, timeout=15)
        if not r.ok:
            aitrends_log.error(f"Telegram send error: {r.status_code} {r.text[:200]}")
        return r.ok
    except Exception as e:
        aitrends_log.error(f"Telegram request failed: {e}")
        return False


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def _at_sanitize(text: str) -> str:
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"[\x00-\x1f\x7f]", "", text)
    return text.strip()

def _at_story_key(title: str) -> str:
    normalised = re.sub(r"[^a-z0-9 ]", "", title.lower().strip())
    normalised = re.sub(r"\s+", " ", normalised)
    return hashlib.md5(normalised.encode()).hexdigest()[:12]

def _at_is_ai_related(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in _AITRENDS_KEYWORDS)

def _at_load_sent() -> dict:
    if _AITRENDS_SENT_STORE.exists():
        try:
            return json.loads(_AITRENDS_SENT_STORE.read_text())
        except:
            return {}
    return {}

def _at_save_sent(store: dict):
    cutoff = (datetime.now() - timedelta(days=_AITRENDS_MAX_DAYS)).isoformat()
    pruned = {k: v for k, v in store.items() if v.get("sent_at", "") > cutoff}
    _AITRENDS_SENT_STORE.write_text(json.dumps(pruned, indent=2))

def _at_save_last_digest(stories: list[dict]):
    data = [
        {
            "index":    i + 1,
            "headline": s.get("headline", s["title"]),
            "summary":  s.get("summary", ""),
            "url":      s.get("url", ""),
            "source":   s.get("source", ""),
        }
        for i, s in enumerate(stories)
    ]
    _AITRENDS_LAST_DIGEST_STORE.write_text(json.dumps(data, indent=2))

def _at_load_last_digest() -> list[dict]:
    if _AITRENDS_LAST_DIGEST_STORE.exists():
        try:
            return json.loads(_AITRENDS_LAST_DIGEST_STORE.read_text())
        except:
            return []
    return []


# ---------------------------------------------------------------------------
# SOURCE 1: HACKER NEWS — Show HN posts
# ---------------------------------------------------------------------------
def _fetch_hackernews(max_age_hours: int = 24) -> list[dict]:
    """
    Fetch 'Show HN' posts from Hacker News via the official Algolia API.
    These are people announcing projects they built — the richest signal.
    """
    cutoff_ts = int((datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).timestamp())
    url = (
        "https://hn.algolia.com/api/v1/search_by_date"
        "?tags=show_hn"
        f"&numericFilters=created_at_i>{cutoff_ts}"
        "&hitsPerPage=50"
    )
    results = []
    try:
        r = requests.get(url, timeout=15,
                         headers={"User-Agent": "AITrendsBot/1.0"})
        r.raise_for_status()
        for hit in r.json().get("hits", []):
            title   = hit.get("title", "").strip()
            link    = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}"
            points  = hit.get("points", 0) or 0
            comments = hit.get("num_comments", 0) or 0
            if not title:
                continue
            if not _at_is_ai_related(title):
                continue
            results.append({
                "title":    title,
                "url":      link,
                "summary":  f"HN Show HN — {points} points, {comments} comments",
                "source":   "Hacker News (Show HN)",
                "score":    points + comments * 2,  # engagement signal
                "type":     "project",
            })
        aitrends_log.info(f"HN: fetched {len(results)} AI Show HN posts")
    except Exception as e:
        aitrends_log.warning(f"HN fetch error: {e}")
    return results


# ---------------------------------------------------------------------------
# SOURCE 2: GITHUB TRENDING
# ---------------------------------------------------------------------------
def _fetch_github_trending() -> list[dict]:
    """
    Scrape GitHub Trending page for AI-related repos.
    Uses the unofficial github-trending-api or direct HTML scraping.
    Falls back to the casual scraper approach if the API is unavailable.
    """
    results = []

    # Try the unofficial trending API first (more reliable parsing)
    api_url = "https://api.gitterapp.com/repositories?since=daily"
    try:
        r = requests.get(api_url, timeout=10, headers={"User-Agent": "AITrendsBot/1.0"})
        if r.ok:
            for repo in r.json():
                name        = repo.get("name", "")
                description = repo.get("description") or ""
                url         = repo.get("url", "")
                stars_today = repo.get("currentPeriodStars", 0) or 0
                stars_total = repo.get("stars", 0) or 0
                language    = repo.get("language") or "unknown"

                if not _at_is_ai_related(name + " " + description):
                    continue

                results.append({
                    "title":    f"{name} ({language})",
                    "url":      url or f"https://github.com/{name}",
                    "summary":  f"⭐ {stars_today} new stars today · {stars_total} total — {description[:200]}",
                    "source":   "GitHub Trending",
                    "score":    stars_today,
                    "type":     "repo",
                })
            aitrends_log.info(f"GitHub (gitterapp): fetched {len(results)} AI repos")
            return results
    except Exception as e:
        aitrends_log.warning(f"GitHub gitterapp API error: {e}, trying direct scrape...")

    # Fallback: scrape github.com/trending directly with regex
    try:
        r = requests.get(
            "https://github.com/trending?since=daily",
            timeout=15,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; AITrendsBot/1.0)",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        r.raise_for_status()
        html = r.text

        # Extract repo entries: pattern looks for /owner/repo in article tags
        repo_names = re.findall(
            r'href="/([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)"[^>]*>\s*\n\s*<span[^>]*>[^<]+</span>',
            html
        )
        descriptions = re.findall(r'<p class="col-9[^"]*"[^>]*>\s*(.*?)\s*</p>', html, re.DOTALL)
        stars_today_list = re.findall(r'([\d,]+)\s+stars today', html)

        for i, name in enumerate(repo_names[:30]):
            desc     = _at_sanitize(descriptions[i]) if i < len(descriptions) else ""
            stars    = stars_today_list[i].replace(",", "") if i < len(stars_today_list) else "0"
            if not _at_is_ai_related(name + " " + desc):
                continue
            results.append({
                "title":   name,
                "url":     f"https://github.com/{name}",
                "summary": f"⭐ {stars} stars today — {desc[:200]}",
                "source":  "GitHub Trending",
                "score":   int(stars) if stars.isdigit() else 0,
                "type":    "repo",
            })

        aitrends_log.info(f"GitHub (scrape): fetched {len(results)} AI repos")
    except Exception as e:
        aitrends_log.warning(f"GitHub scrape error: {e}")

    return results


# ---------------------------------------------------------------------------
# SOURCE 3: HUGGING FACE TRENDING
# ---------------------------------------------------------------------------
def _fetch_huggingface() -> list[dict]:
    """
    Fetch trending models and Spaces from the Hugging Face API.
    Models = what people are training/fine-tuning.
    Spaces = what people are BUILDING (interactive AI apps).
    """
    results = []

    # Trending Spaces (mini-apps built with AI — the most relevant for "what are people building")
    try:
        r = requests.get(_HF_SPACES_URL, timeout=15,
                         headers={"User-Agent": "AITrendsBot/1.0"})
        r.raise_for_status()
        for space in r.json():
            sid     = space.get("id", "")
            likes   = space.get("likes", 0) or 0
            sdk     = space.get("sdk", "")
            tags    = " ".join(space.get("tags") or [])
            title   = sid.split("/")[-1].replace("-", " ").replace("_", " ").title()
            results.append({
                "title":    f"🤗 {sid}",
                "url":      f"https://huggingface.co/spaces/{sid}",
                "summary":  f"Trending HF Space · {likes} likes · SDK: {sdk} · tags: {tags[:100]}",
                "source":   "Hugging Face Spaces",
                "score":    likes,
                "type":     "space",
            })
        aitrends_log.info(f"HF Spaces: fetched {len(results)} trending spaces")
    except Exception as e:
        aitrends_log.warning(f"HF Spaces fetch error: {e}")

    # Trending Models (what people are releasing / fine-tuning)
    try:
        r = requests.get(_HF_MODELS_URL, timeout=15,
                         headers={"User-Agent": "AITrendsBot/1.0"})
        r.raise_for_status()
        for model in r.json():
            mid      = model.get("modelId") or model.get("id", "")
            likes    = model.get("likes", 0) or 0
            pipeline = model.get("pipeline_tag", "")
            tags     = " ".join((model.get("tags") or [])[:6])
            results.append({
                "title":    f"🤗 {mid}",
                "url":      f"https://huggingface.co/{mid}",
                "summary":  f"Trending HF Model · {likes} likes · Task: {pipeline} · tags: {tags}",
                "source":   "Hugging Face Models",
                "score":    likes,
                "type":     "model",
            })
        aitrends_log.info(f"HF Models: fetched {len(results)} trending models")
    except Exception as e:
        aitrends_log.warning(f"HF Models fetch error: {e}")

    return results


# ---------------------------------------------------------------------------
# SOURCE 4: REDDIT (JSON API — no auth needed for public posts)
# ---------------------------------------------------------------------------
def _fetch_reddit(max_age_hours: int = 24) -> list[dict]:
    """
    Fetch top posts from AI/dev subreddits using the Reddit JSON API.
    No OAuth needed for public read access.
    If REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET are set, uses authenticated access
    for higher rate limits.
    """
    results  = []
    cutoff   = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    headers  = {"User-Agent": "AITrendsBot/1.0 by TrendWatcher"}
    token    = None

    # Optionally authenticate for higher rate limits
    client_id     = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    if client_id and client_secret:
        try:
            auth_r = requests.post(
                "https://www.reddit.com/api/v1/access_token",
                auth=(client_id, client_secret),
                data={"grant_type": "client_credentials"},
                headers=headers,
                timeout=10,
            )
            if auth_r.ok:
                token = auth_r.json().get("access_token")
                headers["Authorization"] = f"bearer {token}"
        except Exception as e:
            aitrends_log.warning(f"Reddit auth failed (falling back to anon): {e}")

    base = "https://oauth.reddit.com" if token else "https://www.reddit.com"

    for sub in _REDDIT_SUBS:
        try:
            url = f"{base}/r/{sub}/hot.json?limit=25"
            r   = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            posts = r.json().get("data", {}).get("children", [])

            for post in posts:
                d = post.get("data", {})
                title   = d.get("title", "").strip()
                link    = d.get("url", "")
                score   = d.get("score", 0) or 0
                comments= d.get("num_comments", 0) or 0
                created = datetime.fromtimestamp(d.get("created_utc", 0), tz=timezone.utc)
                permalink = f"https://www.reddit.com{d.get('permalink', '')}"

                if created < cutoff:
                    continue
                if not title:
                    continue
                if not _at_is_ai_related(title + " " + d.get("selftext", "")[:300]):
                    continue
                # Prefer posts that are about building, not just discussing
                build_signals = ["built", "build", "made", "created", "show", "launched",
                                 "released", "open source", "github", "demo", "project",
                                 "side project", "i made", "my ", "tool", "app"]
                engagement = score + comments * 3
                results.append({
                    "title":   title,
                    "url":     link if link.startswith("http") else permalink,
                    "summary": f"r/{sub} · {score} upvotes · {comments} comments",
                    "source":  f"Reddit r/{sub}",
                    "score":   engagement,
                    "type":    "community",
                    "build_signal": any(s in title.lower() for s in build_signals),
                })

            time.sleep(0.5)   # be polite to Reddit

        except Exception as e:
            aitrends_log.warning(f"Reddit r/{sub} error: {e}")

    # Sort: prioritise build-signal posts, then by engagement
    results.sort(key=lambda x: (x.get("build_signal", False), x["score"]), reverse=True)
    aitrends_log.info(f"Reddit: fetched {len(results)} AI posts across {len(_REDDIT_SUBS)} subs")
    return results


# ---------------------------------------------------------------------------
# SOURCE 5: PRODUCT HUNT (new AI tools launched)
# ---------------------------------------------------------------------------
def _fetch_product_hunt() -> list[dict]:
    """
    Fetch today's AI-related products from Product Hunt.
    Uses the GraphQL API — a free token is enough.
    Set PRODUCT_HUNT_TOKEN env var (get one free at producthunt.com/v2/oauth).
    Falls back to the public Algolia-backed search if no token.
    """
    results = []
    ph_token = os.environ.get("PRODUCT_HUNT_TOKEN")

    if ph_token:
        query = """
        {
          posts(first: 30, order: VOTES, postedAfter: "%s") {
            edges {
              node {
                name
                tagline
                url
                votesCount
                commentsCount
                topics { edges { node { name } } }
              }
            }
          }
        }
        """ % (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            r = requests.post(
                "https://api.producthunt.com/v2/api/graphql",
                json={"query": query},
                headers={
                    "Authorization": f"Bearer {ph_token}",
                    "Content-Type":  "application/json",
                    "User-Agent":    "AITrendsBot/1.0",
                },
                timeout=15,
            )
            r.raise_for_status()
            edges = r.json().get("data", {}).get("posts", {}).get("edges", [])
            for edge in edges:
                node   = edge.get("node", {})
                name   = node.get("name", "")
                tag    = node.get("tagline", "")
                url    = node.get("url", "")
                votes  = node.get("votesCount", 0)
                topics = [t["node"]["name"] for t in node.get("topics", {}).get("edges", [])]

                if not _at_is_ai_related(name + " " + tag + " " + " ".join(topics)):
                    continue

                results.append({
                    "title":   name,
                    "url":     url,
                    "summary": f"🚀 {tag} — {votes} upvotes · Topics: {', '.join(topics[:4])}",
                    "source":  "Product Hunt",
                    "score":   votes,
                    "type":    "launch",
                })
            aitrends_log.info(f"Product Hunt: fetched {len(results)} AI launches")
        except Exception as e:
            aitrends_log.warning(f"Product Hunt API error: {e}")
    else:
        # Fallback: scrape PH without auth via their public API
        aitrends_log.info("No PRODUCT_HUNT_TOKEN — skipping Product Hunt (set token for access)")

    return results


# ---------------------------------------------------------------------------
# AGGREGATE ALL SOURCES
# ---------------------------------------------------------------------------
def _at_fetch_all(max_age_hours: int = 24) -> list[dict]:
    """Collect items from all sources and deduplicate by title hash."""
    all_items: list[dict] = []
    seen_keys: set[str]   = set()

    for item in (
        _fetch_hackernews(max_age_hours)
        + _fetch_github_trending()
        + _fetch_huggingface()
        + _fetch_reddit(max_age_hours)
        + _fetch_product_hunt()
    ):
        key = _at_story_key(item["title"])
        if key not in seen_keys:
            seen_keys.add(key)
            item["dedup_key"] = key
            all_items.append(item)

    aitrends_log.info(f"Total trend items collected: {len(all_items)}")
    return all_items


# ---------------------------------------------------------------------------
# LLM SUMMARISER  (same OpenRouter pattern as ainews.py)
# ---------------------------------------------------------------------------
def _at_summarise(items: list[dict], target_count: int = 15) -> list[dict]:
    """
    Send all items to the LLM and ask it to:
    1. Identify the most interesting BUILDING trends
    2. Group related items (e.g. "several people building RAG pipelines")
    3. Write a punchy 2-sentence summary of what people are doing
    Returns up to target_count enriched items.
    """
    _at_get_clients()

    items_text = "\n".join(
        f"[{i+1}] SOURCE={item['source']} | TYPE={item['type']} | "
        f"TITLE={_at_sanitize(item['title'])} | "
        f"SUMMARY={_at_sanitize(item.get('summary', ''))}"
        for i, item in enumerate(items[:80])   # cap to avoid context overflow
    )

    prompt = f"""You are an AI trends analyst. Below are raw signals from Hacker News, GitHub, Hugging Face, Reddit, and Product Hunt showing what developers and individuals are BUILDING with AI right now.

ITEMS:
{items_text}

Your task:
1. Identify the {target_count} most interesting and representative trends — focus on WHAT PEOPLE ARE BUILDING, not news events.
2. Write a punchy, specific summary (2-3 sentences) of what is being built and why it matters.
3. Return ONLY a raw JSON array. No markdown. No preamble.

Format each item as:
{{
  "index": <integer matching the [N] above>,
  "headline": "<catchy headline describing what is being built>",
  "summary": "<2-3 sentence description of what's being built, what AI is used, and why it's interesting>",
  "url": "<best url for this item>",
  "source": "<source name>",
  "score": <1-10 interestingness rating>,
  "trend_theme": "<short theme label e.g. 'Agentic coding tools' or 'Local LLM apps'>"
}}"""

    MAX_RETRIES = 3
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _aitrends_client.chat.completions.create(
                model="openrouter/auto",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=4000,
            )
            raw = resp.choices[0].message.content.strip()
            # Strip markdown fences
            raw = re.sub(r"^```(?:json)?", "", raw).rstrip("`").strip()
            parsed = json.loads(raw) if raw.startswith("[") else None
            if parsed is None:
                m = re.search(r"\[.*\]", raw, re.DOTALL)
                if m:
                    parsed = json.loads(m.group())
            if not isinstance(parsed, list) or not parsed:
                aitrends_log.warning(f"LLM empty response (attempt {attempt})")
                time.sleep(5)
                continue

            # Map back to full items
            result = []
            for llm_item in parsed:
                idx = int(str(llm_item.get("index", 0))) - 1
                if 0 <= idx < len(items):
                    entry = items[idx].copy()
                    entry["headline"]    = llm_item.get("headline", entry["title"])
                    entry["summary"]     = llm_item.get("summary", entry.get("summary", ""))
                    entry["url"]         = llm_item.get("url", entry.get("url", ""))
                    entry["source"]      = llm_item.get("source", entry.get("source", ""))
                    entry["trend_theme"] = llm_item.get("trend_theme", "")
                    entry["llm_score"]   = llm_item.get("score", 5)
                    result.append(entry)

            result.sort(key=lambda x: x.get("llm_score", 0), reverse=True)
            return result[:target_count]

        except json.JSONDecodeError as e:
            aitrends_log.warning(f"JSON parse failed (attempt {attempt}): {e}")
            time.sleep(5)
        except Exception as e:
            aitrends_log.error(f"LLM API error (attempt {attempt}): {e}")
            time.sleep(5)

    aitrends_log.error("LLM summarisation failed after all retries")
    return []


# ---------------------------------------------------------------------------
# FORMATTERS
# ---------------------------------------------------------------------------
_SOURCE_EMOJI = {
    "Hacker News (Show HN)": "🟠",
    "GitHub Trending":        "🐙",
    "Hugging Face Spaces":    "🤗",
    "Hugging Face Models":    "🤗",
    "Product Hunt":           "🚀",
}

def _source_emoji(source: str) -> str:
    for key, emoji in _SOURCE_EMOJI.items():
        if key in source:
            return emoji
    if "Reddit" in source:
        return "💬"
    return "🔗"


def _generate_ai_summary(story: dict) -> str:
    """Generate a 2-3 line AI summary of the trend topic."""
    _at_get_clients()
    
    # Extract relevant information for the summary
    title = story.get("title", "")
    source = story.get("source", "")
    item_type = story.get("type", "")
    
    # Create a concise prompt for summarization
    prompt = f"""Summarize this AI trend in 2-3 lines (max 150 characters):

Title: {title}
Source: {source}
Type: {item_type}

Focus on what people are building or why this is trending. Be concise and informative."""

    try:
        resp = _aitrends_client.chat.completions.create(
            model="openrouter/auto",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100,
        )
        summary = resp.choices[0].message.content.strip()
        return summary
    except Exception as e:
        aitrends_log.error(f"Failed to generate AI summary: {e}")
        # Fallback to a simple description if AI fails
        return f"Trending {item_type.lower()} from {source}"


def _at_format_headlines(stories: list[dict], run_time: datetime) -> list[str]:
    """Numbered headline list for manual digest — same style as ainews.py."""
    date_str = run_time.strftime("%d %b %Y")
    hour_str = run_time.strftime("%I:%M %p")

    lines = [
        f"🔥 *AI TRENDS DIGEST — {date_str}  {hour_str} IST*",
        f"{'━' * 32}",
        f"_What developers & creators are building with AI right now_",
        f"_{len(stories)} trends, ranked by interestingness_\n",
    ]

    # Group by theme if available
    themes_seen = []
    for i, s in enumerate(stories, 1):
        theme = s.get("trend_theme", "")
        emoji = _source_emoji(s.get("source", ""))
        headline = _at_sanitize(s.get("headline", s["title"]))

        if theme and theme not in themes_seen:
            themes_seen.append(theme)
            lines.append(f"\n_— {theme} —_")

        lines.append(f"*{i}.* {emoji} {headline}")
        
        # Use the summary already generated by _at_summarise (rich, 2-3 sentences)
        summary = s.get("summary", "").strip()
        if summary:
            lines.append(f"   _{summary}_")
        else:
            lines.append(f"   `{s.get('source', '')}` · Reply `trends {i}` for details")
        
        lines.append("")

    body = "\n".join(lines)
    chunks = []
    while len(body) > 4096:
        split = body.rfind("\n", 0, 4096)
        chunks.append(body[:split])
        body = body[split:].lstrip("\n")
    chunks.append(body)
    return chunks


def _at_format_detail(s: dict, number: int, total: int, run_time: datetime) -> str:
    """Full detail card for a single trend item."""
    date_str = run_time.strftime("%d %b %Y")
    hour_str = run_time.strftime("%I:%M %p")
    emoji    = _source_emoji(s.get("source", ""))

    return (
        f"🔥 *AI TRENDS — Item \\#{number} of {total}*\n"
        f"📅 {date_str}  🕐 {hour_str} IST\n"
        f"{'━' * 32}\n\n"
        f"{emoji} *{_at_sanitize(s.get('headline', s.get('title', ''))).upper()}*\n\n"
        f"{s.get('summary', '_No summary available._')}\n\n"
        f"🏷️ _{s.get('trend_theme', 'AI Trend')}_\n"
        f"📡 Source: {s.get('source', '')}\n\n"
        f"🔗 {s.get('url', '')}\n"
        f"{'━' * 32}\n"
        f"_Reply_ `trends <number>` _for any other item_"
    )


def _at_send(messages: list[str], chat_id, max_retries: int = 3) -> list[str]:
    sent_bodies: list[str] = []
    for msg in messages:
        for attempt in range(1, max_retries + 1):
            success = _tg_send_trends(chat_id, msg)
            if success:
                sent_bodies.append(msg)
                time.sleep(1)
                break
            else:
                aitrends_log.error(f"Send failed (attempt {attempt}/{max_retries})")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                else:
                    aitrends_log.error(f"Giving up on message: {msg[:80]}...")
    aitrends_log.info(f"AI Trends: {len(sent_bodies)}/{len(messages)} messages sent → {chat_id}")
    return sent_bodies


# ---------------------------------------------------------------------------
# PUBLIC ENTRY POINTS  (call these from your bot dispatcher)
# ---------------------------------------------------------------------------
def run_aitrends_digest(chat_id=None, max_age_hours: int = _AITRENDS_MAX_AGE_HOURS):
    """
    Run the AI Trends digest.

    chat_id: Telegram chat_id. If None, uses TELEGRAM_CHAT_ID env var (scheduled run).
    max_age_hours: How far back to look. Default 24h (trends move slower than news).

    Usage in your bot:
        if msg.lower() in AITRENDS_TRIGGERS:
            run_aitrends_digest(chat_id=update.message.chat_id)
    """
    _at_get_clients()
    now    = datetime.now(_AITRENDS_TZ)
    target = chat_id or _tg_chat_id_t
    is_manual = bool(chat_id)
    count  = 15 if is_manual else 5

    aitrends_log.info(
        f"=== AI Trends digest {now.strftime('%Y-%m-%d %H:%M')} IST "
        f"(window={max_age_hours}h) ==="
    )

    items = _at_fetch_all(max_age_hours)
    if not items:
        aitrends_log.info("No trend items found.")
        if is_manual:
            _tg_send_trends(target, "📭 No trend signals found right now. Try again in a few hours!")
        return

    aitrends_log.info(f"Summarising {len(items)} items via LLM (target={count})...")
    stories = _at_summarise(items, target_count=count)
    if not stories:
        aitrends_log.info("LLM returned no stories.")
        return

    sent_store  = _at_load_sent()
    new_stories = [s for s in stories if s["dedup_key"] not in sent_store]

    if not new_stories:
        aitrends_log.info("All trend items already sent.")
        if is_manual:
            _tg_send_trends(target, "✅ You're up to date — no new trends since your last digest!")
        return

    # Mark as sent before sending (prevents infinite retry loops)
    for s in new_stories:
        sent_store[s["dedup_key"]] = {
            "title":   s.get("headline", s["title"]),
            "sent_at": now.isoformat(),
        }
    _at_save_sent(sent_store)

    if is_manual:
        _at_save_last_digest(new_stories)
        formatted = _at_format_headlines(new_stories, now)
    else:
        # Scheduled: send one rich card per story
        formatted = [
            _at_format_detail(s, i + 1, len(new_stories), now)
            for i, s in enumerate(new_stories)
        ]

    _at_send(formatted, target)
    aitrends_log.info(f"AI Trends digest complete: {len(new_stories)} trends sent.")


def run_aitrends_story_detail(number: int, chat_id):
    """
    Send full detail for trend item #number from the last manual digest.

    Usage in your bot dispatcher:
        m = AITRENDS_DETAIL_PATTERN.match(message_text.strip())
        if m:
            run_aitrends_story_detail(int(m.group(1)), update.message.chat_id)
    """
    _at_get_clients()
    stories = _at_load_last_digest()

    if not stories:
        _tg_send_trends(
            chat_id,
            "📭 No recent trends digest found. Send `trends` first to get the latest!",
        )
        return

    if number < 1 or number > len(stories):
        _tg_send_trends(
            chat_id,
            f"⚠️ Item #{number} not found. Last digest had {len(stories)} items.",
        )
        return

    s   = stories[number - 1]
    now = datetime.now(_AITRENDS_TZ)
    msg = _at_format_detail(s, number, len(stories), now)
    _tg_send_trends(chat_id, msg)
    aitrends_log.info(f"Sent trend detail #{number} to {chat_id}")
