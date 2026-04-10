import yfinance as yf

# Known valid NSE index/ETF symbols that are short and purely alphabetic
# but would fail a live ticker validation (they're real but yf may not list them).
# Add any you commonly use.
_KNOWN_SYMBOLS = set()

def _is_valid_ticker(symbol: str) -> bool:
    """
    Quick check: does yfinance return any price data for this symbol?
    Uses a 5-day download — fast and lightweight.
    """
    try:
        df = yf.download(symbol, period="5d", interval="1d",
                         progress=False, auto_adjust=True)
        return df is not None and len(df) > 0
    except Exception:
        return False


def find_symbol(company_name: str) -> str | None:
    query = company_name.strip().upper()

    # 1. Short-circuit for known symbols (user-curated whitelist)
    if query in _KNOWN_SYMBOLS:
        return f"{query}.NS"

    # 2. If input looks like a bare ticker (1–10 alpha chars), validate before accepting.
    #    Previously this blindly returned <input>.NS — now we confirm it trades.
    if 1 <= len(query) <= 10 and query.isalpha():
        candidate = f"{query}.NS"
        if _is_valid_ticker(candidate):
            return candidate
        # Fall through to search — maybe user typed a partial company name

    try:
        # 3. Search by descriptive name (e.g. "Tata Motors")
        search = yf.Search(query)
        quotes = search.quotes

        if not quotes:
            return None

        # Prefer NSE, then BSE
        for q in quotes:
            symbol = q.get("symbol")
            if symbol and symbol.endswith(".NS"):
                return symbol

        for q in quotes:
            symbol = q.get("symbol")
            if symbol and symbol.endswith(".BO"):
                return symbol

        return quotes[0].get("symbol")

    except Exception as e:
        print(f"Symbol search error for {company_name}: {e}")
        return None
