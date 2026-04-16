import yfinance as yf

# ---------------------------------------------------------------------------
# ALIAS MAP — curated name → NSE symbol for stocks yf.Search() handles poorly.
# Keys are uppercase. Add entries whenever a company name lookup fails.
# ---------------------------------------------------------------------------
_ALIASES: dict[str, str] = {
    # Ola Electric
    "OLA":              "OLAELECTRIC.NS",
    "OLA ELECTRIC":     "OLAELECTRIC.NS",
    "OLAELECTRIC":      "OLAELECTRIC.NS",
    # New-age / IPO stocks that yf.Search often misses
    "ZOMATO":           "ZOMATO.NS",
    "NYKAA":            "NYKAA.NS",
    "FSN":              "NYKAA.NS",
    "DELHIVERY":        "DELHIVERY.NS",
    "PAYTM":            "PAYTM.NS",
    "ONE97":            "PAYTM.NS",
    "POLICYBAZAAR":     "POLICYBZR.NS",
    "PB FINTECH":       "POLICYBZR.NS",
    "MOBIKWIK":         "MOBIKWIK.NS",
    "GO DIGIT":         "GODIGIT.NS",
    "DIGIT":            "GODIGIT.NS",
    "IXIGO":            "IXIGO.NS",
    "AWFIS":            "AWFIS.NS",
    "FIRSTCRY":         "BRAINBEES.NS",
    "BRAINBEES":        "BRAINBEES.NS",
    "SWIGGY":           "SWIGGY.NS",
    "HYUNDAI":          "HYUNDAI.NS",
    "HYUNDAI INDIA":    "HYUNDAI.NS",
    # Common aliases / abbreviations
    "BAJAJ FINANCE":    "BAJFINANCE.NS",
    "BAJAJ FIN":        "BAJFINANCE.NS",
    "BAJFINANCE":       "BAJFINANCE.NS",
    "HDFC BANK":        "HDFCBANK.NS",
    "ICICI BANK":       "ICICIBANK.NS",
    "STATE BANK":       "SBIN.NS",
    "SBI":              "SBIN.NS",
    "TATA MOTORS":      "TATAMOTORS.NS",
    "TATA STEEL":       "TATASTEEL.NS",
    "TATA POWER":       "TATAPOWER.NS",
    "TATA CONSULTANCY": "TCS.NS",
    "INFOSYS":          "INFY.NS",
    "WIPRO":            "WIPRO.NS",
    "RELIANCE":         "RELIANCE.NS",
    "ADANI PORTS":      "ADANIPORTS.NS",
    "ADANI ENTERPRISES":"ADANIENT.NS",
    "ADANI GREEN":      "ADANIGREEN.NS",
    "ADANI POWER":      "ADANIPOWER.NS",
    "ASIAN PAINTS":     "ASIANPAINT.NS",
    "MARUTI":           "MARUTI.NS",
    "MARUTI SUZUKI":    "MARUTI.NS",
    "HERO MOTO":        "HEROMOTOCO.NS",
    "HERO MOTOCORP":    "HEROMOTOCO.NS",
    "BAJAJ AUTO":       "BAJAJ-AUTO.NS",
    "KOTAK":            "KOTAKBANK.NS",
    "KOTAK BANK":       "KOTAKBANK.NS",
    "AXIS BANK":        "AXISBANK.NS",
    "INDUSIND":         "INDUSINDBK.NS",
    "INDUSIND BANK":    "INDUSINDBK.NS",
    "POWER GRID":       "POWERGRID.NS",
    "NTPC":             "NTPC.NS",
    "ONGC":             "ONGC.NS",
    "IOC":              "IOC.NS",
    "BPCL":             "BPCL.NS",
    "HINDALCO":         "HINDALCO.NS",
    "JSPL":             "JINDALSTEL.NS",
    "JINDAL STEEL":     "JINDALSTEL.NS",
    "SAIL":             "SAIL.NS",
    "GRASIM":           "GRASIM.NS",
    "ULTRATECH":        "ULTRACEMCO.NS",
    "ULTRATECH CEMENT": "ULTRACEMCO.NS",
    "SHREECEMENT":      "SHREECEM.NS",
    "SHREE CEMENT":     "SHREECEM.NS",
    "BHARTI AIRTEL":    "BHARTIARTL.NS",
    "AIRTEL":           "BHARTIARTL.NS",
    "JIOFINANCIAL":     "JIOFIN.NS",
    "JIO FINANCIAL":    "JIOFIN.NS",
    "DIXON":            "DIXON.NS",
    "AMBER":            "AMBER.NS",
    "VEDANTA":          "VEDL.NS",
    "HAL":              "HAL.NS",
    "BEL":              "BEL.NS",
    "BHARAT ELECTRONICS":"BEL.NS",
    "IRCTC":            "IRCTC.NS",
    "IRFC":             "IRFC.NS",
    "LIC":              "LICI.NS",
    "NHPC":             "NHPC.NS",
    "RECLTD":           "RECLTD.NS",
    "REC":              "RECLTD.NS",
    "HUDCO":            "HUDCO.NS",
    "SUZLON":           "SUZLON.NS",
    "YES BANK":         "YESBANK.NS",
    "BANDHAN BANK":     "BANDHANBNK.NS",
    "FEDERAL BANK":     "FEDERALBNK.NS",
    "IDFC FIRST":       "IDFCFIRSTB.NS",
    "PNB":              "PNB.NS",
    "CANARA BANK":      "CANBK.NS",
    "BANK OF BARODA":   "BANKBARODA.NS",
    "BOB":              "BANKBARODA.NS",
    "UNION BANK":       "UNIONBANK.NS",
    "INDIAN BANK":      "INDIANB.NS",
    "IOB":              "IOB.NS",
    "UCO BANK":         "UCOBANK.NS",
    "CENTRAL BANK":     "CENTRALBK.NS",
    "MAHANAGAR GAS":    "MGL.NS",
    "MGL":              "MGL.NS",
    "IGL":              "IGL.NS",
    "INDRAPRASTHA GAS": "IGL.NS",
    "GUJARAT GAS":      "GUJGASLTD.NS",
    "PIDILITE":         "PIDILITIND.NS",
    "BERGER PAINTS":    "BERGEPAINT.NS",
    "KANSAI NEROLAC":   "KANSAINER.NS",
    "HAVELLS":          "HAVELLS.NS",
    "VOLTAS":           "VOLTAS.NS",
    "BLUE STAR":        "BLUESTARCO.NS",
    "CROMPTON":         "CROMPTON.NS",
    "ORIENT ELECTRIC":  "ORIENTELEC.NS",
    "CEAT":             "CEATLTD.NS",
    "MRF":              "MRF.NS",
    "APOLLO TYRES":     "APOLLOTYRE.NS",
    "BALKRISHNA":       "BALKRISIND.NS",
    "BKT":              "BALKRISIND.NS",
    "EICHER":           "EICHERMOT.NS",
    "ROYAL ENFIELD":    "EICHERMOT.NS",
    "TVS MOTOR":        "TVSMOTOR.NS",
    "TVS":              "TVSMOTOR.NS",
    "ATUL AUTO":        "ATULAUTO.NS",
    "MOTHERSON":        "MOTHERSON.NS",
    "MINDA":            "MINDAIND.NS",
    "LUPIN":            "LUPIN.NS",
    "DR REDDY":         "DRREDDY.NS",
    "DR. REDDY":        "DRREDDY.NS",
    "CIPLA":            "CIPLA.NS",
    "SUN PHARMA":       "SUNPHARMA.NS",
    "SUNPHARMA":        "SUNPHARMA.NS",
    "DIVI'S":           "DIVISLAB.NS",
    "DIVIS":            "DIVISLAB.NS",
    "BIOCON":           "BIOCON.NS",
    "ALKEM":            "ALKEM.NS",
    "TORRENT PHARMA":   "TORNTPHARM.NS",
    "ABBOTT":           "ABBOTINDIA.NS",
    "PFIZER":           "PFIZER.NS",
    "GLAXO":            "GLAXO.NS",
    "SANOFI":           "SANOFI.NS",
    "COFORGE":          "COFORGE.NS",
    "PERSISTENT":       "PERSISTENT.NS",
    "MPHASIS":          "MPHASIS.NS",
    "LTIMINDTREE":      "LTIM.NS",
    "LTI MINDTREE":     "LTIM.NS",
    "TECH MAHINDRA":    "TECHM.NS",
    "TECHMAHINDRA":     "TECHM.NS",
    "HCL TECH":         "HCLTECH.NS",
    "HCLTECH":          "HCLTECH.NS",
    "ORACLE":           "OFSS.NS",
    "ORACLE FINANCIAL": "OFSS.NS",
    "OFSS":             "OFSS.NS",
    "MINDTREE":         "LTIM.NS",
    "INFOEDGE":         "NAUKRI.NS",
    "INFO EDGE":        "NAUKRI.NS",
    "NAUKRI":           "NAUKRI.NS",
    "JUSTDIAL":         "JUSTDIAL.NS",
    "MATRIMONY":        "MATRIMONY.NS",
    "INDIAMART":        "INDIAMART.NS",
    "MAPMYINDIA":       "MAPMYINDIA.NS",
    "CARTRADE":         "CARTRADE.NS",
    "EASEMYTRIP":       "EASEMYTRIP.NS",
    "MAKEMYTRIP":       "MMYT",       # listed on NASDAQ
}

# Known valid NSE index/ETF symbols (user-curated whitelist)
_KNOWN_SYMBOLS: set[str] = set()


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

    # 0. Alias map — highest priority, fastest path, no API call needed
    if query in _ALIASES:
        return _ALIASES[query]

    # 1. Short-circuit for known symbols (user-curated whitelist)
    if query in _KNOWN_SYMBOLS:
        return f"{query}.NS"

    # 2. If input looks like a bare ticker (1–10 alpha chars), validate before accepting.
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
