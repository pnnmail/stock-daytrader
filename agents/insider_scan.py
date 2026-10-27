"""
Insider purchases (Form 4) across the WHOLE market at once — not per ticker.

WHY NOT BY CIK, AS IN stock_agent3. There `form4_insider.py` pulls Form 4
via the submissions API FOR A KNOWN company — fine when there are
few companies (216 in that sample). Here we need the opposite: "who bought anything
in the last N days" across all ~7000. Walking 7000 tickers one by one is
7000 requests. Instead, EDGAR full-text search returns ALL Form 4s for
a date range in ONE parameterised query (with pagination) — verified
live on 2026-08-05: 559 filings in 2 days, market-wide, without naming a company.
Then only the XML of actually filed forms has to be downloaded (usually 150-300 a
day), instead of guessing for each of 7000 tickers whether it had a filing.

THE TICKER IS RIGHT IN THE DOCUMENT. issuerTradingSymbol is in the Form 4 XML itself
(verified on a live Kennametal filing) — no separate
CIK->ticker mapping and cache are needed.

TRANSACTION FILTER: only code=P (open-market purchase) WITHOUT the
aff10b5One flag (not under a 10b5-1 plan) — same logic as in stock_agent3,
the XML parsing is ported from there almost unchanged (parse_form4).

Not meant to be run as a module — used from
telegram_bot/handlers.py::cmd_insiders. Raw XML is cached on disk
(raw_cache/form4/) — repeated windows with overlapping dates do not re-download
filings already seen.
"""
import os
import re
import time
import hashlib
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

from config import INSIDER_FORM4_CACHE_DIR

USER_AGENT = os.environ.get("SEC_USER_AGENT", "stock-daytrader research contact@example.com")
FTS_URL = "https://efts.sec.gov/LATEST/search-index"
REQUEST_DELAY_SEC = 0.15  # SEC limit is 10 req/s, keep a margin
RAW_CACHE_DIR = INSIDER_FORM4_CACHE_DIR
PAGE_SIZE = 100

# Sanity guard against dirty data IN THE FILED FORM ITSELF (not a parsing
# bug — verified live on 2026-08-05: REBN showed $23.6 BILLION in a single
# transaction because of price=$180,000/share in the XML itself — real share prices
# are almost never in that range, apart from rare exceptions
# like Berkshire class A. Such trades are NOT silently dropped — they are flagged
# separately so they don't get lost in the list and don't distort the sort by amount.
SANE_MAX_PRICE_PER_SHARE = 50_000
SANE_MAX_TRADE_USD = 500_000_000
# Ticker placeholders that actually occur in issuerTradingSymbol
# (verified live: "N/A" is a literal string, not an empty field).
_TICKER_PLACEHOLDERS = {"", "N/A", "NONE", "-"}


def _get(url):
    time.sleep(REQUEST_DELAY_SEC)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read()


def _search_form4(start_date, end_date, max_results=5000):
    """All Form 4s for a date range, market-wide, with pagination (EDGAR returns
    at most 100 hits per page)."""
    fetched, offset = 0, 0
    while fetched < max_results:
        params = {"forms": "4", "startdt": start_date, "enddt": end_date,
                  "from": offset}
        url = f"{FTS_URL}?{urllib.parse.urlencode(params)}"
        try:
            data = _get(url)
        except Exception as exc:
            print(f"   ⚠️ EDGAR request failed (offset={offset}): {str(exc)[:100]}")
            return
        import json
        doc = json.loads(data)
        hits = doc.get("hits", {}).get("hits", [])
        if not hits:
            return
        for h in hits:
            yield h.get("_source", {}), h.get("_id", "")
            fetched += 1
            if fetched >= max_results:
                return
        offset += len(hits)
        total = doc.get("hits", {}).get("total", {}).get("value", 0)
        if offset >= min(total, max_results):
            return


def _filing_doc_url(source, _id):
    """URL of the XML document itself from the search metadata."""
    if ":" not in _id:
        return None
    adsh, filename = _id.split(":", 1)
    cik = (source.get("ciks") or [""])[0].lstrip("0")
    adsh_nodash = adsh.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{adsh_nodash}/{filename}"


def _fetch_xml_cached(url):
    os.makedirs(RAW_CACHE_DIR, exist_ok=True)
    key = hashlib.md5(url.encode()).hexdigest()[:20]
    path = os.path.join(RAW_CACHE_DIR, f"{key}.xml")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    try:
        xml_text = _get(url).decode("utf-8", errors="ignore")
    except Exception:
        xml_text = ""
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml_text)
    return xml_text


def parse_form4(xml_text):
    """
    Ported from stock_agent3/data_layer/form4_insider.py::parse_form4,
    extended with ticker/name/title (they were not in the original version —
    there the company was known in advance by CIK, here it is the other way round).
    Returns a list of transactions (dicts); one Form 4 may contain several.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    def text(el, path, default=""):
        if el is None:
            return default
        node = el.find(path)
        return node.text.strip() if node is not None and node.text else default

    ticker = text(root, "issuer/issuerTradingSymbol")
    issuer_name = text(root, "issuer/issuerName")
    owner_name = text(root, "reportingOwner/reportingOwnerId/rptOwnerName")

    owner_rel = root.find("reportingOwner/reportingOwnerRelationship")
    is_officer = text(owner_rel, "isOfficer", "0") in ("1", "true")
    is_director = text(owner_rel, "isDirector", "0") in ("1", "true")
    is_ten_pct = text(owner_rel, "isTenPercentOwner", "0") in ("1", "true")
    officer_title = text(owner_rel, "officerTitle")

    aff_10b5 = text(root, "aff10b5One", "0") in ("1", "true")

    trades = []
    for tx in root.findall(".//nonDerivativeTransaction"):
        code = text(tx, "transactionCoding/transactionCode")
        shares_s = text(tx, "transactionAmounts/transactionShares/value", "0")
        price_s = text(tx, "transactionAmounts/transactionPricePerShare/value", "0")
        date_s = text(tx, "transactionDate/value")
        try:
            shares, price = float(shares_s), float(price_s)
        except ValueError:
            shares, price = 0.0, 0.0
        usd = shares * price
        # A sanity flag, not dropping — see SANE_MAX_* above. Further down
        # the chain (scan_insider_buys) such trades are excluded from the total,
        # but do not disappear without a trace.
        suspicious = price > SANE_MAX_PRICE_PER_SHARE or usd > SANE_MAX_TRADE_USD
        trades.append({
            "ticker": ticker, "issuer_name": issuer_name, "owner_name": owner_name,
            "officer_title": officer_title, "is_officer": is_officer,
            "is_director": is_director, "is_ten_pct_owner": is_ten_pct,
            "code": code, "shares": shares, "price": price, "usd": usd,
            "aff_10b5_one": aff_10b5, "date": date_s, "suspicious": suspicious,
        })
    return trades


def _role_label(tr):
    """Readable insider role — previously 10% owners (funds/LLCs without
    officer_title) showed up as '?', although is_ten_pct_owner was already
    known."""
    if tr["officer_title"]:
        return tr["officer_title"]
    if tr["is_director"]:
        return "director"
    if tr["is_ten_pct_owner"]:
        return "10%+ owner"
    return "?"


def scan_insider_buys(days_back, min_usd=0, role_contains=None, progress_callback=None):
    """
    Open-market purchases (code=P, NOT 10b5-1) over the last days_back
    days across the WHOLE market. role_contains is a case-insensitive substring of
    officer_title (e.g. "CEO", "Chief Executive") — filters by the insider's
    role, not just "officer/director in general".

    Returns (by_ticker, filings_scanned, suspicious_count),
    where by_ticker is a dict {ticker: {"usd": total, "trades": [trades]}}, already
    WITHOUT price-suspicious trades (see SANE_MAX_*) — they do not count
    in the total and must not mask real purchases at the top of the list.
    """
    end = datetime.now()
    start = end - timedelta(days=days_back)
    start_s, end_s = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    print(f"   Searching Form 4 for {start_s}..{end_s} across the whole market...")
    seen_ids = set()
    n_filings = n_suspicious = 0
    per_ticker = {}

    for source, _id in _search_form4(start_s, end_s):
        if _id in seen_ids:
            continue
        seen_ids.add(_id)
        n_filings += 1
        url = _filing_doc_url(source, _id)
        if not url:
            continue
        xml_text = _fetch_xml_cached(url)
        if not xml_text:
            continue
        for tr in parse_form4(xml_text):
            if tr["code"] != "P" or tr["aff_10b5_one"]:
                continue
            if tr["ticker"].upper() in _TICKER_PLACEHOLDERS:
                continue
            if tr["suspicious"]:
                n_suspicious += 1
                continue
            if role_contains and role_contains.lower() not in (tr["officer_title"] or "").lower():
                continue
            slot = per_ticker.setdefault(tr["ticker"], {"usd": 0.0, "trades": []})
            slot["usd"] += tr["usd"]
            slot["trades"].append(tr)
        if n_filings % 50 == 0:
            print(f"      ...{n_filings} filings scanned")
            if progress_callback:
                progress_callback(n_filings, len(per_ticker))

    result = {t: v for t, v in per_ticker.items() if v["usd"] >= min_usd}
    print(f"   Filings scanned: {n_filings} (price-suspicious "
          f"excluded: {n_suspicious}), tickers with purchases "
          f">= ${min_usd:,.0f}: {len(result)}")
    return result, n_filings, n_suspicious
