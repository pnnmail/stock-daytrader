"""
Whole-market overview — what a person sees after opening Yahoo Finance and
reading the front page: where the indices are, what fear and rates are doing, which
sectors are leading, what is being written about.

WHY NOT SCRAPE THE FRONT PAGE. Yahoo's front page is rendered with JavaScript,
the layout changes without warning, and the same headlines are
available via RSS, which the project already uses (see fetch_news /
fetch_wire_news). Scraping would give exactly the same content at the cost of the most fragile
part of the whole pipeline.

DIVISION OF LABOUR (the same principle as everywhere in the project): ALL NUMBERS are computed
by Python from yfinance — index levels, percentages, VIX, the 10-year yield,
sector breadth. The LLM is not allowed to make up percentages: the numbers are
given to it ready-made, its job is to read the news and connect it with these
numbers. The prompt requires this explicitly (see build_market_prompt).

Run: the /news command without a ticker (telegram_bot/handlers.py::cmd_news)
"""
from typing import Any, Dict, List

import feedparser
import pandas as pd
import yfinance as yf
from urllib.parse import quote

from agents.prompts import build_market_prompt
from utils.ollama import call_ollama_json
from utils.schemas import MARKET_SCHEMA
from utils.helpers import escape_markdown
from utils.market_hours import market_status
from config import NEWS_MODEL

# What we show as "the market". ^RUT (Russell 2000) is not here for completeness:
# divergence between large indices and small caps is itself a signal about the
# breadth of a move, not only its direction.
INDICES = [
    ("^GSPC", "S&P 500"),
    ("^DJI", "Dow Jones"),
    ("^IXIC", "Nasdaq"),
    ("^RUT", "Russell 2000"),
]
VIX_TICKER = "^VIX"
YIELD_TICKER = "^TNX"  # 10-year yield, in percent

SECTORS = [
    ("XLK", "Technology"), ("XLF", "Financials"), ("XLE", "Energy"),
    ("XLV", "Health Care"), ("XLY", "Cons. Discretionary"),
    ("XLP", "Cons. Staples"), ("XLI", "Industrials"),
    ("XLU", "Utilities"), ("XLB", "Materials"),
    ("XLRE", "Real Estate"), ("XLC", "Comm. Services"),
]

# Market queries instead of ticker queries. Google News returns exactly the
# headlines that make it onto Yahoo's front page, but with a stable API.
MARKET_QUERIES = [
    "stock market today",
    "S&P 500",
    "Federal Reserve interest rates",
]
# Index ETFs — unlike the indices themselves, they have a proper
# Yahoo RSS feed with editorial content.
ETF_PROXIES = ["SPY", "QQQ", "DIA"]

NEWS_LOOKBACK_DAYS = 2
MAX_MARKET_NEWS = 25          # this many go into the prompt
MAX_PER_SOURCE = 12

# VIX levels. Computed in Python and given to the model as a ready-made label —
# the interpretation of thresholds must not depend on what the model "remembers" about VIX.
VIX_LEVELS = [
    (15.0, "calm"),
    (20.0, "normal"),
    (30.0, "elevated anxiety"),
    (float("inf"), "panic"),
]


def classify_vix(value):
    if value is None:
        return "no data"
    for threshold, label in VIX_LEVELS:
        if value < threshold:
            return label
    return "panic"


def _pct_change(df, col="Close"):
    """Change of the last bar vs. the previous one, in percent."""
    try:
        if df is None or df.empty or len(df) < 2:
            return None, None
        last = float(df[col].iloc[-1])
        prev = float(df[col].iloc[-2])
        if prev == 0:
            return last, None
        return last, (last / prev - 1) * 100
    except (KeyError, IndexError, TypeError, ValueError):
        return None, None


def _download(tickers, period="7d"):
    """In one batch — one request instead of N (same trick as in screener.py)."""
    try:
        df = yf.download(tickers, period=period, interval="1d", progress=False,
                         auto_adjust=False, threads=False, group_by="ticker")
    except Exception as exc:
        print(f"   ⚠️ Could not load quotes: {str(exc)[:120]}")
        return {}
    if df is None or df.empty:
        return {}

    out = {}
    if isinstance(df.columns, pd.MultiIndex):
        for t in tickers:
            if t in df.columns.get_level_values(0):
                sub = df[t].dropna(how="all")
                if not sub.empty:
                    out[t] = sub
    elif len(tickers) == 1:
        out[tickers[0]] = df.dropna(how="all")
    return out


def fetch_market_snapshot() -> Dict[str, Any]:
    """
    Numeric picture of the market. EVERYTHING here is a deterministic calculation from
    quotes; the model neither generates nor adjusts any of it.
    """
    print("\n📊 MARKET SNAPSHOT")
    all_tickers = [t for t, _ in INDICES] + [VIX_TICKER, YIELD_TICKER] + \
                  [t for t, _ in SECTORS]
    data = _download(all_tickers)

    indices = []
    for ticker, name in INDICES:
        price, chg = _pct_change(data.get(ticker))
        if price is not None:
            indices.append({"ticker": ticker, "name": name,
                            "price": price, "change_pct": chg})
            print(f"   {name:<15} {price:>10.2f}  {chg:+.2f}%" if chg is not None
                  else f"   {name:<15} {price:>10.2f}")

    vix_price, vix_chg = _pct_change(data.get(VIX_TICKER))
    yield_price, yield_chg = _pct_change(data.get(YIELD_TICKER))
    print(f"   {'VIX':<15} {vix_price:>10.2f}  ({classify_vix(vix_price)})"
          if vix_price is not None else f"   {'VIX':<15} no data")
    print(f"   {'10Y yield':<15} {yield_price:>10.2f}%"
          if yield_price is not None else f"   {'10Y yield':<15} no data")

    sectors = []
    for ticker, name in SECTORS:
        price, chg = _pct_change(data.get(ticker))
        if chg is not None:
            sectors.append({"ticker": ticker, "name": name, "change_pct": chg})
    sectors.sort(key=lambda s: s["change_pct"], reverse=True)

    up = sum(1 for s in sectors if s["change_pct"] > 0)
    # Breadth: how many sectors out of how many are up. A direct answer to "is the move
    # broad or are two sectors pulling it" — something a single index does not show.
    breadth = {"up": up, "down": len(sectors) - up, "total": len(sectors)}
    print(f"   Sectors up: {up}/{len(sectors)}")

    status = market_status()
    return {
        "indices": indices,
        "vix": {"value": vix_price, "change_pct": vix_chg,
                "level": classify_vix(vix_price)},
        "yield_10y": {"value": yield_price, "change_pct": yield_chg},
        "sectors": sectors,
        "breadth": breadth,
        "market_status": status.get("status"),
        "extended": status.get("extended"),
    }


def _clean_title(title):
    # Google News appends " - Publisher"; the source comes as a separate field.
    if " - " in title:
        title = title.rsplit(" - ", 1)[0].strip()
    return title.strip()


def fetch_market_news(limit: int = MAX_MARKET_NEWS) -> List[Dict]:
    """
    Market news background from two sources, deduplicated by headline.
    Full article text is NOT fetched: for a market overview, topic coverage matters, not
    depth on each item, and downloading 25 articles is 25 requests
    and minutes of waiting (plus the risk of hitting a rate limit).
    """
    print("\n📰 MARKET NEWS BACKGROUND")
    items, seen = [], set()

    def add(title, source, published):
        key = title.lower()[:80]
        if not title or key in seen:
            return
        seen.add(key)
        items.append({"title": title, "source": source,
                      "published": published, "text": ""})

    for q in MARKET_QUERIES:
        url = (f"https://news.google.com/rss/search?"
               f"q={quote(f'{q} when:{NEWS_LOOKBACK_DAYS}d')}"
               f"&hl=en-US&gl=US&ceid=US:en")
        try:
            feed = feedparser.parse(url)
        except Exception as exc:
            print(f"   ⚠️ Google News '{q}': {str(exc)[:80]}")
            continue
        before = len(items)
        for e in feed.entries[:MAX_PER_SOURCE]:
            src = e.get("source", {})
            add(_clean_title(e.get("title", "")),
                src.get("title", "?") if isinstance(src, dict) else "?",
                e.get("published", ""))
        print(f"   Google News '{q}': +{len(items) - before}")

    for t in ETF_PROXIES:
        url = (f"https://feeds.finance.yahoo.com/rss/2.0/headline?"
               f"s={t}&region=US&lang=en-US")
        try:
            feed = feedparser.parse(url)
        except Exception as exc:
            print(f"   ⚠️ Yahoo RSS {t}: {str(exc)[:80]}")
            continue
        before = len(items)
        for e in feed.entries[:MAX_PER_SOURCE]:
            add(_clean_title(e.get("title", "")), f"Yahoo/{t}",
                e.get("published", ""))
        print(f"   Yahoo RSS {t}: +{len(items) - before}")

    print(f"   Total unique: {len(items)}")
    return items[:limit]


def analyze_market(snapshot: Dict, news_items: List[Dict],
                   model: str = NEWS_MODEL) -> Dict[str, Any]:
    """LLM part: interpretation. The numbers come ready-made, see build_market_prompt."""
    if not news_items:
        return {"overall_tone": "neutral", "summary": "No news found.",
                "key_events": [], "detailed_analysis": "", "themes": [],
                "risks": [], "what_to_watch": []}

    prompt = build_market_prompt(snapshot, news_items)
    analysis = call_ollama_json(prompt, model, json_schema=MARKET_SCHEMA,
                                temperature=0)

    for field, default in (("overall_tone", "neutral"), ("summary", ""),
                           ("detailed_analysis", ""), ("key_events", []),
                           ("themes", []), ("risks", []),
                           ("what_to_watch", [])):
        if field not in analysis or analysis[field] is None:
            analysis[field] = default
    return analysis


TONE_LABEL = {
    "risk_on": "🟢 risk appetite (risk-on)",
    "risk_off": "🔴 flight from risk (risk-off)",
    "mixed": "🟡 mixed",
    "neutral": "⚪ neutral",
}


def format_market_report(snapshot: Dict, analysis: Dict,
                         news_items: List[Dict]) -> str:
    """
    Report: numbers come from snapshot (Python), text from analysis (LLM).
    The split is visible in the report itself: a section with figures and a section with analysis.
    """
    lines = ["📈 *MARKET OVERVIEW*"]

    status = snapshot.get("market_status")
    if status == "open":
        lines.append("_(session in progress" +
                     (", extended hours)_" if snapshot.get("extended") else ")_"))
    else:
        lines.append("_(market closed — figures for the last completed session)_")

    lines.append("")
    lines.append("*Indices*")
    for idx in snapshot.get("indices", []):
        chg = idx.get("change_pct")
        arrow = "▲" if (chg or 0) > 0 else "▼" if (chg or 0) < 0 else "■"
        chg_str = f"{chg:+.2f}%" if chg is not None else "n/a"
        lines.append(f"  {arrow} {escape_markdown(idx['name'])}: "
                     f"{idx['price']:,.2f} ({chg_str})")

    vix = snapshot.get("vix", {})
    if vix.get("value") is not None:
        chg = vix.get("change_pct")
        chg_str = f", {chg:+.1f}% for the session" if chg is not None else ""
        lines.append(f"  😰 VIX: {vix['value']:.2f} — {vix['level']}{chg_str}")

    y = snapshot.get("yield_10y", {})
    if y.get("value") is not None:
        chg = y.get("change_pct")
        chg_str = f" ({chg:+.1f}%)" if chg is not None else ""
        lines.append(f"  🏦 10-year yield: {y['value']:.2f}%{chg_str}")

    sectors = snapshot.get("sectors", [])
    breadth = snapshot.get("breadth", {})
    if sectors:
        lines.append("")
        lines.append(f"*Sectors* ({breadth.get('up', 0)} of "
                     f"{breadth.get('total', 0)})")
        best = sectors[:3]
        worst = sectors[-3:]
        lines.append("  Leaders: " + ", ".join(
            f"{escape_markdown(s['name'])} {s['change_pct']:+.1f}%" for s in best))
        lines.append("  Laggards: " + ", ".join(
            f"{escape_markdown(s['name'])} {s['change_pct']:+.1f}%"
            for s in reversed(worst)))

    lines.append("")
    tone = analysis.get("overall_tone", "neutral")
    lines.append(f"*Market tone:* {TONE_LABEL.get(tone, tone)}")

    if analysis.get("summary"):
        lines.append("")
        lines.append(escape_markdown(analysis["summary"]))

    if analysis.get("key_events"):
        lines.append("")
        lines.append("*Key events*")
        for e in analysis["key_events"][:6]:
            lines.append(f"  • {escape_markdown(str(e))}")

    if analysis.get("themes"):
        lines.append("")
        lines.append("*Themes*")
        for t in analysis["themes"][:5]:
            lines.append(f"  • {escape_markdown(str(t))}")

    if analysis.get("detailed_analysis"):
        lines.append("")
        lines.append("*Analysis*")
        lines.append(escape_markdown(analysis["detailed_analysis"]))

    if analysis.get("risks"):
        lines.append("")
        lines.append("*Risks*")
        for r in analysis["risks"][:5]:
            lines.append(f"  ⚠️ {escape_markdown(str(r))}")

    if analysis.get("what_to_watch"):
        lines.append("")
        lines.append("*What to watch*")
        for w in analysis["what_to_watch"][:5]:
            lines.append(f"  👀 {escape_markdown(str(w))}")

    lines.append("")
    lines.append(f"_Source: {len(news_items)} headlines "
                 f"(Google News + Yahoo RSS), quotes — yfinance._")
    return "\n".join(lines)


def build_market_overview() -> str:
    """Full cycle. Synchronous — call via asyncio.to_thread."""
    snapshot = fetch_market_snapshot()
    news = fetch_market_news()
    analysis = analyze_market(snapshot, news)
    return format_market_report(snapshot, analysis, news)
