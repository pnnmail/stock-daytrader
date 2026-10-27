"""
Preliminary ticker screen (/screen).

Idea: the full pipeline costs 4-5 LLM calls per ticker (~2 minutes), so
running dozens of candidates through it makes no sense. The screener uses ONLY
the deterministic part of the system — the same functions as the main analysis, but without
a single model call. That is seconds per ticker instead of minutes.

What the screener can NOT compute (and so does not claim a verdict):
  News  ±2  — needs NEWS_MODEL
(Order Flow ±2 used to be here too, but was removed from the formula entirely — see
SCORING_VERSION v3_no_fake_orderflow in utils/signals.py: it was not
a deterministic part and not an LLM factor with real data, but an LLM
making up order flow from hourly candles without a single trade inside.)
So a candidate with a deterministic score of +3 has a chance to reach the +4 threshold,
while a candidate with 0 has almost none. The ranking is built exactly on this principle:
we show those who are closest to a signal.
"""
import json
import os
import time
import urllib.request
from datetime import datetime

import pandas as pd
import yfinance as yf

from config import LONG_THRESHOLD, SHORT_THRESHOLD, EXTENDED_HOURS_ENABLED
from data.fetcher import (fetch_price_data, fetch_relative_context,
                          fetch_extended_history, fetch_previous_day_levels)
from data.indicators import (compute_indicators, compute_intraday_levels,
                             compute_volume_profile, compute_moving_averages,
                             check_data_quality, PIVOT_GAP_THRESHOLD_PCT)
from data.stumpy_finder import find_similar_patterns
from utils.helpers import escape_markdown, safe_format_float, load_state, save_state
from utils.signals import (compute_factor_scores, calculate_entry_levels,
                           check_entry_filters, EARNINGS_BLOCK_DAYS)

# The maximum that what is not yet computed at this stage can add —
# this is how much the deterministic score can still grow after the full analysis.
# Order Flow was never included here (even before its removal from the formula —
# see SCORING_VERSION v3_no_fake_orderflow in utils/signals.py — it was already disabled in the cheap
# scan, orderflow=None was always passed). The value 4 is
# News(±2)+STUMPY(±2) for full_market_sweep, where STUMPY is computed only
# for leaders AFTER ranking; for screen_ticker (where STUMPY is already computed
# on input) the real headroom is only News(±2), and the same threshold here gives
# false "ready for analysis" at gap=3 — a separate bug, not touched in this
# change (not about Order Flow).
LLM_FACTORS_MAX = 4

SOURCES = {
    "trending": "trending on Yahoo",
    "actives": "most active",
    "gainers": "top gainers",
    "losers": "top losers",
    "small_gainers": "small-cap gainers",
}


def _fetch_trending(limit):
    """Yahoo's public trending endpoint (yfinance does not have it)."""
    url = f"https://query1.finance.yahoo.com/v1/finance/trending/US?count={limit * 2}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    data = json.loads(urllib.request.urlopen(req, timeout=20).read())
    return [q.get("symbol") for q in data["finance"]["result"][0]["quotes"]]


def _fetch_predefined(name, limit):
    res = yf.screen(name, count=limit * 2)
    quotes = res.get("quotes", []) if isinstance(res, dict) else res
    return [q.get("symbol") for q in quotes]


def get_candidate_tickers(source="trending", limit=15):
    """
    Candidate tickers from Yahoo. Crypto, forex and indices are filtered out:
    the pipeline is designed for stocks (volumes, earnings, valuation metrics).
    """
    try:
        if source == "trending":
            symbols = _fetch_trending(limit)
        else:
            mapping = {"actives": "most_actives", "gainers": "day_gainers",
                       "losers": "day_losers", "small_gainers": "small_cap_gainers"}
            symbols = _fetch_predefined(mapping.get(source, "most_actives"), limit)
    except Exception as e:
        print(f"   ⚠️ Could not get the {source} list: {str(e)[:120]}")
        return []

    clean = []
    for s in symbols:
        if not s or any(ch in s for ch in ("-", "=", "^")):
            continue          # BTC-USD, JPY=X, ^GSPC
        if s in clean:
            continue
        clean.append(s)
    return clean[:limit]


def screen_ticker(ticker):
    """
    Deterministic assessment of one ticker — without LLM calls.
    Returns a dict or None if there is not enough data.
    """
    h, d = fetch_price_data(ticker)
    if h is None or h.empty or len(h) < 20:
        return None
    # Stocks with halted trading are filtered out up front: there is nothing to compute their levels from
    data_ok, _ = check_data_quality(h)
    if not data_ok:
        return None

    indicators = compute_indicators(h, d)
    if not indicators.get("price"):
        return None

    history = fetch_extended_history(ticker)
    hourly = history["hourly"]
    if hourly.empty:
        hourly = h
    levels = compute_intraday_levels(hourly)
    if not history["daily"].empty:
        levels.update(compute_moving_averages(history["daily"]))
    vprofile = compute_volume_profile(hourly)
    relctx = fetch_relative_context(ticker)

    stumpy_data = {}
    try:
        res = find_similar_patterns(ticker, h)
        if res and res.get("n_3d"):
            stumpy_data = {"win_rate": res.get("win_rate_3d") or 50,
                           "avg_return": res.get("avg_ret_3d") or 0,
                           "n_samples": res.get("n_3d") or 0}
    except Exception:
        pass

    # news_analysis=None -> this factor gives 0. That is exactly what is needed: we see
    # the pure deterministic contribution (Order Flow is removed from the formula entirely,
    # see SCORING_VERSION v3_no_fake_orderflow).
    scores, reasons = compute_factor_scores(indicators, levels, None,
                                            stumpy_data, relctx)
    det = scores["total"]

    direction = "LONG" if det >= 0 else "SHORT"
    prev = fetch_previous_day_levels(ticker)
    liquidity = {"prev_high": prev.get("prev_high"), "prev_low": prev.get("prev_low")}
    plan = calculate_entry_levels(direction, indicators["price"], levels, vprofile, liquidity)
    ok, why = check_entry_filters(direction, levels, plan, indicators, relctx)

    price = indicators["price"]
    atr = levels.get("atr_hourly") or 0
    return {
        "ticker": ticker,
        "price": price,
        "det": det,
        "technical": scores["technical"],
        "stumpy": scores["stumpy"],
        # momentum is what actually goes into det; vwap next to it as diagnostics.
        # Since v2 there is no separate VWAP score, it is inside the momentum group
        # (see SCORING_VERSION in utils/signals.py).
        "momentum": scores["momentum"],
        "vwap": scores["vwap"],
        "rsi": indicators.get("rsi"),
        "volume_ratio": relctx.get("volume_ratio") or 0,
        "atr_pct": (atr / price * 100) if price else 0,
        "direction": direction,
        "rr1": plan.get("risk_reward_1") or 0,
        "entry_ok": ok,
        "entry_why": why,
        # How much is missing to the threshold and whether news and flow can realistically make it up
        "gap": (LONG_THRESHOLD - det) if det >= 0 else (det - SHORT_THRESHOLD),
    }


ALL_SOURCES = ["most_actives", "day_gainers", "day_losers", "small_cap_gainers",
               "growth_technology_stocks", "undervalued_growth_stocks",
               "most_shorted_stocks", "aggressive_small_caps", "undervalued_large_caps"]


def get_all_candidates(limit=100):
    """
    Combines all Yahoo screeners plus trending. Gives up to ~200 unique stocks.
    Order is preserved: trending and most active first, then the rest.
    """
    seen, out = set(), []

    def add(symbols):
        for s in symbols:
            if not s or any(ch in s for ch in ("-", "=", "^")) or s in seen:
                continue
            seen.add(s)
            out.append(s)

    try:
        add(_fetch_trending(30))
    except Exception as e:
        print(f"   ⚠️ trending unavailable: {str(e)[:80]}")
    for src in ALL_SOURCES:
        if len(out) >= limit:
            break
        try:
            add(_fetch_predefined(src, 25))
        except Exception as e:
            print(f"   ⚠️ {src}: {str(e)[:60]}")
    return out[:limit]


def _batch_download(tickers, **kwargs):
    """Downloads in a batch and splits by ticker. One request instead of N — ~7x faster."""
    if not tickers:
        return {}
    raw = yf.download(tickers, progress=False, group_by="ticker", **kwargs)
    result = {}
    for t in tickers:
        try:
            df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            df = df.dropna(how="all")
            if not df.empty:
                result[t] = df
        except (KeyError, TypeError):
            continue
    return result


def screen_from_batch(ticker, hourly, daily):
    """
    A cheap assessment FROM ALREADY LOADED data — without a single network request.
    STUMPY is deliberately not computed here: it needs a separate download of a year
    of hourly data per ticker, which is unacceptably expensive for a hundred candidates.
    It is added later only for the leaders (see mass_screen).
    """
    if hourly is None or hourly.empty or len(hourly) < 20:
        return None
    ok, _ = check_data_quality(hourly)
    if not ok:
        return None

    indicators = compute_indicators(hourly, daily)
    price = indicators.get("price") or 0
    if not price:
        return None

    levels = compute_intraday_levels(hourly)
    if daily is not None and not daily.empty:
        levels.update(compute_moving_averages(daily))
    vprofile = compute_volume_profile(hourly)

    # Volume is computed from the batch's daily data, not a separate request
    volume_ratio = 0
    dollar_volume = 0
    prev_high = prev_low = None
    if daily is not None and len(daily) > 1:
        vol = daily["Volume"].dropna()
        vol = vol[vol > 0]
        if len(vol) > 1:
            volume_ratio = float(vol.iloc[-1] / vol.iloc[:-1].tail(20).mean())
        if len(vol) > 0:
            # Absolute dollar turnover — separate from volume_ratio (that one is
            # relative; a cheap stock with small absolute volume easily
            # passes the relative filter but carries real slippage risk)
            dollar_volume = price * float(vol.iloc[-1])
        prev_high = float(daily["High"].iloc[-2])
        prev_low = float(daily["Low"].iloc[-2])

    relctx = {"volume_ratio": volume_ratio}
    scores, _ = compute_factor_scores(indicators, levels, None, {}, relctx)
    det = scores["total"]

    direction = "LONG" if det >= 0 else "SHORT"
    plan = calculate_entry_levels(direction, price, levels, vprofile,
                                  {"prev_high": prev_high, "prev_low": prev_low})
    entry_ok, why = check_entry_filters(direction, levels, plan, indicators, relctx)

    atr = levels.get("atr_hourly") or 0
    return {
        "ticker": ticker, "price": price, "det": det,
        "technical": scores["technical"], "stumpy": 0,
        "momentum": scores["momentum"], "vwap": scores["vwap"],
        "rsi": indicators.get("rsi"), "volume_ratio": volume_ratio,
        "dollar_volume": dollar_volume,
        "atr_pct": (atr / price * 100) if price else 0,
        "direction": direction, "rr1": plan.get("risk_reward_1") or 0,
        "entry_ok": entry_ok, "entry_why": why,
        "gap": (LONG_THRESHOLD - det) if det >= 0 else (det - SHORT_THRESHOLD),
        # Already computed by compute_intraday_levels — if a gap carried the price far from
        # the base session's Pivot, R1/R2/S1/S2 (and the stop/targets derived from them) are unusable
        # (see PIVOT_GAP_THRESHOLD_PCT in data/indicators.py). Pass it
        # along so the intermediate filter can weed out such cases.
        "pivot_reliable": levels.get("pivot_reliable", True),
        "pivot_gap_pct": levels.get("pivot_gap_pct", 0),
    }


# ═══════════════════════════════════════════════════════════════════════════
# FULL STOCK LIST (for /mass) — unlike get_all_candidates() above
# (live Yahoo screeners most_actives/day_gainers/... — essentially the same
# top-N on every call, updated only as the market moves), this is the WHOLE
# NASDAQ+NYSE/AMEX stock list published by the exchange itself. It can be
# paged through with a cursor in batches and actually reach every ticker, instead of seeing
# the same top 100-200 on every /mass.
#
# /screen (get_candidate_tickers/run_screen above) does not use this block and
# is not affected — it stays on the live Yahoo screeners as before.
# ═══════════════════════════════════════════════════════════════════════════
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
UNIVERSE_CACHE_HOURS = 24


def _fetch_full_universe():
    """
    Pulls the official NASDAQ and NYSE/AMEX lists (updated by the exchange daily,
    ~5500 + ~7500 rows). Drops ETFs and test entries (we need stocks, not
    funds), as well as tickers with "." "$" "+" "-" "=" "^" — these are preferred
    shares/warrants/units/share classes and (for another source) crypto pairs;
    for day trading the bulk of these are not needed and often break yfinance requests.
    """
    def _parse(url, symbol_idx, etf_idx, test_idx):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        text = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", errors="ignore")
        out = []
        for line in text.splitlines()[1:]:  # the first line is the header
            parts = line.split("|")
            if len(parts) <= max(symbol_idx, etf_idx, test_idx):
                continue  # a footer like "File Creation Time: ..." — shorter than real rows
            if parts[test_idx] == "Y" or parts[etf_idx] == "Y":
                continue
            out.append(parts[symbol_idx].strip())
        return out

    symbols = []
    symbols += _parse(NASDAQ_LISTED_URL, symbol_idx=0, etf_idx=6, test_idx=3)
    symbols += _parse(OTHER_LISTED_URL, symbol_idx=0, etf_idx=4, test_idx=6)

    clean, seen = [], set()
    for s in symbols:
        if not s or s in seen or any(ch in s for ch in (".", "$", "+", "-", "=", "^")):
            continue
        seen.add(s)
        clean.append(s)
    clean.sort()  # stable order — otherwise the cursor "drifts" between cache updates
    return clean


def _get_universe():
    """
    Cache of the full list in state.json (an already mounted file, see
    utils/helpers.py). Refreshed once every UNIVERSE_CACHE_HOURS, not on every
    /mass — otherwise an extra network request and a delay for nothing.
    """
    state = load_state()
    cache = state.get("_mass_universe") or {}
    stale = True
    if cache.get("fetched_at"):
        try:
            age_h = (datetime.now() - datetime.fromisoformat(cache["fetched_at"])).total_seconds() / 3600
            stale = age_h > UNIVERSE_CACHE_HOURS
        except (ValueError, TypeError):
            stale = True

    if not stale and cache.get("tickers"):
        return cache["tickers"]

    try:
        tickers = _fetch_full_universe()
    except Exception as e:
        print(f"   ⚠️ Could not refresh the full stock list: {str(e)[:120]}")
        return cache.get("tickers", [])  # a stale cache is better than nothing

    state["_mass_universe"] = {"tickers": tickers, "fetched_at": datetime.now().isoformat()}
    save_state(state)
    print(f"   ✅ Full stock list refreshed: {len(tickers)} tickers")
    return tickers


def _next_universe_batch(count):
    """
    A cursor over the full list (also in state.json, key "_mass_cursor") — each
    /mass call continues from where the previous one stopped, instead of
    taking the same tickers every time. When the end of the
    list is reached, the cursor wraps around to the start.

    Returns (tickers_for_this_batch, cursor_before, cursor_after, universe_size).
    """
    universe = _get_universe()
    total = len(universe)
    if total == 0:
        return [], 0, 0, 0

    state = load_state()
    cursor = state.get("_mass_cursor", 0) % total
    count = min(count, total)

    end = cursor + count
    batch = universe[cursor:end] if end <= total else universe[cursor:] + universe[:end - total]
    new_cursor = end % total

    state["_mass_cursor"] = new_cursor
    save_state(state)
    return batch, cursor, new_cursor, total


def mass_screen(limit=100, batch_size=10, status_callback=None, batch_callback=None,
                enrich_top=12):
    """
    Mass pass: a two-stage funnel.

    1) A cheap pass over all candidates on BATCH-loaded data —
       not a single per-ticker request.
    2) Leaders are enriched with STUMPY (it costs a request for a year of data per ticker,
       so only for those who have a chance).

    Tickers are taken with a cursor from the FULL market stock list (see
    _next_universe_batch), not from live Yahoo screeners — otherwise repeated
    calls almost always returned the same top-N.

    batch_callback(number, total, batch_results, market_position, market_size)
    is called after each batch — so the report comes in as it is ready and it is
    visible where the pass currently is relative to the whole market.
    """
    tickers, cursor_start, cursor_end, universe_total = _next_universe_batch(limit)
    if not tickers:
        return [], {"cursor_start": cursor_start, "cursor_end": cursor_end, "universe_total": universe_total}

    print(
        f"\n🔎 MASS SCREENING: {len(tickers)} stocks (position {cursor_start+1}-{cursor_end} "
        f"of {universe_total}) in batches of {batch_size}"
    )
    results = []
    total_batches = (len(tickers) + batch_size - 1) // batch_size

    for bi in range(total_batches):
        chunk = tickers[bi * batch_size:(bi + 1) * batch_size]
        if status_callback:
            status_callback(f"batch {bi+1}/{total_batches}: {', '.join(chunk)}")

        hourly_map = _batch_download(chunk, period="5d", interval="1h",
                                     prepost=EXTENDED_HOURS_ENABLED)
        daily_map = _batch_download(chunk, period="3mo", interval="1d")

        batch_res = []
        for t in chunk:
            try:
                r = screen_from_batch(t, hourly_map.get(t), daily_map.get(t))
                if r:
                    batch_res.append(r)
            except Exception as e:
                print(f"   {t}: error {str(e)[:60]}")

        batch_res.sort(key=lambda x: (not x["entry_ok"], x["gap"], -x["volume_ratio"]))
        results.extend(batch_res)
        print(f"   batch {bi+1}/{total_batches}: processed {len(batch_res)}/{len(chunk)}")

        consumed = min((bi + 1) * batch_size, len(tickers))
        market_pos = (cursor_start + consumed - 1) % universe_total + 1 if universe_total else 0
        if batch_callback:
            batch_callback(bi + 1, total_batches, batch_res, market_pos, universe_total)

    # Stage 2: STUMPY for the leaders only
    results.sort(key=lambda x: (not x["entry_ok"], x["gap"], -x["volume_ratio"]))
    top = [r for r in results if r["entry_ok"]][:enrich_top]
    if top and status_callback:
        status_callback(f"computing STUMPY for {len(top)} leaders...")
    for r in top:
        try:
            h, _ = fetch_price_data(r["ticker"])
            res = find_similar_patterns(r["ticker"], h)
            if res and res.get("n_3d"):
                sd = {"win_rate": res.get("win_rate_3d") or 50,
                      "avg_return": res.get("avg_ret_3d") or 0,
                      "n_samples": res.get("n_3d") or 0}
                from utils.signals import compute_factor_scores as _cfs
                # Recompute only the STUMPY factor and add it to the already known score
                s, _ = _cfs({"rsi": 50, "macd": 0, "macd_signal": 0, "price": r["price"],
                             "bb_lower": r["price"] * 0.95, "bb_upper": r["price"] * 1.05},
                            {}, None, sd, {"volume_ratio": r["volume_ratio"]})
                r["stumpy"] = s["stumpy"]
                r["det"] += s["stumpy"]
                r["gap"] = ((LONG_THRESHOLD - r["det"]) if r["det"] >= 0
                            else (r["det"] - SHORT_THRESHOLD))
        except Exception as e:
            print(f"   STUMPY {r['ticker']}: {str(e)[:60]}")

    results.sort(key=lambda x: (not x["entry_ok"], x["gap"], -x["volume_ratio"]))
    market_range = {"cursor_start": cursor_start, "cursor_end": cursor_end,
                    "universe_total": universe_total}
    return results, market_range


# Pause between batches during a full market pass (full_market_sweep) — not
# about data volume (batches are economical anyway) but about request FREQUENCY: without a pause
# ~278 batches may fly at Yahoo faster than in 40-50 minutes and hit
# a rate limit for the rest of the session. 1.5s × ~278 batches ≈ +7 minutes to the sweep.
BATCH_PAUSE_SEC = 1.5


def full_market_sweep(batch_size=25, progress_every=20, status_callback=None, progress_callback=None):
    """
    A one-off pass over the WHOLE stock list (for /autoscan) — unlike
    /mass, this is not cursor pagination over a slice but the whole market at once
    in one call (~6900 stocks). It does NOT touch the /mass cursor (_mass_cursor) — this is
    an independent operation, /mass afterwards continues from where it was.

    Data is downloaded in batches of `batch_size` tickers (see _batch_download) —
    THIS MATTERS: one giant request for the whole list may get cut off by Yahoo
    or not return everything, so we split into many small requests instead of one
    big one, rather than cutting the stock list itself.

    STUMPY is deliberately not computed here (unlike mass_screen) — it is
    a separate request for a year of data FOR EVERY ticker, for ~7000 tickers that is
    unacceptably expensive. For candidates selected in rank_long_candidates
    it will be computed by the subsequent full /scan anyway.

    progress_callback(checked, total, LONG_candidates_found) is called
    not on every batch (that would be ~280 Telegram messages) but once every
    `progress_every` batches — so you can see where the pass is, without spam.

    BATCH_PAUSE_SEC between batches — without it ~280 batches (556 requests to Yahoo)
    may fly faster than the data volume suggests and get a
    "Too Many Requests" ban for the rest of the session (seen in practice:
    the rate limit then got in the way even of a single /scan run AFTER the sweep).
    """
    universe = _get_universe()
    total = len(universe)
    if total == 0:
        return []

    print(f"\n🌐 FULL MARKET PASS: {total} stocks in batches of {batch_size}")
    results = []
    total_batches = (total + batch_size - 1) // batch_size

    for bi in range(total_batches):
        chunk = universe[bi * batch_size:(bi + 1) * batch_size]
        hourly_map = _batch_download(chunk, period="5d", interval="1h",
                                     prepost=EXTENDED_HOURS_ENABLED)
        daily_map = _batch_download(chunk, period="3mo", interval="1d")

        for t in chunk:
            try:
                r = screen_from_batch(t, hourly_map.get(t), daily_map.get(t))
                if r:
                    results.append(r)
            except Exception as e:
                print(f"   {t}: error {str(e)[:60]}")

        checked = min((bi + 1) * batch_size, total)
        if status_callback:
            status_callback(f"batch {bi+1}/{total_batches}: checked {checked}/{total}")
        print(f"   batch {bi+1}/{total_batches}: checked {checked}/{total}")

        is_last = (bi + 1) == total_batches
        if progress_callback and ((bi + 1) % progress_every == 0 or is_last):
            long_count = sum(1 for r in results if r["direction"] == "LONG" and r["entry_ok"])
            progress_callback(checked, total, long_count)

        if not is_last:
            time.sleep(BATCH_PAUSE_SEC)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# INTERMEDIATE SCREEN between the cheap sweep (7000→~1500 LONG+entry_ok) and
# the expensive stage (LLM+STUMPY on the top 10/via /queue). Without it
# /queue would have to be called ~150 times to go through the whole list of LONG
# candidates, most of which are obviously weak. All fields here are already
# computed by screen_from_batch — not a single new request to Yahoo.
#
# Deliberately accounted for — what is NOT here:
# - Earnings proximity (days_until_earnings) — requires a separate network
#   request FOR EVERY ticker (the same source that already hit the rate limit
#   on 52W/value investing). For 1500 tickers this is
#   exactly the expensive operation this stage was meant to exclude.
#   The sensible place for it is the narrow top-N before the expensive stage, not
#   here; not added now, a separate task if wanted.
# - The real position in the 52-week range — same reason (per-ticker
#   .info). As a cheap proxy pivot_gap_pct is already used (the gap
#   relative to the previous session's Pivot) — not the same thing, but free.
# - "Data freshness" — already checked by check_data_quality() on entry to
#   screen_from_batch (see return None above), no separate criterion needed.
# ═══════════════════════════════════════════════════════════════════════════
PREFILTER_MIN_VOLUME_RATIO = 0.8
PREFILTER_MIN_ATR_PCT = 0.5
PREFILTER_MAX_ATR_PCT = 5.0
PREFILTER_MIN_PRICE = 3.0
PREFILTER_MIN_RR = 1.5
PREFILTER_MIN_DOLLAR_VOLUME = 1_000_000  # $1M/day — protection against slippage on entry/exit

# Defaults if /refilter is called without overrides. The keys are the same
# as used inside apply_prefilter, NOT the short names from the command
# (those are mapped in parse_refilter_args via FILTER_ARG_MAP).
DEFAULT_PREFILTER = {
    "volume_ratio": PREFILTER_MIN_VOLUME_RATIO,
    "atr_min": PREFILTER_MIN_ATR_PCT,
    "atr_max": PREFILTER_MAX_ATR_PCT,
    "price": PREFILTER_MIN_PRICE,
    "rr1": PREFILTER_MIN_RR,
    "dollar_volume": PREFILTER_MIN_DOLLAR_VOLUME,
    "gap_pct": PIVOT_GAP_THRESHOLD_PCT,
}


def apply_prefilter(candidates, overrides=None):
    """
    A strict screen by criteria already computed (no new requests).
    Each candidate is checked against ALL conditions (not dropped at the first
    failure) — needed for fail_counts: if one criterion cuts a
    disproportionate number of candidates, it should be softened or moved into
    ranking rather than a hard cut-off. Without this visibility the risk is exactly
    what was already caught with the gap on HURN/GDDY — only at the level of the whole funnel: the
    more conditions are ANDed, the higher the chance of a quiet day with 0 candidates not
    because the market is empty but because the conditions almost certainly cut everything.

    CONFIRMED ON LIVE DATA: originally there was also a hard score threshold
    (abs(det) >= 2) — on a real run it alone cut 1592 of 1654 (96%),
    an order of magnitude more than all other criteria combined. Reason: at the
    sweep stage Order Flow/news/STUMPY are not computed yet (that is the expensive stage),
    only Technical (±3) and VWAP (±1) are available — a real maximum of ±4, not
    ±10 as in the full analysis, so "abs(score)>=2" required half of
    ALL the score theoretically available at this stage. Removed as a hard
    cut-off — sorting (see sort_candidates) already prioritises
    stronger scores, no need to duplicate that with a cut-off.

    overrides — a dict with DEFAULT_PREFILTER keys, overrides the defaults
    FOR THIS CALL ONLY (the constants themselves do not change) — so /refilter can
    tune thresholds without editing code. The gap is now compared with the raw
    pivot_gap_pct (already stored on each candidate), not with the boolean
    pivot_reliable — which is what allows changing the threshold retroactively.

    Returns (passed, fail_counts) — fail_counts is a dict
    {criterion: how many candidates failed on it}, not necessarily summing
    to total (one candidate can fail several criteria at once).
    """
    cfg = dict(DEFAULT_PREFILTER)
    if overrides:
        cfg.update(overrides)

    passed = []
    fail_counts = {}

    for c in candidates:
        reasons = []
        if c.get("volume_ratio", 0) < cfg["volume_ratio"]:
            reasons.append("volume_ratio")
        atr_pct = c.get("atr_pct", 0)
        if not (cfg["atr_min"] <= atr_pct <= cfg["atr_max"]):
            reasons.append("atr_pct")
        if c.get("price", 0) < cfg["price"]:
            reasons.append("price")
        if c.get("rr1", 0) < cfg["rr1"]:
            reasons.append("rr1")
        if c.get("dollar_volume", 0) < cfg["dollar_volume"]:
            reasons.append("dollar_volume")
        if c.get("pivot_gap_pct", 0) > cfg["gap_pct"]:
            reasons.append("pivot_gap")

        if reasons:
            for r in reasons:
                fail_counts[r] = fail_counts.get(r, 0) + 1
        else:
            passed.append(c)

    return passed, fail_counts


# Fields the final list can be sorted by (/refilter sort=...).
# The value is (value extraction function, descending). descending=True
# means "more is better" (volume, liquidity, R:R, volatility); for gap
# it is the opposite — less means closer to the entry threshold, hence False.
SORT_FIELDS = {
    "gap": (lambda c: c.get("gap", 99), False),
    "volume_ratio": (lambda c: c.get("volume_ratio", 0), True),
    "dollar_volume": (lambda c: c.get("dollar_volume", 0), True),
    "rr1": (lambda c: c.get("rr1", 0), True),
    "atr_pct": (lambda c: c.get("atr_pct", 0), True),
}


def sort_candidates(candidates, sort_key="gap"):
    """
    Sorts candidates by the chosen field (see SORT_FIELDS). The secondary
    criterion is volume descending, unless it is itself the primary sort
    field (otherwise it would just duplicate the primary).
    """
    field_fn, descending = SORT_FIELDS.get(sort_key, SORT_FIELDS["gap"])

    def key(c):
        primary = field_fn(c)
        secondary = -c.get("volume_ratio", 0) if sort_key != "volume_ratio" else 0
        return (-primary if descending else primary, secondary)

    return sorted(candidates, key=key)


def rank_long_candidates(results, overrides=None, sort_key="gap"):
    """
    LONG candidates from the full market pass: first the entry filter
    (entry_ok, already there), now ALSO apply_prefilter (see above) — otherwise
    the intermediate list stays the same ~1500 it was supposed to
    narrow. The rest are sorted by sort_key (default gap — how many
    points are missing to the threshold: the fewer, the more likely the remaining
    expensive stage (news, and STUMPY for non-leaders) pulls the candidate up to
    a real signal). overrides/sort_key come from /refilter — by default (as
    in /autoscan) the defaults and sorting by gap are used, as before.

    Returns (ranked_list, prefilter_fail_counts).
    """
    candidates = [r for r in results if r["direction"] == "LONG" and r["entry_ok"]]
    candidates, fail_counts = apply_prefilter(candidates, overrides)
    candidates = sort_candidates(candidates, sort_key)
    return candidates, fail_counts


# ═══════════════════════════════════════════════════════════════════════════
# RELAXATION LADDER (/find N) — automatically stepping down thresholds until
# N candidates are collected, on ALREADY computed sweep data.
#
# WHAT MOVES HERE AND WHAT DOES NOT. The six prefilter criteria fall into two
# classes, and they must not be mixed:
#   - PREFERENCES (volume_ratio, rr1, dollar_volume, atr_pct) — "which
#     candidates I want". They move.
#   - VALIDITY (price, gap_pct) — "can the calculation be trusted at all".
#     PIVOT_GAP_THRESHOLD_PCT means that with a larger gap the previous session's Pivot
#     is unusable, and the stop and targets are computed from it. Softening it means handing out
#     candidates whose levels are built on garbage. THEY NEVER MOVE,
#     and they are not in any step below.
#
# WHY NO EXPLICIT FLOOR IS NEEDED FOR MOST CRITERIA: the ladder physically cannot
# go below the entry filter of the first step. rank_long_candidates
# takes only entry_ok records, and those already exclude R:R below MIN_ENTRY_RR
# (1.0), volume below 0.4 and ATR outside 0.2-8% — such candidates simply
# do NOT EXIST in the raw data, however much the thresholds are relaxed. An explicit floor is needed only for
# dollar_volume and price: they have no counterpart in check_entry_filters.
#
# Steps must go strictly from strict to soft (each next one is a superset
# of the previous), otherwise the counters stop being monotonic and the step
# number loses its meaning as a measure of market stinginess.
# ═══════════════════════════════════════════════════════════════════════════
FIND_LADDER = [
    {},                                                          # 0 — defaults
    {"rr1": 1.4, "dollar_volume": 750_000},                      # 1
    {"volume_ratio": 0.7, "rr1": 1.3,                            # 2
     "dollar_volume": 500_000, "atr_max": 5.5},
    {"volume_ratio": 0.7, "rr1": 1.2,                            # 3 — floor
     "dollar_volume": 300_000, "atr_min": 0.4, "atr_max": 6.0},
]

# Short labels for the report: internal keys mean nothing in the chat.
_RUNG_LABELS = {"volume_ratio": "volume", "rr1": "rr",
                "dollar_volume": "dv", "atr_min": "atr_min", "atr_max": "atr_max"}


def _money(value):
    """$1M / $750k — compact and without trailing zeros ('%g' instead of '%.2f')."""
    return f"${value/1e6:g}M" if value >= 1e6 else f"${value/1e3:g}k"


def describe_rung(rung):
    """Human-readable "what is relaxed" for a step: 'rr 1.5→1.3, dv $1M→$500k'."""
    parts = []
    for key, value in FIND_LADDER[rung].items():
        was, now = DEFAULT_PREFILTER[key], value
        if key == "dollar_volume":
            was, now = _money(was), _money(now)
        parts.append(f"{_RUNG_LABELS.get(key, key)} {was}→{now}")
    return ", ".join(parts) or "no relaxation"


def find_candidates(results, want, sort_key="gap"):
    """
    Steps down FIND_LADDER to the FIRST step where candidates >= want.

    Returns (candidates, step, counts, fail_counts), where counts is how many
    candidates EACH step passed gave, including step zero. counts[0]
    is mandatory in the report: without it the user sees N names and does not know that
    some of them were added by relaxation, and a command obliged to return N will return
    N even on a day when the market has nothing to offer.

    If even on the last step there are fewer candidates than want — what there is
    is returned (len(counts) == len(FIND_LADDER) signals "not enough").
    Relaxing below the floor is not allowed: the point is measurement, not a guaranteed result.

    Does not touch the network — all fields are computed at the screen_from_batch stage.
    """
    counts, candidates, rung, fail_counts = [], [], 0, {}
    for i, overrides in enumerate(FIND_LADDER):
        candidates, fail_counts = rank_long_candidates(results, overrides, sort_key)
        counts.append(len(candidates))
        rung = i
        if len(candidates) >= want:
            break
    return candidates, rung, counts, fail_counts


def append_find_log(entry):
    """
    Appends a record of the /find run to FIND_LOG_FILE. Not critical for the
    command — if writing fails the search must still answer the user,
    so any I/O error is swallowed with a message in the log.
    """
    from config import FIND_LOG_FILE
    try:
        try:
            with open(FIND_LOG_FILE, encoding="utf-8") as f:
                log = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            log = []
        log.append(entry)
        with open(FIND_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False)
    except OSError as exc:
        print(f"   ⚠️ /find log was not written: {str(exc)[:80]}")


# "Step ∞" — for /autopilot. Removes ALL preferences (volume/RR/ATR/
# turnover) entirely, keeps only VALIDITY (price, gap_pct) — the same
# two fields that do not move at any FIND_LADDER step. This is not yet
# another relaxation but the whole LONG+entry_ok pool: after that, sorting by
# gap and the expensive analysis (news/STUMPY) decide by themselves who is a real signal
# and who is NO_TRADE — the cheap prefilter has nothing more
# to screen out at this stage, only to show the order.
FULL_POOL_OVERRIDES = {"volume_ratio": 0, "rr1": 0, "dollar_volume": 0,
                       "atr_min": 0, "atr_max": 999}


def load_find_log():
    """All /find log records, an empty list if the file does not exist yet."""
    from config import FIND_LOG_FILE
    try:
        with open(FIND_LOG_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def format_find_report(want, candidates, rung, counts, generated_at):
    """
    A single ladder report text — for the /find command AND for the morning task.

    One function, not two formats in two places, ON PURPOSE: the project already
    had exactly this — /scan existed as two diverged copies (in main.py
    with retries, in handlers.py without), and fixing the "obvious" place had no
    effect. The morning report and the manual one must look the same, otherwise they
    cannot be compared by eye from day to day.

    Returns (text, shown_candidates).
    """
    from utils.helpers import age_str

    shown = candidates[:want]
    last = len(FIND_LADDER) - 1
    if len(candidates) >= want:
        head = f"found {len(shown)} at step {rung} of {last}"
    else:
        head = (f"NOT ENOUGH — {len(candidates)} of {want} even at the last "
                f"step {last}")

    lines = [f"🎯 *Candidates requested: {want}* — {head}",
             f"At default thresholds (step 0) there were {counts[0]}."]
    if rung:
        lines.append(f"Relaxed: {describe_rung(rung)}")
    if generated_at:
        lines.append(f"Sweep from {generated_at[:16].replace('T', ' ')} "
                     f"({age_str(generated_at)})")
    lines.append("Candidates by step: "
                 + ", ".join(f"{i}→{c}" for i, c in enumerate(counts)))

    # How many names would NOT pass the default thresholds. Without this line a list of N
    # looks the same on a generous day and an empty one.
    added = len(shown) - min(counts[0], len(shown))
    if added > 0:
        lines.append("")
        lines.append(f"⚠️ {added} of {len(shown)} were added by relaxation — this is not "
                     f"\"found {len(shown)} good ones\" but \"{len(shown)} were collected "
                     f"only at step {rung}\".")
    if len(candidates) < want:
        lines.append("")
        lines.append("I don't go below the last step: the point is to "
                     "measure the market, not to guarantee a result.")
    if shown:
        lines.append("")
        lines.append(f"🎯 *Top {len(shown)}:*\n{format_candidates_table(shown, len(shown))}")
    return "\n".join(lines), shown


# How many top candidates to annotate with earnings dates. Exactly the place
# foreseen by the comment above: earnings proximity is a separate
# network request per ticker; on 1500 names it is the very expensive operation
# the prefilter must exclude, but on a narrow top it is affordable.
# Most names are also in the cache (agents/earnings_calendar.py) and
# come back instantly.
EARNINGS_ANNOTATE_LIMIT = 60


def annotate_earnings(candidates, limit=EARNINGS_ANNOTATE_LIMIT):
    """
    Sets the days_to_earnings field for the top `limit` candidates.
    Missing dates are fetched and stored in the cache — the next call for the same
    names will be instant.

    Returns (candidates, how_many_downloaded) — the number needs to be shown
    to the user: if it is large, the command will not finish instantly.
    """
    from datetime import datetime

    from agents.earnings_calendar import days_to_earnings, fill_cache, load_cache

    top = candidates[:limit]
    tickers = [c["ticker"] for c in top]
    cache = load_cache()
    missing = [t for t in tickers if t not in cache]
    if missing:
        cache = fill_cache(missing, progress_every=25)

    now = datetime.now()
    for c in top:
        c["days_to_earnings"] = days_to_earnings(c["ticker"], now, cache=cache)
    return candidates, len(missing)


def apply_earnings_filter(candidates, keep_within=None, exclude_within=None):
    """
    Filter by earnings proximity. Works ONLY on annotated candidates
    (see annotate_earnings): non-annotated ones have no days_to_earnings.

    keep_within — keep only those with earnings in the next N days
    (the "show me who has a catalyst coming" lens).
    exclude_within — the opposite, remove them (risk logic: I don't want to hold
    a position through a binary event).

    An unknown date is NOT a reason to drop a candidate in
    exclude_within mode: not knowing is not a risk signal, and silently throwing out half
    the list because of data gaps is worse than showing it with a note.
    """
    out = []
    for c in candidates:
        d = c.get("days_to_earnings")
        if keep_within is not None:
            if d is None or not (0 <= d <= keep_within):
                continue
        if exclude_within is not None and d is not None and 0 <= d <= exclude_within:
            continue
        out.append(c)
    return out


# Short /refilter parameter names -> internal DEFAULT_PREFILTER keys.
FILTER_ARG_MAP = {
    "volume": "volume_ratio",
    "rr": "rr1",
    "price": "price",
    "atr_min": "atr_min",
    "atr_max": "atr_max",
    "dv": "dollar_volume",
    "gap_pct": "gap_pct",
}


def parse_refilter_args(args):
    """
    Parses /refilter command arguments like "volume=1.5 rr=2
    sort=dollar_volume" into (overrides, sort_key, unknown, earnings). overrides —
    a dict for apply_prefilter (internal keys, see FILTER_ARG_MAP). unknown —
    a list of tokens that could not be recognised (a typo in a
    parameter name, not a number, an unknown sort field) — they must be shown
    to the user rather than silently ignored.

    earnings — {"keep": N} and/or {"exclude": N} from the earnings= and
    no_earnings= keys. Kept SEPARATE from overrides, because it is not a
    prefilter threshold: the earnings date is not among the sweep fields, it is set
    in a separate step on the narrow top (see annotate_earnings).
    """
    overrides = {}
    sort_key = "gap"
    unknown = []
    earnings = {}

    for arg in args:
        if "=" not in arg:
            unknown.append(arg)
            continue
        key, _, value = arg.partition("=")
        key = key.strip().lower()
        value = value.strip()

        if key == "sort":
            if value in SORT_FIELDS:
                sort_key = value
            else:
                unknown.append(arg)
            continue

        if key in ("earnings", "no_earnings"):
            try:
                days = int(value)
            except ValueError:
                unknown.append(arg)
                continue
            if days < 0:
                unknown.append(arg)
                continue
            earnings["keep" if key == "earnings" else "exclude"] = days
            continue

        field = FILTER_ARG_MAP.get(key)
        if field is None:
            unknown.append(arg)
            continue
        try:
            overrides[field] = float(value)
        except ValueError:
            unknown.append(arg)

    return overrides, sort_key, unknown, earnings


def format_candidates_table(candidates, limit_show=10):
    """
    A detailed candidate table (for /autoscan and /refilter) — not just
    a list of tickers but the metrics they can be filtered/sorted by
    in /refilter. Without it, with sort=volume_ratio/dollar_volume/atr_pct
    the tickers alone did not show WHY they ended up on top.
    """
    if not candidates:
        return ""
    # The earnings column appears only if candidates were annotated
    # (annotate_earnings) — otherwise it would be a column of dashes.
    show_earnings = any("days_to_earnings" in c for c in candidates[:limit_show])
    head = "| Ticker | Price | Score | Gap | Volume | Turnover($) | ATR% | R:R |"
    sep = "|-------|------|------|------|-------|-----------|------|-----|"
    if show_earnings:
        head += " Earnings |"
        sep += "-------|"
    lines = [head, sep]
    for c in candidates[:limit_show]:
        dv = c.get("dollar_volume", 0)
        dv_fmt = f"{dv/1_000_000:.1f}M" if dv >= 1_000_000 else f"{dv/1000:.0f}K"
        row = (
            f"| {escape_markdown(c.get('ticker', '?'))} | ${safe_format_float(c.get('price'))} | "
            f"{c.get('det', 0):+d} | {c.get('gap', 0)} | {c.get('volume_ratio', 0):.1f}x | "
            f"{dv_fmt} | {c.get('atr_pct', 0):.1f}% | 1:{safe_format_float(c.get('rr1', 0), 1)} |"
        )
        if show_earnings:
            d = c.get("days_to_earnings")
            # Mark ⛔ for those the entry filter will not let through anyway:
            # otherwise the candidate looks workable, while /queue silently returns
            # NO_TRADE for it, and the reason is only visible in the ticker report.
            if d is None:
                cell = "—"
            elif d <= EARNINGS_BLOCK_DAYS:
                cell = f"⛔{d}d"
            else:
                cell = f"{d}d"
            row += f" {cell} |"
        lines.append(row)
    return "\n".join(lines)


def save_raw_sweep_results(results):
    """
    Saves the WHOLE (unfiltered) result of the last full_market_sweep —
    all processed tickers, any direction, BEFORE entry_ok and BEFORE
    apply_prefilter. Needed so the filter/ranking can be recomputed
    (e.g. after editing apply_prefilter thresholds — see the abs(score)
    case that removed score because of a 96% cut on live
    data) WITHOUT another 40-50-minute pass over the market — see /refilter.
    Overwritten entirely on every new /autoscan.
    """
    from config import AUTOSCAN_RAW_RESULTS_FILE, SWEEP_ARCHIVE_DIR
    now = datetime.now()
    payload = {
        "generated_at": now.isoformat(),
        "count": len(results),
        "results": results,
    }
    with open(AUTOSCAN_RAW_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    # Plus a copy in the archive. The main file is overwritten by every /autoscan —
    # without an archive the sweep history does not accumulate, and there is nothing to measure the filter on
    # (agents/filter_lab.py). The copy is cheap: ~2 MB per sweep, and there are
    # realistically a few sweeps a week.
    try:
        os.makedirs(SWEEP_ARCHIVE_DIR, exist_ok=True)
        archive = os.path.join(SWEEP_ARCHIVE_DIR,
                               f"sweep_{now:%Y-%m-%d_%H%M}.json")
        with open(archive, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        print(f"   🗄 Sweep archived: {archive}")
    except OSError as exc:
        # The archive is for future analytics, not for /refilter to work. If it
        # was not written (no space, permissions) — that is no reason to crash /autoscan.
        print(f"   ⚠️ Could not archive the sweep: {str(exc)[:120]}")


def load_raw_sweep_results():
    """Returns (results, generated_at) from the last saved /autoscan sweep."""
    from config import AUTOSCAN_RAW_RESULTS_FILE
    try:
        with open(AUTOSCAN_RAW_RESULTS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("results", []), data.get("generated_at")
    except (FileNotFoundError, json.JSONDecodeError):
        return [], None


def save_autoscan_candidates(candidates, queue_start=0):
    """
    Saves the FULL list of LONG candidates to a separate file — reports/ is an already
    mounted directory (see Dockerfile), no new volume mounts
    needed. Overwritten entirely on every /autoscan or /refilter:
    the old list is considered stale as soon as a new calculation appears.

    queue_start — the position to start the queue cursor from
    (_autoscan_queue_cursor in state.json, see next_autoscan_queue_batch).
    It used to make sense to start at 10 for /autoscan (it immediately scanned
    the top 10 itself) — now /autoscan scans nothing automatically (see
    cmd_autoscan), so both there and in /refilter the start is always 0.
    """
    from config import AUTOSCAN_CANDIDATES_FILE
    generated_at = datetime.now().isoformat()
    payload = {"generated_at": generated_at, "count": len(candidates), "candidates": candidates}
    with open(AUTOSCAN_CANDIDATES_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    state = load_state()
    state["_autoscan_queue_cursor"] = min(queue_start, len(candidates))
    state["_autoscan_queue_generated_at"] = generated_at
    save_state(state)


def load_autoscan_candidates():
    """Returns (candidate_list, generated_at) from the last saved /autoscan."""
    from config import AUTOSCAN_CANDIDATES_FILE
    try:
        with open(AUTOSCAN_CANDIDATES_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("candidates", []), data.get("generated_at")
    except (FileNotFoundError, json.JSONDecodeError):
        return [], None


def next_autoscan_queue_batch(count=10):
    """
    A cursor over the saved list of LONG candidates (see save_autoscan_candidates)
    — for /queue: continues from where the previous call stopped,
    wraps around to the start when the list is exhausted. A separate cursor from /mass
    (_autoscan_queue_cursor, not _mass_cursor) — the commands don't touch each other.

    Returns (tickers_for_this_batch, cursor_before, cursor_after, list_size, generated_at).
    """
    candidates, generated_at = load_autoscan_candidates()
    total = len(candidates)
    if total == 0:
        return [], 0, 0, 0, generated_at

    state = load_state()
    cursor = state.get("_autoscan_queue_cursor", 0) % total
    count = min(count, total)

    end = cursor + count
    batch = candidates[cursor:end] if end <= total else candidates[cursor:] + candidates[:end - total]
    new_cursor = end % total

    state["_autoscan_queue_cursor"] = new_cursor
    save_state(state)
    return batch, cursor, new_cursor, total, generated_at


def run_screen(source="trending", limit=15, status_callback=None):
    """Runs the candidates and returns a sorted list."""
    tickers = get_candidate_tickers(source, limit)
    if not tickers:
        return [], []

    print(f"\n🔎 SCREENING ({SOURCES.get(source, source)}): {len(tickers)} tickers")
    print(f"   {', '.join(tickers)}\n")

    results = []
    for i, t in enumerate(tickers, 1):
        if status_callback:
            status_callback(f"{i}/{len(tickers)} — {t}")
        try:
            r = screen_ticker(t)
            if r:
                results.append(r)
                print(f"   [{i}/{len(tickers)}] {t:6} score {r['det']:+d} | "
                      f"volume {r['volume_ratio']:.2f}x | to signal {r['gap']}")
        except Exception as e:
            print(f"   [{i}/{len(tickers)}] {t:6} error: {str(e)[:80]}")

    # Those with the least remaining first; on a tie — the one with higher volume
    results.sort(key=lambda x: (not x["entry_ok"], x["gap"], -x["volume_ratio"]))
    return results, tickers


def format_screen_report(results, source, limit_show=8):
    """Report for Telegram."""
    if not results:
        return "❌ Could not get data for any ticker"

    lines = [f"🔎 *PRELIMINARY SCREEN* ({escape_markdown(SOURCES.get(source, source))})\n"]
    lines.append(f"Tickers checked: {len(results)}")
    lines.append("_Computed factors only — news not included,_")
    lines.append("_the full analysis will add it._\n")

    lines.append("| Ticker | Score | Gap | Volume | ATR | Entry |")
    lines.append("|-------|------|----------|-------|-----|------|")
    for r in results[:limit_show]:
        gate = "✅" if r["entry_ok"] else "⛔"
        lines.append(
            f"| {escape_markdown(r['ticker'])} | {r['det']:+d} | {r['gap']} | "
            f"{r['volume_ratio']:.1f}x | {r['atr_pct']:.1f}% | {gate} |"
        )
    lines.append("")

    # Take those whose gap to the threshold is no more than news and flow can give
    ready = [r for r in results if r["entry_ok"] and r["gap"] <= LLM_FACTORS_MAX - 1]
    if ready:
        lines.append("🎯 *Worth running through the full analysis:*")
        for r in ready[:3]:
            side = "LONG" if r["direction"] == "LONG" else "SHORT"
            lines.append(
                f"• *{escape_markdown(r['ticker'])}* (${safe_format_float(r['price'])}) — "
                f"leaning {side}, {r['gap']} points short; "
                f"volume {r['volume_ratio']:.1f}x, R:R 1:{safe_format_float(r['rr1'], 1)}"
            )
            lines.append(
                f"    _tech. {r['technical']:+d}, STUMPY {r['stumpy']:+d}, "
                f"VWAP {r['vwap']:+d}, RSI {safe_format_float(r['rsi'], 1)}_"
            )
        lines.append("")
        cmds = " ".join(r["ticker"] for r in ready[:3])
        lines.append(f"Run: `/scan {cmds}`")
    else:
        lines.append(
            "😐 No candidate qualifies: either the gap to the threshold is more than "
            f"{LLM_FACTORS_MAX - 1} points, "
            "or it fails the entry filter."
        )

    blocked = [r for r in results[:limit_show] if not r["entry_ok"]]
    if blocked:
        lines.append("\n⛔ *Will not pass the entry filter:*")
        for r in blocked[:3]:
            lines.append(f"• {escape_markdown(r['ticker'])} — {escape_markdown(r['entry_why'])}")

    return "\n".join(lines)


def format_batch_line(idx, total, batch_res, market_pos=None, market_total=None):
    """A short summary of one batch — sent right away, without waiting for the pass to finish."""
    progress = f" (market: {market_pos}/{market_total})" if market_total else ""
    if not batch_res:
        return f"⏳ Batch {idx}/{total}{progress}: no suitable tickers"
    passed = [r for r in batch_res if r["entry_ok"]]
    lines = [f"⏳ *Batch {idx}/{total}*{progress} — checked {len(batch_res)}, passed the filter {len(passed)}"]
    for r in passed[:4]:
        lines.append(
            f"  • *{escape_markdown(r['ticker'])}* ${safe_format_float(r['price'])} "
            f"score {r['det']:+d} (to signal {r['gap']}), volume {r['volume_ratio']:.1f}x, "
            f"R:R 1:{safe_format_float(r['rr1'], 1)}"
        )
    if not passed:
        lines.append("  _all screened out by the entry filter_")
    return "\n".join(lines)


def format_mass_report(results, market_range=None, limit_show=12):
    """Final report of the mass pass."""
    if not results:
        return "❌ Could not get data for any ticker"

    passed = [r for r in results if r["entry_ok"]]
    lines = ["🏁 *MASS PASS RESULT*\n"]
    lines.append(f"Stocks checked: {len(results)} | passed the entry filter: {len(passed)}")
    if market_range and market_range.get("universe_total"):
        lines.append(
            f"🌐 Position in the full market list: {market_range['cursor_start']+1}–"
            f"{market_range['cursor_end']} of {market_range['universe_total']} — "
            f"the next /mass will continue from here"
        )
    lines.append("_News and STUMPY (for non-leaders) not included — the full analysis will add them._")
    lines.append("_STUMPY is computed for leaders only._\n")

    lines.append("| Ticker | Score | Gap | Volume | ATR | R:R |")
    lines.append("|-------|------|------|-------|-----|-----|")
    for r in passed[:limit_show]:
        lines.append(
            f"| {escape_markdown(r['ticker'])} | {r['det']:+d} | {r['gap']} | "
            f"{r['volume_ratio']:.1f}x | {r['atr_pct']:.1f}% | 1:{safe_format_float(r['rr1'], 1)} |"
        )
    lines.append("")

    ready = [r for r in passed if r["gap"] <= LLM_FACTORS_MAX - 1]
    if ready:
        lines.append("🎯 *Candidates for full analysis:*")
        for r in ready[:5]:
            lines.append(
                f"• *{escape_markdown(r['ticker'])}* (${safe_format_float(r['price'])}) — "
                f"{r['direction']}, {r['gap']} short; tech. {r['technical']:+d}, "
                f"STUMPY {r['stumpy']:+d}, VWAP {r['vwap']:+d}, RSI {safe_format_float(r['rsi'], 1)}"
            )
        lines.append("")
        lines.append(f"Run: `/scan {' '.join(r['ticker'] for r in ready[:5])}`")
    else:
        lines.append("😐 Nobody is close to the threshold — nothing to enter today.")
    return "\n".join(lines)
