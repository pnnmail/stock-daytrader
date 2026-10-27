"""
Earnings calendar: earnings dates per ticker, with a disk cache.

Needed in three places:
  - `agents/filter_lab.py` — testing the hypothesis "candidates with earnings around the corner
    behave differently from the rest" on archived sweeps (research);
  - `/earnings` in the Telegram bot — a direct filter "which of ~7000 stocks
    report in N days", no scoring, for manual analysis
    (telegram_bot/handlers.py::cmd_earnings). This is NOT a hypothesis, just
    a working tool — 2026-08-05, at the user's direct request;
  - a live candidate-source filter for signals, IF further testing
    confirms it (not confirmed yet, not used as a signal).

    python -m agents.earnings_calendar fill        # cache for archived sweeps (~1800 tickers, for filter_lab)
    python -m agents.earnings_calendar fill-full    # cache for ALL ~7000 (for /earnings), takes hours
    python -m agents.earnings_calendar status       # how much is there already, for both sets

WHY A DISK CACHE AND RESUMABILITY. The top 60 candidates of 41 archived sweeps
give 1792 unique tickers, at ~1.3 s per request — about 40 minutes in
the ideal case. On long runs Yahoo consistently returns
YFRateLimitError (measured in `replay_lab`: ~20 s pauses lift the block),
so the same trick is used here — backoff and full resumability: tickers already collected
are skipped, the run can be interrupted and continued.

AN HONEST CAVEAT ABOUT LOOK-AHEAD. `days_to_earnings(t, D)` takes
the first report date STRICTLY AFTER D, i.e. it uses today's knowledge of
when the report actually happened. In reality a company confirms the exact date
usually 2-4 weeks ahead, so for short windows (our case — "report within
the next 7-14 days") the date was almost always known in advance and
the bias is negligible. For windows of a couple of months that is no longer true, and such
thresholds are deliberately not tested here.
"""
import json
import os
import sys
import time

from config import EARNINGS_CALENDAR_CACHE

# Same backoff as in replay_lab: the rate limit will trigger more than once on a long
# run, and we must not crash on it — the run is long.
BACKOFF = [15, 30, 60, 120, 300]


def load_cache():
    try:
        with open(EARNINGS_CALENDAR_CACHE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_cache(cache):
    try:
        os.makedirs(os.path.dirname(EARNINGS_CALENDAR_CACHE), exist_ok=True)
        with open(EARNINGS_CALENDAR_CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except OSError as exc:
        print(f"   ⚠️ Earnings date cache was not written: {str(exc)[:80]}")


def _fetch_one(ticker):
    """
    {date YYYY-MM-DD: surprise in % or None} for a ticker, {} if there is no data.
    An empty dict is cached too — otherwise tickers without earnings (ETFs, funds)
    would be re-downloaded on every run.

    The surprise (Reported EPS vs EPS Estimate) is stored together with the date
    on purpose: it is available retrospectively and lets us test not only
    "earnings soon" but also PEAD — the drift AFTER a report with a positive surprise.
    A separate pass would cost another 45 minutes for the same 1792 tickers.
    """
    import math

    import yfinance as yf

    for attempt, pause in enumerate([0] + BACKOFF):
        if pause:
            time.sleep(pause)
        try:
            df = yf.Ticker(ticker).earnings_dates
            if df is None or df.empty:
                return {}
            out = {}
            for idx, row in df.iterrows():
                surprise = row.get("Surprise(%)")
                try:
                    surprise = (None if surprise is None or math.isnan(float(surprise))
                                else round(float(surprise), 2))
                except (TypeError, ValueError):
                    surprise = None
                out[idx.date().isoformat()] = surprise
            return out
        except Exception as exc:
            msg = str(exc)
            if "RateLimit" not in msg and "Too Many Requests" not in msg:
                return {}
            if attempt == len(BACKOFF):
                print(f"   ⚠️ {ticker}: rate limit did not lift, skipping")
                return {}
    return {}


def fill_cache(tickers, save_every=25, progress_every=100):
    """Fills the cache. Tickers already collected are skipped (resumability)."""
    cache = load_cache()
    need = [t for t in tickers if t not in cache]
    print(f"   In cache {len(cache)}, still need {len(need)} of {len(tickers)}")

    for i, ticker in enumerate(need, 1):
        cache[ticker] = _fetch_one(ticker)
        if i % save_every == 0:
            save_cache(cache)
        if i % progress_every == 0:
            print(f"   {i}/{len(need)}...", flush=True)

    save_cache(cache)
    return cache


def _as_date(value):
    from datetime import date
    y, m, d = map(int, value.split("-"))
    return date(y, m, d)


def days_to_earnings(ticker, when, cache=None):
    """
    How many days from `when` to the nearest report AFTER that date.
    None if the ticker is not in the cache or no future dates were found for it.
    """
    cache = load_cache() if cache is None else cache
    dates = cache.get(ticker)
    if not dates:
        return None
    base = when.date() if hasattr(when, "date") else when
    future = sorted(d for d in dates if d > base.isoformat())
    if not future:
        return None
    return (_as_date(future[0]) - base).days


def last_earnings(ticker, when, cache=None):
    """
    (how many days ago the last report was, surprise in %) as of `when`.
    (None, None) if there were no reports before that date or the ticker is not in the cache.

    Needed for PEAD: the hypothesis looks BACK (the report is out, the surprise is known),
    unlike days_to_earnings, which looks forward.
    """
    cache = load_cache() if cache is None else cache
    dates = cache.get(ticker)
    if not dates:
        return None, None
    base = when.date() if hasattr(when, "date") else when
    past = sorted(d for d in dates if d <= base.isoformat())
    if not past:
        return None, None
    latest = past[-1]
    return (base - _as_date(latest)).days, dates.get(latest)


def _sweep_universe():
    """Tickers from the top N of all archived sweeps — exactly what filter_lab measures."""
    from agents.filter_lab import list_sweeps, TOP_N
    from agents.screener import rank_long_candidates

    uni = set()
    for _, _, results in list_sweeps():
        ranked, _ = rank_long_candidates(results)
        uni.update(c["ticker"] for c in ranked[:TOP_N])
    return sorted(uni)


def full_universe():
    """
    The WHOLE market (~7000, the same list /autoscan downloads), not only
    tickers from archived sweeps. Needed for /earnings — the days-to-earnings
    filter must look at all stocks, not only those that once
    passed LONG scoring.
    """
    from agents.screener import _get_universe
    return _get_universe()


def scan_by_days(min_days, max_days, cache=None, today=None):
    """
    Tickers from the FULL universe with a report in the window [min_days, max_days] from
    `today` (default — now). Returns a list of
    (ticker, days_to_earnings, earnings_date), sorted by ascending
    days. Tickers without a date in the cache or without a report in the window do not make it into
    the list, but their count is tracked separately (see cmd_earnings) — this is
    A COVERAGE GAP, not a negative result, and the two must not be
    confused: a ticker without data could fall into the window, we just don't know.
    """
    from datetime import datetime

    today = today or datetime.now()
    cache = load_cache() if cache is None else cache
    universe = full_universe()

    covered = [t for t in universe if t in cache]
    out = []
    for ticker in covered:
        d = days_to_earnings(ticker, today, cache=cache)
        if d is not None and min_days <= d <= max_days:
            dates = sorted(x for x in cache[ticker] if x > today.date().isoformat())
            out.append((ticker, d, dates[0] if dates else None))
    out.sort(key=lambda x: x[1])
    return out, len(universe), len(covered)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "fill":
        universe = _sweep_universe()
        print(f"Unique tickers in archived sweeps: {len(universe)}")
        fill_cache(universe)
        print("Done.")
    elif cmd == "fill-full":
        universe = full_universe()
        print(f"Full universe: {len(universe)} tickers. This takes hours "
              f"(~1.3 s per ticker minimum, plus the rate limit) — resumable, "
              f"can be interrupted.")
        fill_cache(universe)
        print("Done.")
    else:
        cache = load_cache()
        with_dates = sum(1 for v in cache.values() if v)
        print(f"Tickers in cache: {len(cache)} (with dates: {with_dates})")
        try:
            sweep_uni = _sweep_universe()
            missing_sweep = [t for t in sweep_uni if t not in cache]
            print(f"For filter_lab (archived sweeps): need {len(sweep_uni)}, "
                  f"missing {len(missing_sweep)}")
        except Exception as exc:
            print(f"Could not compute the archived-sweep set: {str(exc)[:80]}")
        try:
            full_uni = full_universe()
            missing_full = [t for t in full_uni if t not in cache]
            print(f"For /earnings (whole market): need {len(full_uni)}, "
                  f"missing {len(missing_full)}"
                  + (" — run: python -m agents.earnings_calendar fill-full"
                     if missing_full else ""))
        except Exception as exc:
            print(f"Could not compute the full universe: {str(exc)[:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
