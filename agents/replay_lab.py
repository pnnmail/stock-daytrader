"""
Replay of sweeps on historical data: reconstructs what /autoscan would have returned
on each past day, so filter thresholds can be measured WITHOUT waiting
for live sweeps to accumulate.

    python -m agents.replay_lab download        # phase 1: downloads history (hours)
    python -m agents.replay_lab replay           # phase 2: computes sweeps (hours)
    python -m agents.replay_lab replay --force   # recompute ALL dates from scratch
    python -m agents.replay_lab status           # what is already done
    python -m agents.replay_lab dates            # which dates will be computed

Both phases are RESUMABLE: an interrupted run continues from where it stopped.
This is not a convenience but a necessity — Yahoo regularly returns YFRateLimitError
(verified: triggers on a sample of 10 tickers), and over a multi-hour download
it will happen more than once.

WHY THE REPLAY IS EXACT. screen_from_batch(ticker, hourly, daily) is a PURE
function of two dataframes. So nothing is rewritten here: the history
is cut at date D and passed INTO THE SAME function. The result matches
what the sweep would have returned that day, by construction rather than by effort.

The live sweep takes hourly period="5d" (with prepost) and daily period="3mo" —
slices are made for the same windows.

WHAT THE REPLAY DOES NOT FIX (keep this in mind when reading the result):

1. UNIVERSE SURVIVORSHIP. The ticker list is today's from nasdaqtrader.com.
   Stocks delisted during the year are not in it. Upward bias:
   it is mostly losers that leave.
2. SPLITS. We download auto_adjust=False (like the live sweep), but a stock with a split
   inside the window still produces a break in the series.
3. DEPENDENCE OF NEIGHBOURING DATES. Dates are taken every STEP trading days precisely
   for this reason: consecutive days share both the candidate set and the market regime.
   The step reduces overlap but does not eliminate it.
4. THE REPLAY IS IDEALISED. The live /autoscan runs at a specific moment,
   sometimes mid-session, with a partial last bar. The replay always sees
   completed sessions. This makes it slightly MORE OPTIMISTIC than reality.

The main check of the result is not the overall edge but the split by market
direction (done by filter_lab): if a filter wins only on up days,
that is leverage to the market, not selection.
"""
import json
import os
import pickle
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from agents.screener import _get_universe, screen_from_batch
from config import (REPLAY_CACHE_DIR, SWEEP_ARCHIVE_DIR,
                    EXTENDED_HOURS_ENABLED)

BATCH = 25
MONTHS_BACK = 12
STEP = 5                 # take every N-th trading day
HOURLY_SESSIONS = 5      # live sweep window: period="5d"
DAILY_ROWS = 63          # live sweep window: period="3mo" (~63 trading days)
MIN_HOURLY_BARS = 20     # below this, screen_from_batch itself returns None

# Rate-limit backoff. Measured: a 20-second pause lifts the block, but over
# a multi-hour download we will hit it more than once — a growing pause is more reliable than a fixed one.
BACKOFF = [15, 30, 60, 120, 300]
PAUSE_BETWEEN_BATCHES = 2.0


def _batch_path(idx):
    return os.path.join(REPLAY_CACHE_DIR, f"batch_{idx:04d}.pkl")


def _download_with_retry(tickers, **kwargs):
    for attempt, wait in enumerate([0] + BACKOFF):
        if wait:
            print(f"      rate limit, waiting {wait}s...", flush=True)
            time.sleep(wait)
        try:
            df = yf.download(tickers, progress=False, threads=False,
                             group_by="ticker", auto_adjust=False, **kwargs)
            if df is not None and not df.empty:
                return df
        except Exception as exc:
            if "rate" not in str(exc).lower() and "too many" not in str(exc).lower():
                print(f"      error: {str(exc)[:100]}", flush=True)
                return None
    return None


def _split_by_ticker(raw, tickers):
    out = {}
    if raw is None:
        return out
    for t in tickers:
        try:
            df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            df = df.dropna(how="all")
            if not df.empty:
                out[t] = df
        except (KeyError, TypeError):
            continue
    return out


def phase_download():
    os.makedirs(REPLAY_CACHE_DIR, exist_ok=True)
    universe = _get_universe()
    print(f"Universe: {len(universe)} tickers, in batches of {BATCH}")
    batches = [universe[i:i + BATCH] for i in range(0, len(universe), BATCH)]
    print(f"Batches: {len(batches)}. Already downloaded: "
          f"{sum(1 for i in range(len(batches)) if os.path.exists(_batch_path(i)))}")

    # Hourly for 12 months + margin for the window; daily for 15 months (12 + 3 for the slice).
    start_h = (datetime.now() - timedelta(days=MONTHS_BACK * 31 + 10))
    start_d = (datetime.now() - timedelta(days=(MONTHS_BACK + 3) * 31))

    done = skipped = failed = 0
    for i, chunk in enumerate(batches):
        path = _batch_path(i)
        if os.path.exists(path):
            skipped += 1
            continue
        hourly = _split_by_ticker(
            _download_with_retry(chunk, start=start_h.strftime("%Y-%m-%d"),
                                 interval="1h", prepost=EXTENDED_HOURS_ENABLED),
            chunk)
        time.sleep(PAUSE_BETWEEN_BATCHES)
        daily = _split_by_ticker(
            _download_with_retry(chunk, start=start_d.strftime("%Y-%m-%d"),
                                 interval="1d"),
            chunk)
        payload = {t: {"h": hourly.get(t), "d": daily.get(t)}
                   for t in chunk if t in hourly and t in daily}
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=4)
        done += 1
        if not payload:
            failed += 1
        if done % 10 == 0 or i == len(batches) - 1:
            pct = (i + 1) / len(batches) * 100
            print(f"  [{i+1}/{len(batches)}] {pct:.1f}% — downloaded {done}, "
                  f"skipped {skipped}, empty {failed}", flush=True)
        time.sleep(PAUSE_BETWEEN_BATCHES)
    print(f"\nDone. New batches: {done}, existing: {skipped}, empty: {failed}")


# The end of the date-sampling window is FIXED, not "today". A bug found on
# 2026-08-05: previously _sample_dates() took period=f"{MONTHS_BACK}mo" from
# datetime.now() — re-running a day or two after the original gave
# a DIFFERENT date grid (each shifted by a few days), and `replay --force`
# for adding new fields (ret_5d/max_ret_20d) silently produced a SECOND,
# non-overlapping set of synthetic_*.json instead of recomputing the old one —
# 76 files in reports/sweeps/ instead of 38, half without the new fields. The date
# below is the end of the window the history was downloaded for (agents.replay_lab
# download), not "today": this way the grid is stable regardless of which
# day replay is run.
SAMPLE_WINDOW_END = "2026-08-03"


def _sample_dates():
    """
    The trading-day calendar is taken from SPY — it trades whenever the market
    is open and needs no separate holiday logic. The window is FIXED
    (see SAMPLE_WINDOW_END) — otherwise repeated runs on different days give
    synthetic sweeps that are not comparable with each other.
    """
    end = pd.Timestamp(SAMPLE_WINDOW_END)
    start = end - pd.DateOffset(months=MONTHS_BACK) - pd.Timedelta(days=10)
    # Via _download_with_retry, not directly: the rate limit triggers even on
    # a single request (verified), and without the calendar the whole phase does not start.
    spy = _download_with_retry(["SPY"], start=start.strftime("%Y-%m-%d"),
                               end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                               interval="1d")
    if spy is None or spy.empty:
        print("   ⚠️ SPY did not download even with retries — try later.")
        return []
    days = sorted({d.date() for d in spy.index})
    # The first DAILY_ROWS days are unusable: they do not have a full history window yet.
    usable = days[DAILY_ROWS:]
    return usable[::STEP]


def compute_research_signals(d_slice):
    """
    Two extra signals on top of the usual screen_from_batch fields —
    added on 2026-08-05 to test the literature on short-term reversal and the
    MAX effect (Lehmann 1990, Bali/Cakici/Whitelaw 2011). Computed
    from THE SAME precisely dated d_slice (<= D) as everything else in the
    replay — no look-ahead.

    ret_5d      — return over the last 5 trading days. The short-term reversal
                  literature predicts an INVERSE relation with future
                  returns: the higher ret_5d, the worse the expected outcome.
    max_ret_20d — maximum DAILY return over the last ~20 trading
                  days (the "jackpot" day). The MAX effect predicts the same
                  inverse relation: high max_ret_20d -> low future
                  returns (lottery stocks are systematically overpriced).

    Deliberately not written into the production filter (agents/screener.py) — these are
    RESEARCH fields for filter_lab, the live /autoscan does not compute them.
    """
    closes = d_slice["Close"].dropna()
    if len(closes) < 6:
        return {"ret_5d": None, "max_ret_20d": None}
    ret_5d = float(closes.iloc[-1] / closes.iloc[-6] - 1)
    daily_rets = closes.pct_change().dropna().tail(20)
    max_ret_20d = float(daily_rets.max()) if len(daily_rets) >= 5 else None
    return {"ret_5d": ret_5d, "max_ret_20d": max_ret_20d}


def phase_replay(force=False):
    dates = _sample_dates()
    if not dates:
        print("Could not get the trading calendar (SPY).")
        return
    os.makedirs(SWEEP_ARCHIVE_DIR, exist_ok=True)
    if force:
        # A recompute is needed when new fields are added to screen_from_batch/compute_research_signals
        # — old synthetic_*.json files don't contain them, and
        # "todo = only missing ones" would silently skip already computed
        # dates and leave them without the new fields.
        todo = list(dates)
        print("--force: recomputing ALL dates (will overwrite existing ones)")
    else:
        todo = [d for d in dates
                if not os.path.exists(os.path.join(
                    SWEEP_ARCHIVE_DIR, f"synthetic_{d:%Y-%m-%d}.json"))]
    print(f"Dates to compute: {len(dates)} (step {STEP} trading days), "
          f"remaining: {len(todo)}")
    if not todo:
        print("All dates already computed. Use --force to recompute "
              "(e.g. after adding new fields).")
        return

    files = sorted(f for f in os.listdir(REPLAY_CACHE_DIR)
                   if f.startswith("batch_"))
    if not files:
        print(f"Cache is empty — first run: python -m agents.replay_lab download")
        return
    print(f"Batches in cache: {len(files)}")

    # {date: [results]} — accumulate across all tickers, write at the end.
    per_date = {d: [] for d in todo}
    t0 = time.time()
    for n, fname in enumerate(files, 1):
        try:
            with open(os.path.join(REPLAY_CACHE_DIR, fname), "rb") as f:
                payload = pickle.load(f)
        except Exception as exc:
            print(f"  ⚠️ {fname}: {str(exc)[:80]}")
            continue

        for ticker, frames in payload.items():
            h_all, d_all = frames.get("h"), frames.get("d")
            if h_all is None or d_all is None or h_all.empty or d_all.empty:
                continue
            h_dates = pd.Series([ts.date() for ts in h_all.index],
                                index=h_all.index)
            for D in todo:
                # Slice exactly for the live sweep windows: 5 trading sessions
                # of hourly and 63 rows of daily, all strictly <= D.
                d_slice = d_all[d_all.index.date <= D].tail(DAILY_ROWS)
                if len(d_slice) < 2:
                    continue
                sessions = sorted({x for x in h_dates.values if x <= D})
                if not sessions:
                    continue
                keep = set(sessions[-HOURLY_SESSIONS:])
                h_slice = h_all[h_dates.isin(keep).values]
                if len(h_slice) < MIN_HOURLY_BARS:
                    continue
                try:
                    row = screen_from_batch(ticker, h_slice, d_slice)
                except Exception:
                    row = None
                if row:
                    row.update(compute_research_signals(d_slice))
                    per_date[D].append(row)

        if n % 20 == 0 or n == len(files):
            el = time.time() - t0
            eta = el / n * (len(files) - n) / 60
            print(f"  [{n}/{len(files)}] batches, {el/60:.1f} min, "
                  f"~{eta:.0f} min left", flush=True)

    for D, rows in per_date.items():
        if not rows:
            continue
        path = os.path.join(SWEEP_ARCHIVE_DIR, f"synthetic_{D:%Y-%m-%d}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "generated_at": datetime.combine(
                    D, datetime.min.time()).replace(hour=23).isoformat(),
                "count": len(rows),
                "synthetic": True,   # so filter_lab can tell them apart from live sweeps
                "results": rows,
            }, f, ensure_ascii=False)
        print(f"  {D}: {len(rows)} tickers -> {os.path.basename(path)}")
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")


def phase_status():
    batches = len([f for f in os.listdir(REPLAY_CACHE_DIR)
                   if f.startswith("batch_")]) if os.path.isdir(REPLAY_CACHE_DIR) else 0
    synth = len([f for f in os.listdir(SWEEP_ARCHIVE_DIR)
                 if f.startswith("synthetic_")]) if os.path.isdir(SWEEP_ARCHIVE_DIR) else 0
    live = len([f for f in os.listdir(SWEEP_ARCHIVE_DIR)
                if f.startswith("sweep_")]) if os.path.isdir(SWEEP_ARCHIVE_DIR) else 0
    size = 0
    if os.path.isdir(REPLAY_CACHE_DIR):
        size = sum(os.path.getsize(os.path.join(REPLAY_CACHE_DIR, f))
                   for f in os.listdir(REPLAY_CACHE_DIR)) / 1024 / 1024
    print(f"Batches in cache: {batches} ({size:.0f} MB)")
    print(f"Synthetic sweeps: {synth}")
    print(f"Live sweeps: {live}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "download":
        phase_download()
    elif cmd == "replay":
        phase_replay(force="--force" in sys.argv)
    elif cmd == "dates":
        for d in _sample_dates():
            print(" ", d)
    else:
        phase_status()
    return 0


if __name__ == "__main__":
    sys.exit(main())
