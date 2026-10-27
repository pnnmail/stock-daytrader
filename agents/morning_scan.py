"""
Autonomous morning loop: once per trading day, before the open, build a list of
candidates and send it to Telegram. Not a single model call.

WHAT IT IS AND IS NOT. The job is "where to start today": N tickers from
which a human continues the analysis by hand. It is NOT a prediction and not
a claim of an edge: the list is ranked by the same gap ordering as /find,
which only estimates who is closest to the entry threshold. That is exactly why
there is no LLM here and no choosing thresholds "by market conditions" — switching sets
by regime was already tested on 37 archived sweeps and lost to a coin
flip, so there is nothing to automate.

WHY NOT AT 9:00 AS REQUESTED. full_market_sweep takes 40-50 minutes. Starting at
9:00 ET means ready by 9:40-9:50 with the market opening at 9:30 — i.e.
the fallback "didn't make it, use yesterday's list" would fire almost every
day and would be the main path, not an emergency one. The 7:45 ET default leaves a margin.

THE FALLBACK IS LOUD. If the sweep failed or returned nothing, raw data from
the previous run is used, but the stale flag goes into the message — silently sending a list
computed on yesterday's prices is worse than sending none.
"""
import json
import os
from datetime import datetime

from config import MORNING_CONFIG_FILE

# Batches and pause are the same as the manual /autoscan (see full_market_sweep):
# Yahoo's rate limit is per IP, so "speed it up with a thread pool" is not possible —
# parallelism gets you banned sooner, not later.
SWEEP_BATCH_SIZE = 25
SWEEP_PROGRESS_EVERY = 20

DEFAULTS = {
    "enabled": False,   # enabled only by an explicit command
    "want": 10,
    "hour": 7,          # ET, see the docstring about 9:00
    "minute": 45,
    "chat_id": None,    # remembered when enabled, falls back to MY_CHAT_ID
}


def load_config():
    """Settings with defaults filled in for missing keys."""
    cfg = dict(DEFAULTS)
    try:
        with open(MORNING_CONFIG_FILE, encoding="utf-8") as f:
            cfg.update(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return cfg


def save_config(cfg):
    try:
        os.makedirs(os.path.dirname(MORNING_CONFIG_FILE), exist_ok=True)
        with open(MORNING_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False)
        return True
    except OSError as exc:
        print(f"   ⚠️ Morning scan settings were not saved: {str(exc)[:80]}")
        return False


def run_morning_scan(want, status_callback=None, progress_callback=None):
    """
    Full morning run: market sweep → relaxation ladder → list for
    /queue. Synchronous and takes 40-50 minutes — call via asyncio.to_thread,
    otherwise the whole bot stalls (same class of bug that was fixed in /now).

    Returns (report_text, candidates, stale). text=None means nothing
    worked at all: the sweep failed and there is no saved data.
    """
    from agents.screener import (append_find_log, find_candidates,
                                 format_find_report, full_market_sweep,
                                 load_raw_sweep_results,
                                 save_autoscan_candidates,
                                 save_raw_sweep_results)

    stale = False
    results = []
    try:
        results = full_market_sweep(SWEEP_BATCH_SIZE, SWEEP_PROGRESS_EVERY,
                                    status_callback, progress_callback)
    except Exception as exc:
        # A sweep failure must not silently turn into "nothing today":
        # yesterday's sweep is picked up below, and the reason goes to the log.
        print(f"   ❌ [morning] sweep failed: {str(exc)[:200]}")

    if results:
        save_raw_sweep_results(results)
        generated_at = datetime.now().isoformat()
    else:
        results, generated_at = load_raw_sweep_results()
        stale = True

    if not results:
        return None, [], True

    candidates, rung, counts, _ = find_candidates(results, want)
    text, shown = format_find_report(want, candidates, rung, counts, generated_at)
    if shown:
        save_autoscan_candidates(shown, 0)

    append_find_log({
        "at": datetime.now().isoformat(), "want": want, "rung": rung,
        "counts": counts, "found": len(shown),
        "enough": len(candidates) >= want, "source": "morning",
        "stale": stale, "sweep_generated_at": generated_at,
    })
    return text, shown, stale
