# -*- coding: utf-8 -*-
"""FINRA Reg SHO short-volume share — diagnostics, NOT part of the score.

SOURCE. FINRA publishes ONE file per day for the whole market (not per ticker):
https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt — the combined
consolidated short volume (all reporting facilities at once), pipe-delimited:
Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market. Free, no
key, no registration — verified live on 2026-09-11 (real rows, not
documentation). The format was confirmed by requesting the file itself, not from a description.

LAG. The file for date D appears on the next trading day — trying to
download today's file will almost certainly give 403/404. The file published is for the
LAST COMPLETED trading day for which exchange data has been processed.
The function falls back to an earlier date by itself if the requested one is not yet / no longer
available (weekends, holidays, today).

WHAT IT DOES NOT MEAN. ShortVolume is the volume of trades marked short at
submission (mostly market makers/short sellers for ordinary reasons:
hedging, market making), NOT necessarily a "bearish bet". A high short
share by itself is not a direction signal — it is an external fact about volume
structure, not a reinterpretation of the same candle (unlike the former Order Flow).

WHY DIAGNOSTICS ONLY — AND THIS IS A CLOSED QUESTION, NOT AN OPEN ONE. Tested on
history on 2026-09-12 (agents/historical_theories.py, task 2): short share
in hindsight over 2346 observations (25 tickers, 3 years) vs. actual
return over the next 3 days — Pearson correlation r=0.02, share of down
days at high short (>=50%) 49.5% vs. 50.9% at low — i.e.
no difference at all. This is NOT "too little data, let's wait" (as in real_flow.py,
where a backtest is physically impossible — there is no historical Alpaca trade archive)
— it is a negative result on a sufficient sample. It is not in the score not
because it was not measured, but because it was measured and no effect was found. Revisiting
makes sense only with a new hypothesis/new data, not as "time has passed,
let's try again".
"""
import io
from datetime import datetime, timedelta

import requests

from config import SHORT_VOLUME_CACHE_DIR

_session = requests.Session()
_MAX_LOOKBACK_DAYS = 6   # long weekend + holiday — no more than a week back
_TIMEOUT = 15

BASE_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"


def _cache_path(date_str: str) -> str:
    import os
    os.makedirs(SHORT_VOLUME_CACHE_DIR, exist_ok=True)
    return os.path.join(SHORT_VOLUME_CACHE_DIR, f"CNMSshvol{date_str}.txt")


def _download_day(date_str: str) -> str | None:
    """Downloads (or takes from the disk cache) the raw file for one day.
    Cached because within an hour of autopilot the same date is requested
    dozens of times, and the whole-market file does not change retroactively."""
    import os
    path = _cache_path(date_str)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    url = BASE_URL.format(date=date_str)
    resp = _session.get(url, timeout=_TIMEOUT)
    if resp.status_code != 200 or not resp.text.startswith("Date|Symbol"):
        return None
    with open(path, "w", encoding="utf-8") as f:
        f.write(resp.text)
    return resp.text


def _parse_day(raw: str) -> dict:
    out = {}
    lines = raw.splitlines()
    for line in lines[1:]:
        parts = line.split("|")
        if len(parts) < 5:
            continue
        date_s, symbol, short_v, _short_exempt_v, total_v = parts[:5]
        try:
            short_v = float(short_v)
            total_v = float(total_v)
        except ValueError:
            continue
        if total_v <= 0:
            continue
        out[symbol] = {
            "date": date_s,
            "short_volume": short_v,
            "total_volume": total_v,
            "short_ratio": round(short_v / total_v, 4),
        }
    return out


def get_short_volume(ticker: str, as_of: datetime = None) -> dict:
    """Short-volume share for a ticker on the latest available trading day.

    Returns {} if there is no data (the file is not published for any day
    in the fallback window, the ticker is not in the file, or a network error) — calling code
    must not fail because diagnostics are missing.
    """
    as_of = as_of or datetime.now()
    for back in range(_MAX_LOOKBACK_DAYS):
        day = as_of - timedelta(days=back)
        date_str = day.strftime("%Y%m%d")
        try:
            raw = _download_day(date_str)
        except Exception:
            raw = None
        if raw is None:
            continue
        parsed = _parse_day(raw)
        row = parsed.get(ticker.upper())
        if row:
            return row
        # The file for this day exists but the ticker is not in it (low turnover that
        # day, delisting, typo) — do not fall back further by date,
        # this is a different case from "the file is not there yet".
        return {}
    return {}
