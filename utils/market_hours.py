"""
Trading session — a single primitive for the pipeline, signal history and statistics.

NYSE/NASDAQ hours: 9:30–16:00 ET, Mon–Fri. The container runs with
TZ=America/New_York (docker-compose.yml), so datetime.now() is already in exchange
time. yfinance also returns the index of US hourly candles in America/New_York
(verified) — after tz_localize(None) the timestamps compare with datetime.now() directly.

Holidays are deliberately NOT hardcoded and no dependency like
pandas_market_calendars is pulled in: the fact "there was a session" is derived from the data itself (are there
bars for that date). The same principle is used when resolving a signal outcome.
"""
from datetime import datetime, time, timedelta
from config import EXTENDED_HOURS_ENABLED, PREMARKET_OPEN_HHMM, AFTERHOURS_CLOSE_HHMM

# Regular session
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

# Extended session (pre-market + after-hours), if enabled in config
EXTENDED_OPEN = time(*PREMARKET_OPEN_HHMM)
EXTENDED_CLOSE = time(*AFTERHOURS_CLOSE_HHMM)


def session_bounds():
    """Trading-window bounds, depending on whether the extended session is enabled."""
    if EXTENDED_HOURS_ENABLED:
        return EXTENDED_OPEN, EXTENDED_CLOSE
    return MARKET_OPEN, MARKET_CLOSE

# yfinance hourly bars appear with a delay: in the first minutes of the session there may be
# no data for today yet. In this window we trust the clock, not the absence of bars,
# otherwise a live morning signal would wrongly move to "the next session".
DATA_LAG_GRACE_MIN = 60


# ═══════════════════════════════════════════════════════════════════════════
# EXCHANGE HOLIDAYS — for PLANNING only, not for analysing the past.
#
# The module's principle (see the docstring above) is to derive "there was a session" from the
# data itself, and for the past that is correct: bars either exist or they don't. But
# the morning task decides BEFORE the open whether to run at all, and at 07:45
# there is nothing to derive from — today's bars do not exist yet. So the list
# is needed, and it does not replace the old method but complements it for a different question.
#
# THE LIST IS MANUAL AND FINITE. Pulling in pandas_market_calendars for twenty
# lines was not worth it. Dates are observed NYSE/NASDAQ holidays: when
# a holiday falls on a Saturday the exchange is closed on Friday, when on a
# Sunday — on Monday (e.g. July 4, 2026 is a Saturday, hence July 3).
# Early-close sessions (day after Thanksgiving, Christmas Eve) are NOT included:
# the exchange is open, it just closes at 13:00, and that does not change the morning sweep.
# When the list runs out, is_trading_holiday starts warning in the log —
# see HOLIDAYS_KNOWN_THROUGH.
# ═══════════════════════════════════════════════════════════════════════════
MARKET_HOLIDAYS = {
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    # 2027
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}
HOLIDAYS_KNOWN_THROUGH = "2027-12-31"


def is_trading_holiday(day=None):
    """
    True if the exchange is closed for a holiday on that day. Weekends are NOT considered
    (weekday handles those) — the function is responsible for holidays only.

    Reports list exhaustion to the log instead of silently returning False: quietly
    treating all future days as trading days is exactly the failure you notice
    a year later through strangely empty sweeps.
    """
    day = day or datetime.now()
    day = day.date() if hasattr(day, "date") else day
    iso = day.isoformat()
    if iso > HOLIDAYS_KNOWN_THROUGH:
        print(f"   ⚠️ The exchange holiday list ended on "
              f"{HOLIDAYS_KNOWN_THROUGH}, {iso} is treated as a trading day — "
              f"time to extend MARKET_HOLIDAYS in utils/market_hours.py")
    return iso in MARKET_HOLIDAYS


def is_trading_day(day=None):
    """A weekday and not a holiday. For the morning task scheduler."""
    day = day or datetime.now()
    return day.weekday() < 5 and not is_trading_holiday(day)


def is_within_market_hours(now=None):
    """
    Mon–Fri and the time is inside the ET trading window. Knows nothing about holidays.
    The window is the regular session (9:30-16:00) or the extended one (see config), if enabled.
    """
    now = now or datetime.now()
    if now.weekday() >= 5:
        return False
    open_t, close_t = session_bounds()
    return open_t <= now.time() <= close_t


def is_extended_hours(now=None):
    """True if it is currently pre-market or after-hours (but not the regular session)."""
    now = now or datetime.now()
    if not EXTENDED_HOURS_ENABLED or now.weekday() >= 5:
        return False
    t = now.time()
    return (EXTENDED_OPEN <= t < MARKET_OPEN) or (MARKET_CLOSE < t <= EXTENDED_CLOSE)


def last_bar_date(df):
    """Date of the last bar in the data, or None."""
    if df is None or len(df) == 0:
        return None
    try:
        return df.index[-1].date()
    except AttributeError:
        return None


def pending_session_key(now=None):
    """
    Logical session key — which session the plan belongs to, as a "YYYY-MM-DD" string.
    Used for deduplication: all signals belonging to one session get
    the same key.

    A weekday before the end of the trading window (16:00, or 20:00 with the extended
    session enabled) -> today. After the close / weekends -> the nearest
    next weekday.

    Holiday accuracy does NOT matter here: the key must be STABLE (the same
    for all signals of one window), and the actual session is still derived from
    the data when the outcome is resolved. If a holiday makes the key a day earlier
    than the real session, all signals of that window still stick together correctly.
    """
    now = now or datetime.now()
    _, close_t = session_bounds()

    if now.weekday() < 5 and now.time() <= close_t:
        return str(now.date())

    candidate = now.date() + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return str(candidate)


def market_status(df_hourly=None, now=None):
    """
    Determines whether a live trading session is in progress.

    df_hourly (optional) is used to rule out holidays: if the clock
    fits but there are no fresh bars for today and the data-delay grace window
    has passed, it is most likely a holiday. Without df, only the clock is used.

    Returns a dict:
      status:           "open" | "closed"
      session_date:     date of the live session (date), or None if no session is running
      outside_session:  True if the plan is built outside a live session
      session_key:      string session key for deduplication (see pending_session_key)
      extended:         True if it is now pre-market/after-hours rather than the regular session
    """
    now = now or datetime.now()
    session_key = pending_session_key(now)
    extended = is_extended_hours(now)

    closed = {"status": "closed", "session_date": None, "outside_session": True,
              "session_key": session_key, "extended": False}

    if not is_within_market_hours(now):
        return closed

    if df_hourly is not None:
        bar_date = last_bar_date(df_hourly)
        if bar_date != now.date():
            open_t, _ = session_bounds()
            minutes_since_open = (
                now - now.replace(hour=open_t.hour, minute=open_t.minute,
                                  second=0, microsecond=0)
            ).total_seconds() / 60
            if minutes_since_open > DATA_LAG_GRACE_MIN:
                # Trading hours, but no data for today and a delay no longer
                # explains it — probably a holiday.
                return closed

    # Pre-market/after-hours count as tradable time (outside_session=False):
    # the broker allows entering right now, so the price is real and there is no need
    # to re-anchor the entry to the regular-session open.
    return {"status": "open", "session_date": now.date(), "outside_session": False,
            "session_key": session_key, "extended": extended}


def format_market_status(info):
    """A short human-readable note for reports."""
    if info.get("status") != "open":
        return (
            "⚠️ *Market is closed now.* Price and levels are as of the last close; "
            "the plan is indicative, the entry must be recomputed at the next session's open."
        )
    if info.get("extended"):
        return (
            "🌙 *Extended session* (pre-market/after-hours). Trading is available, but "
            "liquidity is noticeably lower: wider spread, sharper slippage, levels are less "
            "reliable — position size should be reduced."
        )
    return ""
