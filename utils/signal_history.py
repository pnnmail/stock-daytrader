import collections
import json
import statistics
from datetime import datetime, time as dtime
from config import SIGNAL_HISTORY_FILE, NO_TRADE_HISTORY_FILE, LONG_THRESHOLD
from utils.helpers import safe_format_float, escape_markdown
from utils.signals import MIN_ENTRY_RR, SCORING_VERSION, MAX_TOTAL_SCORE
from utils.market_hours import MARKET_OPEN

# Score formula version before the tag was introduced — all records without the field
# were computed as a flat sum of five factors with a ±4 threshold.
LEGACY_SCORING_VERSION = "v1_flat_sum"

# Below this n in WINDOWED statistics (/stats 7d etc.) a warning is printed:
# on such a sample the win rate is noise, not signal.
MIN_WINDOW_N = 10

# Outcomes that mean "there was effectively no trade" — the entry turned out
# impossible or pointless because of a gap. Neither a win nor a loss: they are not
# included in return, Sharpe or win rate, they are counted separately.
NO_TRADE_OUTCOMES = ("gap_invalidated", "gap_no_edge")

# ═══════════════════════════════════════════════════════════════════════════
# DAILY RESOLUTION RULE (day trade) — one session, close at 15:30.
#
# HOW IT DIFFERS FROM THE OLD ONE. _resolve_signal_outcome looks at the signal session AND
# THE NEXT ONE. For a day trade that answers a different question: a trade that failed
# intraday and recovered the next day at the open was counted as a win, although
# the rule would have closed it the day before.
#
# WHY THE "none" OUTCOME DISAPPEARS. With two sessions "reached neither the target nor
# the stop" is an unresolved case: it went into the win_rate denominator but never
# into the numerator (recorded as an open question). With
# a forced close at 15:30 this is not a gap but a CLOSED TRADE with a real
# result — the "close" outcome with its own sign. There is no third bucket any more.
#
# CUT-OFF BY BIRTH TIME. A signal at 15:20 lives ten minutes — that is noise, not
# an observation. Later than INTRADAY_CUTOFF such signals are marked "too_late" and
# go into the statistics on neither side.
#
# REGULAR SESSION ONLY, PRE-MARKET DOES NOT COUNT. market_status()
# (utils/market_hours.py) treats pre-market/after-hours as tradable time
# (outside_session=False) — the broker lets you enter, so the price is real. Because of
# this the daily rule silently caught pre-market signals by default as well
# (04:00–09:30): "generated_outside_session" did not cut them, and they fitted
# before INTRADAY_CUTOFF. Found on live data on 2026-08-14 —
# 4 of 6 signals closed that day turned out to be pre-market (08:22–08:30),
# which mixed fundamentally different liquidity and spread into one statistic.
# The explicit lower bound MARKET_OPEN separates this NOT via
# outside_session (that one stays about real price tradability and is needed
# by other consumers of market_status), but right here, as a decision of the
# DAILY RULE specifically — pre-market is labelled with a separate "premarket" outcome.
#
# THE TWO RULES MUST NOT BE MIXED. A record has resolution_rule, and statistics
# are computed separately for each: adding them gives a number that means
# nothing. The project already got burned on the definition of a win (the
# gap_no_edge fix struck out three wins out of four).
# ═══════════════════════════════════════════════════════════════════════════
INTRADAY_CUTOFF = dtime(15, 0)    # later signals are not resolved
INTRADAY_CLOSE = dtime(15, 30)    # forced close
RULE_INTRADAY = "intraday_1530"
RULE_TWO_SESSIONS = "two_sessions"

# Outcomes that are NOT a trade and go into the statistics as neither numerator nor
# denominator. They must be distinguished from a loss: silently dropping
# unresolved ones is survivorship bias if non-resolution correlates with something
# (e.g. a trading halt, which is exactly what the most volatile stocks get).
EXCLUDED_OUTCOMES = ("too_late", "unresolved", "out_of_session", "premarket")

# Fixed amount for the money simulation in /stats ($100 per signal,
# regardless of the plan's real position_size_pct — that one varies with
# trade confidence, while the simulation needs ONE amount, the same for all,
# otherwise days with different sets of signals could not be compared fairly
# with each other). $100 * return_pct/100 is numerically equal to return_pct in dollars — this is
# not a new quantity but the same percentage, just in a form that feels
# more concrete: "+$8 for the day" reads differently from "+1.4% average per trade".
SIMULATED_POSITION_USD = 100

# data.fetcher pulls in requests/yfinance — imported lazily inside the function
# so the signal_history module can be imported (and the history read)
# without network dependencies, if needed.

TRADE_DIRECTIONS = {"LONG", "SHORT", "STRONG_LONG", "STRONG_SHORT"}
MAX_RECORDS_PER_TICKER = 50

# Per-ticker limit for the control group — lower than for trades: NO_TRADE is written
# on EVERY run of every ticker, and without a stricter cap the file would grow
# disproportionately faster, while a few dozen observations per stock are enough
# to compare distributions.
NO_TRADE_MAX_RECORDS_PER_TICKER = 30


def load_signal_history():
    try:
        with open(SIGNAL_HISTORY_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_signal_history(history):
    with open(SIGNAL_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def load_no_trade_history():
    try:
        with open(NO_TRADE_HISTORY_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_no_trade_history(history):
    with open(NO_TRADE_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def tickers_seen_today(when=None):
    """
    Tickers that ALREADY have a record TODAY — a trade or NO_TRADE.

    Needed by /autopilot: without it an expensive LLM call can be burned confirming
    an already known outcome for today (deduplication by session_key in
    append_signal fires AFTER the full pipeline run, it does not save the call
    itself). Meaningful only during real trading hours — outside the session or
    on synthetic runs "today" guarantees nothing, so
    /autopilot can bypass this check with the force flag.
    """
    today = (when or datetime.now()).date().isoformat()
    seen = set()
    for history in (load_signal_history(), load_no_trade_history()):
        for ticker, records in history.items():
            for r in records:
                if (r.get("timestamp") or "")[:10] == today:
                    seen.add(ticker)
                    break
    return seen


def _evidence_fields(trader_plan):
    """
    THE EVIDENCE the decision is built on — what cannot be reconstructed
    after the fact. Price and candles are always available in yfinance by (ticker, date), but
    the sentiment and number of risk flags the news assessment came out with at
    that moment are not. Without these fields one cannot answer whether a factor deserves
    its weight: the history only had the final decision, not its grounds.
    """
    return {
        "total_score": trader_plan.get("direction_score"),
        "scores": trader_plan.get("scores"),
        "reasons": trader_plan.get("reasons"),
        "features": trader_plan.get("features"),
        # Score FORMULA version. Without it observations of different versions are indistinguishable,
        # and they must not be added up: the score means different things (v1 — a flat sum of five
        # factors with a ±4 threshold, v2 — the momentum group compressed to ±1, threshold ±3).
        # The same logic as resolution_rule for the outcome resolution rule.
        "scoring_version": SCORING_VERSION,
    }


def append_signal(ticker, trader_plan, market_info=None):
    """
    Saves a signal to history. Real trades (LONG/SHORT) go to
    signal_history, rejected ones (NO_TRADE) — to a separate
    control-group store (see append_no_trade).

    Previously NO_TRADE was simply dropped ("nothing to remember about it"). That
    was survivorship bias baked into data collection: the history kept only
    signals that passed the threshold, so the question "does the threshold distinguish anything at all?"
    could not be tested — there was nothing to compare against.

    market_info (from utils.market_hours.market_status) marks which session
    the plan belongs to and whether it was built outside a live session. This is needed to
    compute return honestly later: for an out-of-session signal entry_price is
    the last close, you cannot enter at it, and when the outcome is resolved the entry
    is recomputed to the session open (see _resolve_signal_outcome).
    """
    direction = trader_plan.get("direction")
    if direction not in TRADE_DIRECTIONS:
        append_no_trade(ticker, trader_plan, market_info=market_info)
        return

    market_info = market_info or {}
    session_date = market_info.get("session_date")
    session_key = market_info.get("session_key")
    now_iso = datetime.now().isoformat(timespec="seconds")

    history = load_signal_history()
    ticker_history = history.get(ticker, [])

    # Deduplication: one session + one direction = ONE observation.
    # /go polls every N minutes and recreates the same plan dozens of times per session;
    # without this /stats would count one idea as N independent trades (inflated n with
    # zero new information), and the MAX_RECORDS_PER_TICKER limit would push out real
    # history within a couple of days. We keep the FIRST signal of the session — that is the moment
    # one could have entered; repeats are only recorded with a counter.
    if session_key:
        for existing in ticker_history:
            if existing.get("session_key") == session_key and existing.get("direction") == direction:
                existing["repeat_count"] = existing.get("repeat_count", 1) + 1
                existing["last_seen"] = now_iso
                save_signal_history(history)
                return

    record = {
        "timestamp": now_iso,
        "session_date": str(session_date) if session_date else None,
        "session_key": session_key,
        "generated_outside_session": bool(market_info.get("outside_session", False)),
        "repeat_count": 1,
        "direction": direction,
        "entry_price": trader_plan.get("entry_price"),
        "stop_loss": trader_plan.get("stop_loss"),
        "target_1": trader_plan.get("target_1"),
        "target_2": trader_plan.get("target_2"),
        "risk_reward_1": trader_plan.get("risk_reward_1"),
        "risk_reward_2": trader_plan.get("risk_reward_2"),
        "confidence": trader_plan.get("confidence"),
        "position_size_pct": trader_plan.get("position_size_pct"),
        **_evidence_fields(trader_plan),
    }

    ticker_history.append(record)
    history[ticker] = ticker_history[-MAX_RECORDS_PER_TICKER:]
    save_signal_history(history)


def append_no_trade(ticker, trader_plan, market_info=None):
    """
    Saves a REJECTED signal — the control group for testing the entry threshold.

    The outcome here is not computed by stop/target: NO_TRADE has no levels. It must be
    measured by forward return from the price at decision time, and that is reconstructed
    after the fact from (ticker, timestamp) via yfinance — so here it is enough
    to record THE FACT of the decision itself and its grounds. It is the fact that cannot be reconstructed:
    which tickers the system rejected on a given day is not recorded in quotes.
    """
    market_info = market_info or {}
    session_key = market_info.get("session_key")
    now_iso = datetime.now().isoformat(timespec="seconds")

    total_score = trader_plan.get("direction_score")
    # The same criterion as in format_no_trade_report: the score reached the threshold, but
    # the trade was removed by the entry filter — this is a DIFFERENT population from "not enough points",
    # and they must not be mixed in analysis.
    passed_threshold = (
        total_score is not None and abs(total_score) >= LONG_THRESHOLD
    )

    history = load_no_trade_history()
    ticker_history = history.get(ticker, [])

    # Deduplication as for trades: /go recreates the same output dozens of times per
    # session; without it the control group would be inflated by repeats of one decision.
    if session_key:
        for existing in ticker_history:
            if existing.get("session_key") == session_key:
                existing["repeat_count"] = existing.get("repeat_count", 1) + 1
                existing["last_seen"] = now_iso
                save_no_trade_history(history)
                return

    record = {
        "timestamp": now_iso,
        "session_date": str(market_info.get("session_date") or "") or None,
        "session_key": session_key,
        "generated_outside_session": bool(market_info.get("outside_session", False)),
        "repeat_count": 1,
        "direction": "NO_TRADE",
        "no_trade_kind": "entry_filter" if passed_threshold else "below_threshold",
        "entry_filter_reason": trader_plan.get("entry_filter_reason"),
        **_evidence_fields(trader_plan),
    }

    ticker_history.append(record)
    history[ticker] = ticker_history[-NO_TRADE_MAX_RECORDS_PER_TICKER:]
    save_no_trade_history(history)


def _session_key_of(record):
    """
    The record's session key. For records created before deduplication appeared,
    it is derived from timestamp by the same rule (pending_session_key).
    """
    key = record.get("session_key")
    if key:
        return key
    try:
        from utils.market_hours import pending_session_key
        return pending_session_key(datetime.fromisoformat(record["timestamp"]))
    except Exception:
        return record.get("timestamp", "")


def _dedupe_records(records):
    """
    Collapses records of one session and direction into one observation (keeping
    the earliest — the moment one could actually have entered), summing repeat_count.

    Needed at READ time in addition to deduplication at write time: old records
    saved before it appeared would otherwise keep counting as several
    independent trades in /stats. Deletes nothing from disk — only groups
    during calculation.
    """
    grouped = {}
    for r in sorted(records, key=lambda x: x.get("timestamp", "")):
        key = (_session_key_of(r), r.get("direction"))
        if key in grouped:
            grouped[key]["repeat_count"] = grouped[key].get("repeat_count", 1) + r.get("repeat_count", 1)
        else:
            grouped[key] = r
    return list(grouped.values())


def _humanize_when(timestamp_str):
    try:
        dt = datetime.fromisoformat(timestamp_str)
    except ValueError:
        return timestamp_str, 0
    days = (datetime.now().date() - dt.date()).days
    if days <= 0:
        return f"today {dt.strftime('%H:%M')}", days
    elif days == 1:
        return f"yesterday {dt.strftime('%H:%M')}", days
    elif days == 2:
        return f"2 days ago {dt.strftime('%H:%M')}", days
    else:
        return f"{days} d. ago ({dt.strftime('%d.%m %H:%M')})", days


def _resolve_signal_outcome(ticker, record):
    """
    Determines a signal's outcome from the REAL hourly candles of the day the signal was
    given and the next trading day — whether the price touched the stop/target in exactly that
    period (not "where the price is now").

    Returns a dict {"outcome", "day", "price", "time"} or None if there is not yet enough data
    for a final conclusion (the next trading day has not closed yet
    or yfinance did not return data) — in that case try again later.
    """
    from data.fetcher import fetch_signal_day_ohlc  # lazy import: yfinance/requests are not needed to read history

    try:
        signal_dt = datetime.fromisoformat(record["timestamp"])
    except (ValueError, KeyError, TypeError):
        return None

    direction = record.get("direction", "")
    is_long = direction in ("LONG", "STRONG_LONG")
    is_short = direction in ("SHORT", "STRONG_SHORT")
    if not (is_long or is_short):
        return None

    stop = record.get("stop_loss")
    t1 = record.get("target_1")
    t2 = record.get("target_2")
    if record.get("entry_price") is None:
        return None

    df = fetch_signal_day_ohlc(ticker, signal_dt)
    if df is None or df.empty:
        return None

    if getattr(df.index, "tz", None) is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)

    # The applicable session = the first session that has bars STRICTLY AFTER the moment
    # of the signal. Derived from the data, so it is equally correct for all cases:
    #   signal at 11:00 in a live session   -> this same session
    #   pre-market at 07:00               -> that day's session
    #   after the close at 18:00          -> the next trading session
    #   weekend/holiday               -> the next trading session
    future_bars = df[df.index > signal_dt]
    if future_bars.empty:
        # The session the plan belongs to has not started yet (or there is no data yet)
        return None

    primary_session = future_bars.index[0].date()
    primary_bars = future_bars[future_bars.index.date == primary_session]

    outside = record.get("generated_outside_session")
    if outside is None:
        # Records created before this flag appeared: treat them as out-of-session if
        # the applicable session is not the signal day (i.e. that day's session had already closed).
        outside = primary_session != signal_dt.date()

    # For an out-of-session signal entry_price is the last close, you cannot enter at it
    # (there is almost always a gap at the open). We recompute the entry to the actual
    # open of the applicable session, otherwise the return in /stats would be over/understated.
    if outside:
        effective_entry = float(primary_bars["Open"].iloc[0])
        reanchored = True
    else:
        effective_entry = record.get("entry_price")
        reanchored = False

    later_dates = sorted({d for d in df.index.date if d > primary_session})

    def scan(rows, day_label):
        for ts, row in rows.iterrows():
            try:
                high, low = float(row["High"]), float(row["Low"])
            except (TypeError, ValueError):
                continue
            if is_long:
                stop_hit = stop is not None and low <= stop
                t2_hit = t2 is not None and high >= t2
                t1_hit = t1 is not None and high >= t1
            else:
                stop_hit = stop is not None and high >= stop
                t2_hit = t2 is not None and low <= t2
                t1_hit = t1 is not None and low <= t1
            # If both hit in one bar, assume the stop fired first (conservative)
            if stop_hit:
                return {"outcome": "stop", "day": day_label, "price": stop, "time": ts.isoformat()}
            if t2_hit:
                return {"outcome": "target_2", "day": day_label, "price": t2, "time": ts.isoformat()}
            if t1_hit:
                return {"outcome": "target_1", "day": day_label, "price": t1, "time": ts.isoformat()}
        return None

    def finish(result, session):
        """Adds the applicable session and the actual entry to the outcome."""
        result["session"] = str(session)
        result["effective_entry"] = effective_entry
        result["reanchored"] = reanchored
        return result

    # The gap can be so large that the open is already BEYOND the planned stop
    # (the stop was computed from a stale entry_price). In reality such a trade
    # could not have been taken — the stop would have fired before the entry. Case HURN 2026-07-31:
    # LONG entry=166.49/stop=163.07, actual open 159.57 — BELOW the stop;
    # the naive formula (outcome_price-entry)/entry gave +2.19% by signs
    # instead of a real loss, because effective_entry was beyond the stop.
    if outside and stop is not None:
        gap_past_stop = (is_long and effective_entry <= stop) or (is_short and effective_entry >= stop)
        if gap_past_stop:
            return finish(
                {
                    "outcome": "gap_invalidated",
                    "day": f"gap past the stop at the open ({primary_session:%d.%m})",
                    "price": effective_entry,
                    "time": primary_bars.index[0].isoformat(),
                },
                primary_session,
            )

        # The mirror case of the previous one, and it is more dangerous because it looks
        # plausible. A gap IN THE DIRECTION of the trade does not cancel the stop, but it eats
        # the potential: the entry moves toward the target and away from the stop, R:R collapses. At the same time
        # the target becomes EASIER to reach — i.e. the bias is directional,
        # such records mostly land in the statistics as WINS.
        # Case VEON 2026-08-02: plan entry 51.58 / stop 50.97 / target 52.93
        # (R:R 2.2); Monday open 52.85 — the target 0.15% above, the stop
        # 3.56% below, actual R:R 0.04. Recorded as target_1 in the very first
        # bar of the session (09:30). Of the 4 wins in the history at the time of the fix, 3 were
        # like this.
        # The threshold is the same MIN_ENTRY_RR as in check_entry_filters: if on
        # the actual entry the trade would not have passed the filter, it could not
        # have been taken, and it must not go into the return — exactly the same logic as
        # gap_invalidated above.
        if t1 is not None:
            risk = abs(effective_entry - stop)
            reward = abs(t1 - effective_entry)
            actual_rr = (reward / risk) if risk > 0 else 0.0
            if actual_rr < MIN_ENTRY_RR:
                return finish(
                    {
                        "outcome": "gap_no_edge",
                        "day": (f"the gap ate the potential at the open "
                                f"({primary_session:%d.%m}, R:R became "
                                f"{actual_rr:.2f})"),
                        "price": effective_entry,
                        "time": primary_bars.index[0].isoformat(),
                    },
                    primary_session,
                )

    result = scan(primary_bars, f"in its own session ({primary_session:%d.%m})")
    if result:
        return finish(result, primary_session)

    if not later_dates or later_dates[0] >= datetime.now().date():
        # The next session has not started/closed yet — too early for a final conclusion
        return None

    next_session = later_dates[0]
    next_bars = df[df.index.date == next_session]
    result = scan(next_bars, f"in the next session ({next_session:%d.%m})")
    if result:
        return finish(result, next_session)

    close_price = float(next_bars["Close"].iloc[-1]) if not next_bars.empty else None
    return finish(
        {"outcome": "none", "day": "over 2 sessions", "price": close_price, "time": None},
        next_session,
    )


def _resolve_intraday_outcome(ticker, record):
    """
    Daily resolution: stop/target on the signal day's 5-minute bars, and if neither
    was touched — a forced close at the 15:30 price (the "close" outcome with its own
    sign, not a data gap). Only REGULAR-session signals count,
    born in the window [MARKET_OPEN, INTRADAY_CUTOFF) = [09:30, 15:00) — pre-market
    (04:00–09:30) is cut off with the "premarket" outcome, after-hours and out-of-session
    hours — "too_late"/"out_of_session".

    Order within a bar — stop first, as in the two-session resolution.
    The assumption is conservative, i.e. it biases the result AGAINST the system, which is
    the only safe direction of error when evaluating your own
    strategy.

    Returns a dict with the outcome, or None if the record cannot be resolved
    at all (no direction, entry or time).
    """
    from data.fetcher import fetch_signal_day_intraday

    try:
        signal_dt = datetime.fromisoformat(record["timestamp"])
    except (ValueError, KeyError, TypeError):
        return None

    direction = record.get("direction", "")
    is_long = direction in ("LONG", "STRONG_LONG")
    is_short = direction in ("SHORT", "STRONG_SHORT")
    if not (is_long or is_short) or record.get("entry_price") is None:
        return None

    if record.get("generated_outside_session"):
        return {"outcome": "out_of_session",
                "day": "signal born outside the session — the daily rule does not apply",
                "price": None, "time": None}

    if signal_dt.time() < MARKET_OPEN:
        # market_status() treats pre-market as tradable time (the broker lets you
        # enter), so generated_outside_session is False here and it was not
        # cut above. But pre-market liquidity and spread are not those of the regular
        # session — the daily rule must not mix them (see the block comment
        # above about the 2026-08-14 finding: 4 of 6 trades that day were
        # pre-market and distorted the statistics).
        return {"outcome": "premarket",
                "day": f"signal at {signal_dt:%H:%M}, before the open "
                       f"{MARKET_OPEN:%H:%M} — pre-market, not the regular session",
                "price": None, "time": None}

    if signal_dt.time() >= INTRADAY_CUTOFF:
        return {"outcome": "too_late",
                "day": f"signal at {signal_dt:%H:%M}, after the cut-off "
                       f"{INTRADAY_CUTOFF:%H:%M} — window too short",
                "price": None, "time": None}

    df = fetch_signal_day_intraday(ticker, signal_dt)
    if df is None or df.empty:
        return {"outcome": "unresolved",
                "day": "there was no intraday data for the signal day",
                "price": None, "time": None}

    if getattr(df.index, "tz", None) is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)

    day = signal_dt.date()
    bars = df[(df.index > signal_dt)
              & (df.index.date == day)
              & (df.index.time <= INTRADAY_CLOSE)]
    if bars.empty:
        return {"outcome": "unresolved",
                "day": "there was not a single bar between the signal and 15:30",
                "price": None, "time": None}

    stop = record.get("stop_loss")
    t1, t2 = record.get("target_1"), record.get("target_2")
    for ts, row in bars.iterrows():
        try:
            high, low = float(row["High"]), float(row["Low"])
        except (TypeError, ValueError):
            continue
        if is_long:
            stop_hit = stop is not None and low <= stop
            t2_hit = t2 is not None and high >= t2
            t1_hit = t1 is not None and high >= t1
        else:
            stop_hit = stop is not None and high >= stop
            t2_hit = t2 is not None and low <= t2
            t1_hit = t1 is not None and low <= t1
        if stop_hit:
            return {"outcome": "stop", "day": f"stop at {ts:%H:%M} ({day:%d.%m})",
                    "price": stop, "time": ts.isoformat()}
        if t2_hit:
            return {"outcome": "target_2", "day": f"target 2 at {ts:%H:%M} ({day:%d.%m})",
                    "price": t2, "time": ts.isoformat()}
        if t1_hit:
            return {"outcome": "target_1", "day": f"target 1 at {ts:%H:%M} ({day:%d.%m})",
                    "price": t1, "time": ts.isoformat()}

    last_ts = bars.index[-1]
    return {"outcome": "close",
            "day": f"closed by the rule at {last_ts:%H:%M} ({day:%d.%m})",
            "price": float(bars["Close"].iloc[-1]), "time": last_ts.isoformat()}


def _apply_outcome(record, resolved, rule):
    """
    Transfers the resolution result into the record and RECORDS THE RULE. Without the rule label
    it is later impossible to tell what computed the outcome, and two definitions
    of a win must not be mixed in one statistic.
    """
    record["outcome"] = resolved["outcome"]
    record["outcome_day"] = resolved["day"]
    record["outcome_price"] = resolved["price"]
    record["outcome_time"] = resolved.get("time")
    record["outcome_session"] = resolved.get("session")
    record["resolution_rule"] = rule
    if "effective_entry" in resolved:
        record["effective_entry_price"] = resolved.get("effective_entry")
        record["entry_reanchored"] = resolved.get("reanchored", False)


# ═══════════════════════════════════════════════════════════════════════════
# COUNTERFACTUAL OUTCOME FOR NO_TRADE — "what would have happened if the system had not
# been listened to". Without it the control group (append_no_trade) accumulates scores and
# rejection reasons but cannot answer the question it was set up for:
# does the entry threshold distinguish anything at all.
#
# THE DIRECTION IS TAKEN FROM THE SIGN OF total_score, EVEN FOR below_threshold. The two kinds
# of rejection are different hypotheses, and they must not be compared together:
#   - entry_filter: the score REALLY crossed the direction threshold, the rejection is by
#     a separate numeric criterion (R:R, volume...). Here the question is direct:
#     "did the entry filter save us from a loss or would it have cut off a profit?"
#   - below_threshold: the direction threshold was NOT passed at all. A weaker question:
#     "does even an insufficient score correlate with direction at all?" —
#     but this is exactly the test of whether the scoring itself distinguishes anything useful
#     below the official threshold, not only above it.
#
# NO STOP AND TARGET. NO_TRADE had no plan — nothing to check along the way,
# only the sign of the move to the 15:30 close, by the same daily rule as for
# real trades (see _resolve_intraday_outcome above). Separate fields
# (cf_*), not reusing outcome/resolution_rule — this is NOT the same
# observation as a real trade, and they must not be confused in one field.
# ═══════════════════════════════════════════════════════════════════════════
EXCLUDED_CF_OUTCOMES = ("premarket", "too_late", "out_of_session",
                        "unresolved", "no_direction")


def _resolve_no_trade_counterfactual(ticker, record):
    """
    Counterfactual outcome of one NO_TRADE record. Returns a dict with the outcome
    or None if the record cannot be resolved at all (no score,
    time or price at decision time).
    """
    from data.fetcher import fetch_signal_day_intraday

    try:
        signal_dt = datetime.fromisoformat(record["timestamp"])
    except (ValueError, KeyError, TypeError):
        return None

    total_score = record.get("total_score")
    # The same price the system saw at decision time (indicators['price'] at
    # analysis time) — no real trade takes "the price from a bar
    # minutes later" either, the entry in /now is built exactly the same way (calculate_entry_levels
    # from the same indicators['price']).
    entry_price = (record.get("features") or {}).get("price")
    if total_score is None or not entry_price:
        return None
    if total_score == 0:
        return {"outcome": "no_direction", "direction": None,
                "day": "score exactly 0 — no direction hypothesis",
                "price": None, "time": None}

    direction = "LONG" if total_score > 0 else "SHORT"

    if record.get("generated_outside_session"):
        return {"outcome": "out_of_session", "direction": direction,
                "day": "signal born outside the session — the daily rule does not apply",
                "price": None, "time": None}

    if signal_dt.time() < MARKET_OPEN:
        return {"outcome": "premarket", "direction": direction,
                "day": f"signal at {signal_dt:%H:%M}, before the open "
                       f"{MARKET_OPEN:%H:%M} — pre-market",
                "price": None, "time": None}

    if signal_dt.time() >= INTRADAY_CUTOFF:
        return {"outcome": "too_late", "direction": direction,
                "day": f"signal at {signal_dt:%H:%M}, after the cut-off "
                       f"{INTRADAY_CUTOFF:%H:%M}",
                "price": None, "time": None}

    df = fetch_signal_day_intraday(ticker, signal_dt)
    if df is None or df.empty:
        return {"outcome": "unresolved", "direction": direction,
                "day": "there was no intraday data for the signal day",
                "price": None, "time": None}

    if getattr(df.index, "tz", None) is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)

    day = signal_dt.date()
    bars = df[(df.index > signal_dt) & (df.index.date == day)
              & (df.index.time <= INTRADAY_CLOSE)]
    if bars.empty:
        return {"outcome": "unresolved", "direction": direction,
                "day": "there was not a single bar between the signal and 15:30",
                "price": None, "time": None}

    close_price = float(bars["Close"].iloc[-1])
    close_time = bars.index[-1]
    if direction == "LONG":
        ret = (close_price - entry_price) / entry_price * 100
    else:
        ret = (entry_price - close_price) / entry_price * 100

    return {"outcome": "would_win" if ret > 0 else "would_lose",
            "direction": direction, "return_pct": round(ret, 3),
            "day": f"closed by the rule at {close_time:%H:%M} ({day:%d.%m})",
            "price": close_price, "time": close_time.isoformat()}


def _apply_no_trade_outcome(record, resolved):
    record["cf_outcome"] = resolved["outcome"]
    record["cf_direction"] = resolved.get("direction")
    record["cf_return_pct"] = resolved.get("return_pct")
    record["cf_day"] = resolved["day"]
    record["cf_time"] = resolved.get("time")
    record["cf_resolution_rule"] = RULE_INTRADAY


def resolve_no_trade_backlog(when=None, limit=None, progress_every=25):
    """
    Computes the counterfactual outcome for NO_TRADE records whose window (signal day
    + 15:30) has already passed. Like resolve_intraday_today — by WINDOW, not
    by "today", self-healing after any skipped run.

    limit — process at most this many records per call. The backlog is hundreds
    of network requests; it is called both from the daily task (a small limit so as
    not to stretch it to dozens of minutes) and manually for a one-off
    catch-up of the history (limit=None — everything at once).

    Returns {"processed", "resolved", "excluded"} — counters, not a list:
    there may be hundreds of records, a detailed list in the chat makes no sense.
    """
    import time as _time

    when = when or datetime.now()
    history = load_no_trade_history()

    pending = []
    for ticker, records in history.items():
        for r in records:
            if r.get("cf_outcome"):
                continue
            try:
                ts = datetime.fromisoformat(r["timestamp"])
            except (ValueError, KeyError, TypeError):
                continue
            if when < datetime.combine(ts.date(), INTRADAY_CLOSE):
                continue
            pending.append((ticker, r))

    if limit is not None:
        pending = pending[:limit]

    counts = {"processed": 0, "resolved": 0, "excluded": 0}
    for i, (ticker, r) in enumerate(pending, 1):
        resolved = _resolve_no_trade_counterfactual(ticker, r)
        if resolved is None:
            continue
        _apply_no_trade_outcome(r, resolved)
        counts["processed"] += 1
        bucket = "excluded" if resolved["outcome"] in EXCLUDED_CF_OUTCOMES else "resolved"
        counts[bucket] += 1
        if i % progress_every == 0:
            save_no_trade_history(history)
            print(f"   [NO_TRADE counterfactual] {i}/{len(pending)} processed...", flush=True)
        _time.sleep(0.3)  # the same principle as the sweep's BATCH_PAUSE_SEC — don't burn Yahoo in a burst

    if counts["processed"]:
        save_no_trade_history(history)
    return counts


def compute_no_trade_counterfactual_stats():
    """
    Comparison: how rejected signals would have played out if they had been taken.
    Separately by rejection kind (below_threshold/entry_filter) — they must not be
    mixed, these are different hypotheses (see the block comment above).
    """
    history = load_no_trade_history()
    by_kind = {"below_threshold": [], "entry_filter": []}
    for records in history.values():
        for r in records:
            kind = r.get("no_trade_kind")
            ret = r.get("cf_return_pct")
            if kind in by_kind and ret is not None:
                by_kind[kind].append(ret)

    out = {}
    for kind, rets in by_kind.items():
        if not rets:
            out[kind] = {"n": 0}
            continue
        wins = sum(1 for r in rets if r > 0)
        out[kind] = {
            "n": len(rets),
            "would_win_rate": round(wins / len(rets) * 100, 1),
            "avg_return_pct": round(statistics.mean(rets), 2),
        }
    return out


def resolve_intraday_today(when=None):
    """
    Closes by the daily rule ALL still unresolved signals whose
    own window (signal day + INTRADAY_CLOSE) has already passed — not
    only today's. Called by the task after 15:30 ET.

    WHY NOT "only today's" (as before). The old version
    looked strictly at ts.date() == today at the moment it ran. If on
    the signal day the task did not fire (the container was not running at 15:35 —
    a rebuild, host sleep; in practice this is exactly what happened: Docker
    timestamps confirmed that the daily task was skipped for several
    days in a row, and the function body never ran once), the record dropped out of
    the check FOREVER — the next day that date stopped being "today"
    and never would be again. Result: almost the entire signal history
    hung with outcome=null indefinitely, although the data for an honest resolution was still
    available.

    Resolving the WINDOW rather than "today" fixes this without a retroactive script:
    it is enough for the task to fire at least once AFTER the gap, and it
    will pick up everything accumulated by itself — up to the limit of ~60 days while
    yfinance still returns 5-minute bars (see fetch_signal_day_intraday).
    Older records get an honest outcome="unresolved" instead of hanging
    silently.

    Returns {"resolved": [...], "excluded": [...]} for the chat message.
    """
    when = when or datetime.now()
    history = load_signal_history()
    changed = False
    summary = {"resolved": [], "excluded": []}

    for ticker, records in history.items():
        for r in records:
            if r.get("outcome"):
                continue
            try:
                ts = datetime.fromisoformat(r["timestamp"])
            except (ValueError, KeyError, TypeError):
                continue
            # The window has not closed yet (today's signal, before 15:30) — too early,
            # not "skipped". Checked by the day of the SIGNAL ITSELF, not by the
            # current day — so yesterday's and the day before's records still
            # pass this condition (their window is long in the past).
            window_close = datetime.combine(ts.date(), INTRADAY_CLOSE)
            if when < window_close:
                continue

            resolved = _resolve_intraday_outcome(ticker, r)
            if resolved is None:
                continue
            _apply_outcome(r, resolved, RULE_INTRADAY)
            changed = True
            bucket = ("excluded" if resolved["outcome"] in EXCLUDED_OUTCOMES
                      else "resolved")
            summary[bucket].append({
                "ticker": ticker, "outcome": resolved["outcome"],
                "day": resolved["day"], "return_pct": _trade_return_pct(r),
            })

    if changed:
        save_signal_history(history)
    return summary


def _get_or_resolve_outcome(ticker, record):
    """
    If the outcome is already computed and cached in the record — returns it without network
    requests. Otherwise tries to compute it via _resolve_signal_outcome and caches
    the result right in record (mutates in place).
    Returns True if the record was changed (the history needs to be saved to disk).
    """
    if record.get("outcome"):
        return False

    resolved = _resolve_signal_outcome(ticker, record)
    if resolved is None:
        return False

    _apply_outcome(record, resolved, RULE_TWO_SESSIONS)
    return True


def _format_outcome_line(record):
    outcome = record.get("outcome")
    price_str = safe_format_float(record.get("outcome_price"))
    when_label = record.get("outcome_day", "")

    reanchor_note = ""
    if record.get("entry_reanchored"):
        reanchor_note = (
            f"\n      ↳ the signal was out of session: entry recomputed to the open "
            f"(${safe_format_float(record.get('effective_entry_price'))} "
            f"instead of ${safe_format_float(record.get('entry_price'))})"
        )

    if outcome == "stop":
        return f"❌ forecast missed — stop hit {when_label} at ${price_str}{reanchor_note}"
    elif outcome == "target_1":
        return f"✅ forecast hit — target 1 reached {when_label} at ${price_str}{reanchor_note}"
    elif outcome == "target_2":
        return f"✅✅ forecast hit — target 2 reached {when_label} at ${price_str}{reanchor_note}"
    elif outcome == "none":
        return (
            f"🤷 neither target nor stop hit over 2 sessions "
            f"(price at the 2nd session's close: ${price_str}){reanchor_note}"
        )
    elif outcome == "gap_no_edge":
        return (
            f"⚪ no trade — {when_label}: the entry at the open (${price_str}) "
            f"was too close to the target and far from the stop, the entry filter "
            f"does not let such an R:R through{reanchor_note}"
        )
    elif outcome == "gap_invalidated":
        return (
            f"⚪ no trade — {when_label}: the open (${price_str}) was already beyond "
            f"the stop, entering per plan was impossible{reanchor_note}"
        )
    return "⏳ the check window (signal session + next session) has not closed yet — will update later"


def format_history_report(ticker, limit=5):
    """
    Returns text with past signals for the ticker (not including the current one —
    call BEFORE append_signal for the same run).
    For signals older than today it pulls the real hourly candles
    (signal day + next trading day) and says whether the stop/target fired
    and on which day — the result is cached in signal_history.json, so
    yfinance is not called again for already resolved signals.
    An empty string if there is no history yet.
    """
    history = load_signal_history()
    records = history.get(ticker, [])
    if not records:
        return ""

    records_sorted = sorted(
        _dedupe_records(records), key=lambda r: r["timestamp"], reverse=True
    )[:limit]

    lines = [f"📜 *SIGNAL HISTORY — {escape_markdown(ticker)}*"]
    history_changed = False
    for r in records_sorted:
        when, days_ago = _humanize_when(r["timestamp"])
        direction = escape_markdown(r.get("direction", "?"))
        entry = safe_format_float(r.get("entry_price"))
        stop = safe_format_float(r.get("stop_loss"))
        t1 = safe_format_float(r.get("target_1"))
        t2 = safe_format_float(r.get("target_2"))
        confidence = r.get("confidence")
        conf_str = f"{confidence * 100:.0f}%" if confidence is not None else "?"
        repeats = r.get("repeat_count", 1)
        repeat_str = f", confirmed {repeats}× during the session" if repeats > 1 else ""
        lines.append(
            f"• {when} — *{direction}*, entry ${entry}, stop ${stop}, "
            f"targets ${t1}/${t2}, confidence {conf_str}{repeat_str}"
        )

        if days_ago <= 0:
            lines.append("   ⏳ today's signal (plan for the session until the close) — too early to judge")
            continue

        if r.get("direction") not in TRADE_DIRECTIONS:
            continue

        if _get_or_resolve_outcome(ticker, r):
            history_changed = True
        lines.append(f"   {_format_outcome_line(r)}")

    if history_changed:
        # r are the same objects as in history[ticker] (sorted() copies the list
        # but not the elements), so the mutations above are already reflected in history
        save_signal_history(history)

    return "\n".join(lines)


def _trade_return_pct(record):
    """
    % return of the trade: actual entry -> outcome_price (the level at which
    the outcome was recorded — target/stop/2nd-session close price).

    The base is effective_entry_price if present: for out-of-session signals
    it is the open of the applicable session, not the last close at which entering
    was impossible. Otherwise Sharpe and profit factor would systematically be off
    by the size of the gap (see _resolve_signal_outcome).
    """
    # gap_invalidated / gap_no_edge: the entry was impossible (gap past the stop) or
    # pointless (the gap ate R:R below the filter threshold) — see _resolve_signal_outcome.
    # There was effectively no trade, it must not go into return/Sharpe with any sign.
    if (not record.get("outcome")
            or record.get("outcome") in NO_TRADE_OUTCOMES + EXCLUDED_OUTCOMES):
        return None
    entry = record.get("effective_entry_price")
    if entry is None:
        entry = record.get("entry_price")
    outcome_price = record.get("outcome_price")
    if entry is None or outcome_price is None or entry == 0:
        return None
    direction = record.get("direction", "")
    if direction in ("LONG", "STRONG_LONG"):
        return (outcome_price - entry) / entry * 100
    elif direction in ("SHORT", "STRONG_SHORT"):
        return (entry - outcome_price) / entry * 100
    return None


def scoring_version_of(record):
    """Score formula version of a record. A missing field means v1: the tag was introduced
    together with v2, everything older was computed as a flat sum."""
    return record.get("scoring_version") or LEGACY_SCORING_VERSION


def compute_performance_stats(ticker=None, rule=None, scoring=None,
                              since=None, until=None):
    """
    Risk-adjusted statistics on ALREADY RESOLVED signals (with an outcome) —
    for one ticker (if given) or for the entire accumulated history.
    "Sharpe per trade" is mean(returns) / std(returns) over individual trades, WITHOUT
    annualisation (trades are not tied to fixed calendar periods,
    annualising via sqrt(N) would give a false sense of precision).

    rule — count only records resolved by this rule (RULE_INTRADAY
    or RULE_TWO_SESSIONS). They must not be mixed: they have different definitions of a win,
    and the sum of such observations means nothing.

    scoring — the same for the SCORE FORMULA VERSION (SCORING_VERSION /
    LEGACY_SCORING_VERSION). Two independent slices for the same reason: the rule
    defines what counts as a win, the formula version what the score means. A record
    can be old by one and new by the other.

    since/until — a window by SIGNAL date (ISO "YYYY-MM-DD", inclusive on
    both ends). A THIRD, independent slicing axis: it answers not "what the score
    means" and not "what counts as a win" but "how have things been lately" —
    a fundamentally different question, not about the statistical significance of an edge (that needs
    the whole sample, it must not be cut) but about operational monitoring of the mood
    of recent days. So the window does not replace the full report but complements it.
    """
    history = load_signal_history()
    if ticker:
        tagged_records = [(ticker, r) for r in _dedupe_records(history.get(ticker, []))]
    else:
        # Deduplicate within each ticker separately (the same session for
        # different tickers is different observations). We keep the ticker on each
        # record — records in JSON are grouped by ticker key, the record itself
        # does not store it, and for "Best/worst trade" in the summary across ALL tickers
        # we need to know which stock produced that result.
        tagged_records = [
            (tkr, r) for tkr, recs in history.items() for r in _dedupe_records(recs)
        ]

    if rule is not None:
        # Not yet resolved records (resolution_rule missing) belong to the
        # CURRENT rule: that is the one that will resolve them. Without this branch
        # a signal created today is not visible in the statistics at all before the 15:35 resolution
        # — the report showed "Total signals: 0" with a live record in
        # the history. Pending ones do not go into the old series: it will not be the one resolving them.
        tagged_records = [
            (t, r) for t, r in tagged_records
            if r.get("resolution_rule") == rule
            or (rule == RULE_INTRADAY and not r.get("resolution_rule"))
        ]

    if scoring is not None:
        tagged_records = [(t, r) for t, r in tagged_records
                          if scoring_version_of(r) == scoring]

    if since is not None or until is not None:
        # Comparing ISO date strings works correctly lexicographically —
        # the same trick already used in tickers_seen_today.
        tagged_records = [
            (t, r) for t, r in tagged_records
            if (lambda d: (since is None or d >= since)
                and (until is None or d <= until))((r.get("timestamp") or "")[:10])
        ]

    returns = []  # list of (ticker, return_pct) for tracing the best/worst trade
    trades = []   # per-trade detail — for the breakdown by outcome type and the $ simulation
    wins = losses = none_count = reanchored = gap_invalidated = gap_no_edge = 0
    closed_at_1530 = excluded = pending = 0
    for tkr, r in tagged_records:
        outcome = r.get("outcome")
        if not outcome:
            # There is a signal but no result yet — the next resolution will close it.
            # Counted separately so that "resolved: 0" does not look like "nothing
            # is happening".
            pending += 1
        if outcome == "gap_invalidated":
            gap_invalidated += 1
        elif outcome == "gap_no_edge":
            gap_no_edge += 1
        elif outcome in EXCLUDED_OUTCOMES:
            # Not a trade and not a loss: a signal too late or unresolved.
            # Counted separately and shown — silently dropping is not allowed.
            excluded += 1
        ret = _trade_return_pct(r)
        if ret is None:
            continue
        returns.append((tkr, ret))
        trade_date = r.get("session_date") or (r.get("timestamp") or "")[:10] or None
        trades.append({"ticker": tkr, "outcome": outcome, "return_pct": round(ret, 2),
                       "date": trade_date})
        if r.get("entry_reanchored"):
            reanchored += 1
        if outcome in ("target_1", "target_2"):
            wins += 1
        elif outcome == "stop":
            losses += 1
        elif outcome == "close":
            # A close by the rule at 15:30 is a full trade, win or
            # loss is decided by the sign, not by whether a level was reached.
            closed_at_1530 += 1
            if ret > 0:
                wins += 1
            else:
                losses += 1
        elif outcome == "none":
            none_count += 1

    total_signals = len(tagged_records)
    resolved = len(returns)

    if resolved == 0:
        return {"ticker": ticker, "total_signals": total_signals, "resolved": 0,
                "gap_invalidated": gap_invalidated, "gap_no_edge": gap_no_edge,
                "excluded": excluded, "pending": pending, "rule": rule,
                "scoring": scoring, "since": since, "until": until}

    return_values = [ret for _, ret in returns]
    avg_return = statistics.mean(return_values)
    std_return = statistics.stdev(return_values) if resolved > 1 else 0.0
    sharpe_like = (avg_return / std_return) if std_return > 0 else None

    gains = sum(r for r in return_values if r > 0)
    losses_sum = abs(sum(r for r in return_values if r < 0))
    profit_factor = (gains / losses_sum) if losses_sum > 0 else None

    best_ticker, best_return = max(returns, key=lambda x: x[1])
    worst_ticker, worst_return = min(returns, key=lambda x: x[1])

    # Money simulation: $100 per signal, grouped by signal day
    # (not by resolution day — the close at 15:30 is the same day, the difference exists only
    # for stop/target outcomes that fired intraday, and there the date is the same anyway).
    daily_pnl = {}
    for t in trades:
        day = daily_pnl.setdefault(t["date"], {"trades": [], "total_usd": 0.0})
        usd = round(SIMULATED_POSITION_USD * t["return_pct"] / 100, 2)
        day["trades"].append({**t, "usd": usd})
        day["total_usd"] = round(day["total_usd"] + usd, 2)

    return {
        "ticker": ticker,
        "total_signals": total_signals,
        "resolved": resolved,
        "wins": wins,
        "losses": losses,
        "none_count": none_count,
        "closed_at_1530": closed_at_1530,
        "excluded": excluded,
        "pending": pending,
        "rule": rule,
        "scoring": scoring,
        "since": since,
        "until": until,
        "reanchored": reanchored,
        "gap_invalidated": gap_invalidated,
        "gap_no_edge": gap_no_edge,
        "trades": trades,
        "daily_pnl": daily_pnl,
        # The denominator is wins+losses, NOT resolved. The old formula included
        # the "none" outcome (neither target nor stop over 2 sessions), which never
        # went into the numerator and so understated the win rate. Under the daily rule
        # "none" does not occur at all, but the formula must be right for both.
        "win_rate": round(wins / (wins + losses) * 100, 1) if (wins + losses) else None,
        "avg_return_pct": round(avg_return, 2),
        "std_return_pct": round(std_return, 2),
        "sharpe_like": round(sharpe_like, 2) if sharpe_like is not None else None,
        "profit_factor": round(profit_factor, 2) if profit_factor is not None else None,
        "best_trade_pct": round(best_return, 2),
        "best_trade_ticker": best_ticker,
        "worst_trade_pct": round(worst_return, 2),
        "worst_trade_ticker": worst_ticker,
    }


def compute_score_distribution():
    """
    Distribution of the final score for TAKEN and REJECTED signals.

    This is a primary diagnostic of ranking, not of return. If almost all
    taken signals sit at one score, the shortlist does not rank: everything
    that passed the filter looks the same, and sorting candidates by such
    confidence makes no sense.
    """
    taken = collections.Counter()
    rejected = collections.Counter()
    kinds = collections.Counter()

    for records in load_signal_history().values():
        for r in _dedupe_records(records):
            score = r.get("total_score")
            if score is not None:
                taken[score] += 1

    for records in load_no_trade_history().values():
        for r in records:
            score = r.get("total_score")
            if score is not None:
                rejected[score] += 1
            kinds[r.get("no_trade_kind", "?")] += 1

    return {
        "taken": dict(taken),
        "rejected": dict(rejected),
        "kinds": dict(kinds),
        "taken_total": sum(taken.values()),
        "rejected_total": sum(rejected.values()),
    }


def format_score_distribution(dist):
    """Formats the score distribution for Telegram."""
    taken, rejected = dist["taken"], dist["rejected"]
    lines = ["📐 *SCORE SPREAD (ranking diagnostics)*"]

    if not taken and not rejected:
        lines.append(
            "\nEmpty so far. Scores and the control group are written only for "
            "signals created AFTER this change — old records do not have them."
        )
        return "\n".join(lines)

    if taken:
        lines.append(f"\n*Taken* ({dist['taken_total']}):")
        peak = max(taken.values())
        for score in sorted(taken, reverse=True):
            n = taken[score]
            bar = "█" * max(1, round(n / peak * 12))
            lines.append(f"  {score:+d} │ {bar} {n}")
        spread = max(taken) - min(taken)
        if spread <= 1 and dist["taken_total"] >= 10:
            lines.append(
                f"\n⚠️ The spread is only {spread} points — there is effectively no ranking: "
                f"all passing signals are indistinguishable by strength."
            )

    if rejected:
        lines.append(f"\n*Rejected* ({dist['rejected_total']}):")
        peak = max(rejected.values())
        for score in sorted(rejected, reverse=True):
            n = rejected[score]
            bar = "█" * max(1, round(n / peak * 12))
            lines.append(f"  {score:+d} │ {bar} {n}")
        kinds = dist["kinds"]
        if kinds:
            parts = []
            if kinds.get("below_threshold"):
                parts.append(f"not enough points: {kinds['below_threshold']}")
            if kinds.get("entry_filter"):
                parts.append(f"removed by the entry filter: {kinds['entry_filter']}")
            if parts:
                lines.append(f"  ({', '.join(parts)})")
    else:
        lines.append(
            "\n*Rejected:* not accumulated yet — this is the control group, "
            "without it one cannot test whether the threshold distinguishes anything."
        )

    return "\n".join(lines)


def _format_trade_list(trades):
    """
    A detailed list of trades by outcome type — not just a win counter, but showing
    HOW each win was obtained: by actually touching the target or by the sign of the price at the
    forced close at 15:30. These are observations of different strength: the aggregate
    "4 wins" does not tell a trade where the target fired from one that simply
    closed slightly above zero by 15:30 — and the difference directly answers the question
    "are the wins a real signal or close-time noise".
    """
    if not trades:
        return []
    groups = {"target_2": [], "target_1": [], "stop": [],
             "close_win": [], "close_loss": []}
    for t in trades:
        key = "close_win" if t["outcome"] == "close" and t["return_pct"] > 0 \
            else "close_loss" if t["outcome"] == "close" \
            else t["outcome"]
        groups.setdefault(key, []).append(t)

    labels = {
        "target_2": "🎯 Target 2 reached", "target_1": "🎯 Target 1 reached",
        "stop": "❌ Stop", "close_win": "⏹ Closed on time, in profit",
        "close_loss": "⏹ Closed on time, at a loss",
    }
    lines = []
    for key in ("target_2", "target_1", "stop", "close_win", "close_loss"):
        items = groups.get(key)
        if not items:
            continue
        parts = ", ".join(f"{escape_markdown(t['ticker'])} {t['return_pct']:+.2f}%"
                          for t in items)
        lines.append(f"{labels[key]} ({len(items)}): {parts}")
    return lines


def _format_daily_pnl(daily_pnl):
    """
    $100 per signal, by day, with a CUMULATIVE total after each day.
    The number is the same as the average return — just in a form that
    feels more concrete ("+$8 for the day" reads differently from "+1.4% per trade
    on average"). The cumulative column is there so the trend ("the last three days
    in the plus or the minus") is visible at a glance, without a separate request.
    """
    if not daily_pnl:
        return []
    lines = [f"💵 *If every signal had been opened with ${SIMULATED_POSITION_USD}:*"]
    total = 0.0
    for day in sorted(d for d in daily_pnl if d):
        info = daily_pnl[day]
        parts = ", ".join(
            f"{escape_markdown(t['ticker'])} {t['usd']:+.2f}$" for t in info["trades"]
        )
        total += info["total_usd"]
        lines.append(f"  {day}: {parts} → day: {info['total_usd']:+.2f}$ "
                     f"(cumulative: {total:+.2f}$)")
    lines.append(f"  Total for the whole period: {total:+.2f}$")
    return lines


def format_counterfactual_comparison(cf_stats):
    """
    Comparison of real trades with the counterfactual of rejected ones — a separate block,
    not built into compute_performance_stats: it compares TWO DIFFERENT
    sources (signal_history and no_trade_history), not a detail of one.
    """
    below = cf_stats.get("below_threshold", {})
    entry = cf_stats.get("entry_filter", {})
    if not below.get("n") and not entry.get("n"):
        return ""

    lines = ["", "🔬 *COUNTERFACTUAL OF REJECTED (if they had been taken)*"]
    if entry.get("n"):
        lines.append(
            f"Screened out by the entry filter (n={entry['n']}): would have won "
            f"{entry['would_win_rate']}%, average return "
            f"{entry['avg_return_pct']:+.2f}% — a direct answer to whether the "
            f"entry filter saves from losses or cuts off profit"
        )
    if below.get("n"):
        lines.append(
            f"Did not reach the direction threshold (n={below['n']}): in the direction "
            f"of the score they would have won {below['would_win_rate']}%, average "
            f"return {below['avg_return_pct']:+.2f}% — a weaker hypothesis "
            f"(score below threshold), but if this number is noticeably above 50%, "
            f"the scoring distinguishes something even BELOW the official cut-off"
        )
    return "\n".join(lines)


def format_performance_stats(stats):
    """Formats signal performance statistics for Telegram."""
    ticker = stats.get("ticker")
    title = (
        f"📊 *SIGNAL STATISTICS — {escape_markdown(ticker)}*"
        if ticker else "📊 *SIGNAL STATISTICS — ALL TICKERS*"
    )
    lines = [title]

    # The resolution rule goes in the header, not in small print: the same figures under
    # different rules mean different things, and they must not be read without this line.
    rule = stats.get("rule")
    if rule == RULE_INTRADAY:
        lines.append(f"_Daily rule: one session, close at "
                     f"{INTRADAY_CLOSE:%H:%M}, signals after "
                     f"{INTRADAY_CUTOFF:%H:%M} are not counted_")
    elif rule == RULE_TWO_SESSIONS:
        lines.append("_Old rule: the signal session plus the next one_")

    scoring = stats.get("scoring")
    if scoring == SCORING_VERSION:
        lines.append(f"_Score formula {scoring}: momentum (OF+MACD+VWAP) compressed "
                     f"to ±1, threshold ±{LONG_THRESHOLD}, maximum ±{MAX_TOTAL_SCORE}_")
    elif scoring == LEGACY_SCORING_VERSION:
        lines.append(f"_Score formula {scoring}: a flat sum of five factors, "
                     f"threshold ±4, maximum ±10_")

    since, until = stats.get("since"), stats.get("until")
    windowed = since is not None or until is not None
    if windowed:
        if since == until:
            lines.append(f"_Window: {since}_")
        elif until is None:
            lines.append(f"_Window: from {since}_")
        else:
            lines.append(f"_Window: {since} — {until}_")

    total = stats.get("total_signals", 0)
    resolved = stats.get("resolved", 0)
    lines.append(f"Total signals: {total} (resolved: {resolved})")

    # The window answers "how have things been lately", not "is there an edge" —
    # the second question needs the whole sample (see the compute_performance_
    # stats docstring). On a short sample the win rate jumps between 0% and 100% on a single
    # trade, and without this warning a short window is easily read as
    # a diagnosis of the system rather than the weather of a couple of days.
    if windowed and 0 < resolved < MIN_WINDOW_N:
        lines.append(
            f"⚠️ n={resolved} — too little for a conclusion, only for monitoring "
            f"the mood of recent days, not for decisions about the system"
        )

    if resolved == 0:
        skipped = stats.get("gap_invalidated", 0) + stats.get("gap_no_edge", 0)
        if skipped:
            lines.append(
                f"\n⚠️ No resolved signals for statistics yet — {skipped} "
                f"excluded: a gap at the open made the entry impossible or pointless, "
                f"there was effectively no trade."
            )
        elif stats.get("pending"):
            lines.append(
                f"\n⏳ {stats['pending']} signal(s) awaiting resolution — the result "
                f"is computed at {INTRADAY_CLOSE:%H:%M} the same day, "
                f"automatically. Figures will appear after that."
            )
        elif stats.get("rule") == RULE_INTRADAY:
            lines.append(
                "\n⚠️ No closed trades yet. Under the daily rule a signal "
                "is closed at 15:30 the same day — the result will appear after the first "
                "intra-session trade."
            )
        else:
            lines.append(
                "\n⚠️ No resolved signals yet — at least the signal day + "
                "the next trading day are needed for the system to compute a result. Check back later."
            )
        if stats.get("excluded"):
            lines.append(
                f"↳ {stats['excluded']} signals outside the statistics: born after "
                f"the {INTRADAY_CUTOFF:%H:%M} cut-off, out of session, or no data was found"
            )
        return "\n".join(lines)

    lines.append(
        f"✅ Wins: {stats['wins']} | ❌ Losses: {stats['losses']}"
        + (f" | 🤷 Neither: {stats['none_count']}"
           if stats.get("none_count") else "")
    )
    if stats.get("closed_at_1530"):
        lines.append(
            f"↳ of which {stats['closed_at_1530']} were closed by the rule at "
            f"{INTRADAY_CLOSE:%H:%M}, reaching neither the target nor the stop — "
            f"the outcome is decided by the sign of the result, not by a level"
        )
    win_rate = stats.get("win_rate")
    lines.append(f"Win rate: {win_rate}%" if win_rate is not None
                 else "Win rate: no closed trades")

    trade_lines = _format_trade_list(stats.get("trades", []))
    if trade_lines:
        lines.extend(trade_lines)

    if stats.get("excluded"):
        lines.append(
            f"↳ another {stats['excluded']} outside the statistics: after the cut-off "
            f"{INTRADAY_CUTOFF:%H:%M}, out of session or without data — "
            f"neither wins nor losses, not in the denominator"
        )
    if stats.get("pending"):
        lines.append(
            f"⏳ awaiting resolution: {stats['pending']} — result at "
            f"{INTRADAY_CLOSE:%H:%M} the same day"
        )
    if stats.get("reanchored"):
        lines.append(
            f"↳ of which {stats['reanchored']} were generated out of session — "
            f"entry recomputed to the open (otherwise the result would be inflated by the gap)"
        )
    if stats.get("gap_invalidated"):
        lines.append(
            f"↳ another {stats['gap_invalidated']} excluded from the statistics: the gap at the open "
            f"went past the stop before the entry — there was effectively no trade"
        )
    if stats.get("gap_no_edge"):
        lines.append(
            f"↳ another {stats['gap_no_edge']} excluded: a gap in the trade's direction ate R:R below "
            f"the entry filter threshold (R:R < {MIN_ENTRY_RR:g}) — the system would not have taken "
            f"such a trade. Without this they would land in the statistics mostly "
            f"as wins: a gap toward the target also raises the chance of reaching it"
        )
    lines.append("")
    lines.append(f"Average return per trade: {stats['avg_return_pct']:+.2f}%")
    lines.append(f"Standard deviation: {stats['std_return_pct']:.2f}%")

    sharpe = stats.get("sharpe_like")
    if sharpe is not None:
        lines.append(f"📐 Sharpe (per trade, not annualised): {sharpe:.2f}")
        if sharpe > 1:
            lines.append("   → a good risk-adjusted result")
        elif sharpe > 0:
            lines.append("   → positive but weak — the return is close to its spread")
        else:
            lines.append("   → negative — on average the system loses after accounting for trade volatility")
    else:
        lines.append("📐 Sharpe: not enough data (need ≥2 resolved signals with a spread of returns)")

    pf = stats.get("profit_factor")
    if pf is not None:
        lines.append(f"Profit factor: {pf:.2f} (>1 — winning trades outweigh losing ones)")

    # The ticker of the best/worst trade is only useful in the summary across ALL tickers —
    # when requesting one ticker it is already in the report header.
    if ticker:
        lines.append(f"Best trade: {stats['best_trade_pct']:+.2f}% | Worst: {stats['worst_trade_pct']:+.2f}%")
    else:
        best_ticker = escape_markdown(stats.get("best_trade_ticker", "?"))
        worst_ticker = escape_markdown(stats.get("worst_trade_ticker", "?"))
        lines.append(
            f"Best trade: {best_ticker} {stats['best_trade_pct']:+.2f}% | "
            f"Worst: {worst_ticker} {stats['worst_trade_pct']:+.2f}%"
        )

    daily_lines = _format_daily_pnl(stats.get("daily_pnl", {}))
    if daily_lines:
        lines.append("")
        lines.extend(daily_lines)

    return "\n".join(lines)
