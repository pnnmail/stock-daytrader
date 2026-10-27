"""
Filter lab: measures whether the /refilter thresholds give a real
edge — on the sweep archive and real prices.

    python -m agents.filter_lab                  # 5-day horizon
    python -m agents.filter_lab --days 3 --days 10
    python -m agents.filter_lab --list           # which sweeps exist

FOUR METHODOLOGICAL RULES BUILT INTO THE CODE. All four come from
specific mistakes caught in a sibling project, stock_agent3, where each one
on its own flipped the conclusion:

1. COMPARE WITH THE REJECTED, NOT WITH "EVERYONE". The control group is
   LONG candidates that the threshold did NOT let through. Comparing with "everyone" would put
   the selected names into their own control group and understate the difference.

2. BENCHMARK CORRECTION. The return is computed IN EXCESS OF SPY over the same holding
   window. Without this, any set looks good in a rising market:
   in the 08/03 measurement even the names rejected by the filter showed 74% positive.

3. A SIGN TEST ON THE MEDIAN, NOT A t-TEST ON THE MEAN. Stock returns have
   a heavy right tail: the mean is driven by 3-5 observations, the t-statistic
   falls apart. The share that beat the benchmark is insensitive to tails.

4. MOST IMPORTANT: THE UNIT OF OBSERVATION IS THE SWEEP, NOT THE TICKER. 119 tickers of one sweep
   lived through ONE trading day, their returns are tied together by the market almost entirely.
   Treating them as 119 independent observations is the easiest way to get
   a "significant" result out of a single day. So the main line of the report is
   binary: did the set beat the control group ON THIS SWEEP, and the sign
   test runs over the number of sweeps. Ticker-level figures are printed for reference and
   labelled as such.

WHAT TO EXPECT TIME-WISE: a sign test over sweeps needs at least 6-8
sweeps (with 5 out of 5 successes the two-sided p = 0.0625, i.e. even a perfect
result on five sweeps does not reach 0.05). That is weeks of regular
/autoscan, not one evening.
"""
import json
import os
import statistics
import sys
from datetime import datetime, timedelta
from math import comb

import pandas as pd
import yfinance as yf

from agents.screener import rank_long_candidates
from config import SWEEP_ARCHIVE_DIR, FILTER_LAB_PRICE_CACHE

BENCHMARK = "SPY"
DEFAULT_HORIZONS = [5]
# How many top candidates of a set go into the measurement. A set may give 400
# names, but only dozens are actually analysed (/queue 10 at a time) — we have to measure
# the part that is actually looked at.
TOP_N = 60

_QUALITY = {"dollar_volume": 20_000_000, "price": 10.0, "volume_ratio": 1.0}
_MOMENTUM = {"volume_ratio": 1.5, "atr_min": 1.0}

# ═══════════════════════════════════════════════════════════════════════════
# REGIME RULES — switch the threshold set depending on the market state
# ON THE SWEEP DAY. They test exactly one hypothesis: "fitting the filter to the
# regime is better than always keeping the default".
#
# THREE CONSTRAINTS WITHOUT WHICH THE MEASUREMENT WOULD BE FICTION:
#
# 1. NO LOOK-AHEAD. The regime is computed FROM THE SWEEP ITSELF (the share of
#    LONG among scanned names, median ATR) — data that
#    existed at sweep time. It is tempting to take SPY's return over the holding
#    window, but that is exactly the outcome we are measuring.
#
# 2. THE THRESHOLD IS CHOSEN FROM THE REGIME DISTRIBUTION, NOT FROM RETURNS. Measured on
#    41 archived sweeps: breadth 56-87% (median 70.4), median ATR
#    0.83-1.52% (median 1.30). The thresholds below are rounded to 70 and 1.3 so that
#    the sample splits roughly in half — a rule that puts all sweeps into one
#    branch tests nothing. This is the only free knob in
#    the design, and on 37 sweeps a result around p≈0.05 should be considered
#    preliminary.
#
# 3. THERE IS A NEGATIVE CONTROL. "regime: RANDOM" switches between
#    the same two sets, but by the parity of the date — i.e. ignoring the regime.
#    If the meaningful rules don't beat it, what works is not the regime signal
#    but the mere fact of switching.
# ═══════════════════════════════════════════════════════════════════════════
REGIME_BREADTH_SPLIT = 70.0
REGIME_ATR_SPLIT = 1.3


def _branch_label(overrides):
    """Readable name of the branch the regime rule chose."""
    if overrides is _MOMENTUM:
        return "momentum"
    if overrides is _QUALITY:
        return "liquid"
    return "default"


def sweep_regime(results):
    """Regime metrics of a sweep — from the sweep itself only, no future data."""
    if not results:
        return {"breadth": 0.0, "atr": 0.0}
    longs = sum(1 for r in results if r.get("direction") == "LONG")
    return {
        "breadth": longs / len(results) * 100,
        "atr": statistics.median([r.get("atr_pct", 0) or 0 for r in results]),
    }


def regime_by_breadth(results, when):
    """Broad market participation -> momentum; narrow -> move into liquidity."""
    return _MOMENTUM if sweep_regime(results)["breadth"] >= REGIME_BREADTH_SPLIT else _QUALITY


def regime_by_volatility(results, when):
    """High volatility -> liquid large caps; quiet market -> momentum."""
    return _QUALITY if sweep_regime(results)["atr"] >= REGIME_ATR_SPLIT else _MOMENTUM


def regime_random(results, when):
    """
    Negative control: switches between the same sets, but by
    date parity. Deterministic — a repeated run of the measurement must give
    the same result, otherwise comparing against it means nothing.
    """
    seed = when.year * 372 + when.month * 31 + when.day
    return _MOMENTUM if seed % 2 else _QUALITY


# Threshold sets to compare. An empty dict = apply_prefilter defaults.
# The value can be a dict (a static set) OR a function
# (results, when) -> dict (a regime rule, see above).
PRESETS = {
    "default": {},
    "liquid large caps": _QUALITY,
    "moving today": _MOMENTUM,
    "strict R:R": {"rr1": 2.5},
    "narrow combo": {"volume_ratio": 1.5, "dollar_volume": 10_000_000,
                    "price": 5.0, "atr_min": 1.0, "atr_max": 3.0},
    "regime: breadth": regime_by_breadth,
    "regime: volatility": regime_by_volatility,
    "regime: RANDOM (control)": regime_random,
}


# ═══════════════════════════════════════════════════════════════════════════
# CANDIDATE SOURCES (SELECTORS) — a separate concept from PRESETS.
#
# PRESETS answer the question "which THRESHOLDS to apply to a sweep". SELECTORS answer
# another: "where to get candidates from at all". The earnings calendar cannot be a threshold,
# it is not among the sweep fields: it is a feature external to the sweep
# (an event date), hence a separate mechanism — a function that returns a
# list of tickers directly.
#
# THE HYPOTHESIS THIS TESTS: the sweep finds stocks REACTIVELY (by the trace
# of volume that has already happened), while the calendar would let us find them PROACTIVELY —
# before the event. Question: does earnings proximity add anything beyond the usual rank.
#
# A CONTROL IS MANDATORY AND MUST BE SIZE-MATCHED. A set "earnings ≤7
# days" on a sweep gives, say, 12 names instead of 60 — a smaller sample by
# itself changes the spread of the median. So the control takes THE SAME NUMBER of names from
# THE SAME ranked list, but ignoring earnings dates. If the earnings
# set does not beat such a control, what works is not the calendar but narrowing
# the sample. Lesson from the regime measurement: without a matched control one of
# the sets will almost certainly come out "significant" by chance.
# ═══════════════════════════════════════════════════════════════════════════
def _ranked_tickers(results):
    ranked, _ = rank_long_candidates(results)
    return [c["ticker"] for c in ranked]


# The holding horizon is 5 TRADING days, roughly 7 calendar days. Hence
# the boundary of 8: with earnings 8+ days away the position closes BEFORE the event.
_HOLD_CALENDAR_DAYS = 7


def _select_before_earnings(results, when):
    """
    BEFORE earnings, without holding through them: the event is 8-20 days away, i.e. the holding
    window ends before the announcement.

    Tests the ATTENTION drift (volume and interest grow as earnings
    approach), NOT the outcome of the report. Mixing this with holding through the event
    is not allowed: there the binary announcement risk dominates, and the overall result
    would come out as a murky zero from two different mechanisms.
    """
    from agents.earnings_calendar import days_to_earnings, load_cache

    cache = load_cache()
    if not cache:
        return []
    out = []
    for ticker in _ranked_tickers(results):
        d = days_to_earnings(ticker, when, cache=cache)
        if d is not None and _HOLD_CALENDAR_DAYS < d <= 20:
            out.append(ticker)
        if len(out) >= TOP_N:
            break
    return out


def _select_pead(results, when):
    """
    AFTER a report with a positive surprise (PEAD, post-earnings announcement
    drift) — the report came out no more than 3 days ago and beat expectations.

    Unlike the calendar rules this is a DIRECTIONAL hypothesis: it selects
    specifically for LONG, because the sign of the surprise sets the side. The surprise is
    a fact (Reported EPS vs EPS Estimate), not an interpretation.
    """
    from agents.earnings_calendar import last_earnings, load_cache

    cache = load_cache()
    if not cache:
        return []
    out = []
    for ticker in _ranked_tickers(results):
        days_since, surprise = last_earnings(ticker, when, cache=cache)
        if days_since is not None and days_since <= 3 and (surprise or 0) > 0:
            out.append(ticker)
        if len(out) >= TOP_N:
            break
    return out


def _matched_control(results, picked, pool=None):
    """
    A size-matched control: the same number of names from THE SAME pool
    the selector itself used, but WITHOUT its signal.

    Names are taken at an even step through the list, not from the top: a sparse
    sample across the whole ranking, otherwise a top-N control would already be better
    ranked by default, i.e. it would compare the quality of the rank
    rather than the signal itself.

    pool — which population to draw from. By default `_ranked_tickers(results)`
    (the pool AFTER the default apply_prefilter — fine for the calendar
    selectors below, they were designed as an add-on to the usual
    ranking). For signals that themselves claim to replace the apply_prefilter
    thresholds (see `_select_reversal`/`_select_anti_max`), pool must be
    passed explicitly — the same RAW population (LONG+entry_ok, BEFORE
    apply_prefilter) the selector itself sees. Otherwise the comparison is unfair:
    a signal against someone else's population already filtered by another rule.
    """
    if not picked:
        return []
    if pool is None:
        pool = _ranked_tickers(results)
    step = max(1, len(pool) // len(picked))
    return pool[::step][:len(picked)]


# ═══════════════════════════════════════════════════════════════════════════
# SIGNALS AGAINST THE LITERATURE — added 2026-08-05 after a review.
#
# HYPOTHESIS: apply_prefilter selects by volume_ratio/ATR — i.e. it literally
# looks for stocks with a recent burst of activity. Three independent lines
# of academic literature call this a predictor of LOWER, not
# higher, future returns:
#   - short-term reversal (Lehmann 1990, Jegadeesh 1990, Bali et al. 2016):
#     high return over the past week/month -> negative excess
#     return in the next period; the effect is stronger for high-turnover
#     stocks (Conrad et al., 1994) — exactly our case.
#   - MAX effect (Bali, Cakici, Whitelaw, 2011): an extreme "jackpot" day
#     in the last month -> lower future returns, >1%/month,
#     robust to controlling for liquidity and short-term reversal itself.
#
# The ret_5d/max_ret_20d fields are NOT computed in screen_from_batch (that would change
# the live /autoscan), but separately in agents/replay_lab.py::compute_research_signals
# — for the replay only. Live sweeps (see save_raw_sweep_results) do not have these
# fields, so the selectors below silently return [] on live data —
# see SELECTORS: an empty list = "the source gave nothing on this sweep",
# the same contract the earnings selectors already use.
#
# THE CONTROL comes from THE SAME raw population (LONG+entry_ok, BEFORE apply_prefilter),
# not from _ranked_tickers (that one has already passed the DEFAULT apply_prefilter). Otherwise
# the comparison would be unfair: a new signal against someone else's population filtered by
# another rule. See the _matched_control docstring.
# ═══════════════════════════════════════════════════════════════════════════
def _long_entry_pool(results):
    """LONG candidates that passed the entry filter, BEFORE apply_prefilter — the same
    starting point as all PRESETS (they just cut this pool differently)."""
    return [r for r in results if r.get("direction") == "LONG" and r.get("entry_ok")]


def _select_reversal(results, when):
    """Bottom tail of ret_5d (recent laggards) among LONG+entry_ok — a bet
    on short-term reversal instead of chasing momentum."""
    pool = [r for r in _long_entry_pool(results) if r.get("ret_5d") is not None]
    pool.sort(key=lambda r: r["ret_5d"])
    return [r["ticker"] for r in pool[:TOP_N]]


def _select_anti_max(results, when):
    """Bottom tail of max_ret_20d (no recent "jackpot" day) among
    LONG+entry_ok — the opposite of what volume_ratio implicitly rewards."""
    pool = [r for r in _long_entry_pool(results) if r.get("max_ret_20d") is not None]
    pool.sort(key=lambda r: r["max_ret_20d"])
    return [r["ticker"] for r in pool[:TOP_N]]


def _signal_control_pool(results, field):
    """The same raw population the signal selector sees (LONG+entry_ok
    with a non-empty field), sorted by gap — a deterministic order
    for even sampling in _matched_control, without the signal."""
    pool = [r for r in _long_entry_pool(results) if r.get(field) is not None]
    pool.sort(key=lambda r: r.get("gap", 0))
    return [r["ticker"] for r in pool]


# ═══════════════════════════════════════════════════════════════════════════
# EXCLUSION BY ret_5d — added 2026-08-05 after the first measurement.
#
# _select_reversal (above) SELECTED the bottom tail of ret_5d, betting on the classic
# short-term reversal. Measured over 4 horizons (5/10/20/30 d) it gave the opposite
# sign, and it grew stronger with the horizon:
#   10d: 9/38 above control, p=0.0017, median -3.15%
#   20d: 9/36 above control, p=0.0039, median -4.11%
#   30d: 6/34 above control, p=0.0002, median -7.50%
# Inside the already selected LONG pool, recent laggards do not bounce but
# keep falling — the strongest and most consistent result of the whole project,
# just with the opposite practical use: not a BUY SIGNAL
# but a READY-MADE ANTI-FILTER. We test here: if the bottom tail of ret_5d is dropped
# FROM the existing presets (rather than building a separate selection from it),
# do default/liquid large caps get better?
#
# WHY A PERCENTILE AND NOT A FIXED NUMBER OF TICKERS. _select_reversal took
# the 60 worst FROM THE WHOLE LONG+entry_ok pool (which can be 200 or 1500+
# names) — i.e. effectively the bottom ~5-10% on average. RET5D_EXCLUDE_PCT=15
# is a round number of the same order, NOT tuned to the result
# (otherwise it would be fitting the threshold to an answer already seen).
#
# THE COMPARISON IS DIRECTLY WITH THE PARENT PRESET; no separate matched
# control is set up: the "default" row is already in the main table and already
# compared with the standard control; "default, excl. ret5d<15%" is the same
# comparison point, read both p/win rates side by side. A dedicated matched control
# here is redundant and would just add rows without new information.
# ═══════════════════════════════════════════════════════════════════════════
RET5D_EXCLUDE_PCT = 15


def _exclude_ret5d_bottom(results, pct=RET5D_EXCLUDE_PCT):
    """
    results WITHOUT the bottom pct% by ret_5d among LONG+entry_ok with a known field.
    Candidates without ret_5d (SHORT, live sweeps without research fields)
    stay untouched — the filter only applies where the signal exists.
    """
    pool = [r for r in _long_entry_pool(results) if r.get("ret_5d") is not None]
    if len(pool) < 10:
        return results  # too little data for a percentile — don't filter at all
    vals = sorted(r["ret_5d"] for r in pool)
    cutoff = vals[int(len(vals) * pct / 100)]
    excluded = {r["ticker"] for r in pool if r["ret_5d"] < cutoff}
    return [r for r in results if r["ticker"] not in excluded]


def _select_default_excl_ret5d(results, when):
    ranked, _ = rank_long_candidates(_exclude_ret5d_bottom(results))
    return [c["ticker"] for c in ranked[:TOP_N]]


def _select_quality_excl_ret5d(results, when):
    ranked, _ = rank_long_candidates(_exclude_ret5d_bottom(results), overrides=_QUALITY)
    return [c["ticker"] for c in ranked[:TOP_N]]


# Four hypotheses (plus the exclusion filter), EACH hypothesis has its own
# size-matched control (except the exclusion one — see the explanation
# above). The third obvious hypothesis ("hold through the announcement") is deliberately
# not tested: holding through a binary event is not something worth
# recommending whatever the measurement says, so its decision value is
# zero, while it adds table rows and multiple comparisons.
SELECTORS = {
    "before earnings (8-20 d out)": _select_before_earnings,
    "  └ control for 'before earnings'": lambda r, w: _matched_control(r, _select_before_earnings(r, w)),
    "after earnings, surprise>0": _select_pead,
    "  └ control for 'after earnings'": lambda r, w: _matched_control(r, _select_pead(r, w)),
    "reversal (low ret_5d)": _select_reversal,
    "  └ control for 'reversal'": lambda r, w: _matched_control(
        r, _select_reversal(r, w), pool=_signal_control_pool(r, "ret_5d")),
    "anti-MAX (low max_ret_20d)": _select_anti_max,
    "  └ control for 'anti-MAX'": lambda r, w: _matched_control(
        r, _select_anti_max(r, w), pool=_signal_control_pool(r, "max_ret_20d")),
    "default, excl. ret5d<15%": _select_default_excl_ret5d,
    "liquid large caps, excl. ret5d<15%": _select_quality_excl_ret5d,
}


def all_set_names():
    """All compared sets — thresholds and sources — in output order."""
    return list(PRESETS) + list(SELECTORS)


def binom_two_sided(k, n, p=0.5):
    """Exact two-sided binomial test without scipy."""
    if n == 0:
        return float("nan")
    probs = [comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    return float(min(1.0, sum(pr for pr in probs if pr <= probs[k] * (1 + 1e-9))))


def list_sweeps():
    """Archived sweeps, oldest to newest: [(path, datetime, results)]."""
    if not os.path.isdir(SWEEP_ARCHIVE_DIR):
        return []
    out = []
    for name in sorted(os.listdir(SWEEP_ARCHIVE_DIR)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(SWEEP_ARCHIVE_DIR, name)
        try:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
            when = datetime.fromisoformat(doc["generated_at"])
            out.append((path, when, doc.get("results", [])))
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            print(f"   ⚠️ Skipping {name}: {str(exc)[:80]}")
    return out


def _load_cache():
    try:
        with open(FILTER_LAB_PRICE_CACHE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(cache):
    try:
        with open(FILTER_LAB_PRICE_CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except OSError as exc:
        print(f"   ⚠️ Price cache was not written: {str(exc)[:80]}")


def fetch_closes(tickers, start, end, cache):
    """
    {ticker: {date: close}} for the window. Disk cache — repeated runs of the measurement
    (and there will be many, thresholds are tuned iteratively) must not
    re-download hundreds of tickers every time and hit Yahoo's rate limit.
    """
    key_range = f"{start:%Y-%m-%d}_{end:%Y-%m-%d}"
    need = [t for t in tickers if f"{t}@{key_range}" not in cache]
    if need:
        print(f"   Downloading {len(need)} tickers for {key_range}...")
        for i in range(0, len(need), 200):
            chunk = need[i:i + 200]
            try:
                df = yf.download(chunk, start=start.strftime("%Y-%m-%d"),
                                 end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
                                 interval="1d", progress=False, auto_adjust=False,
                                 threads=False, group_by="ticker")
            except Exception as exc:
                print(f"   ⚠️ Batch failed to download: {str(exc)[:100]}")
                continue
            for t in chunk:
                series = {}
                try:
                    sub = (df[t] if isinstance(df.columns, pd.MultiIndex)
                           else df)
                    closes = sub["Close"].dropna()
                    series = {d.strftime("%Y-%m-%d"): float(v)
                              for d, v in closes.items()}
                except Exception:
                    pass
                cache[f"{t}@{key_range}"] = series
        _save_cache(cache)
    return {t: cache.get(f"{t}@{key_range}", {}) for t in tickers}


def _return_pct(series, base_date, horizon_days):
    """
    Return from the last close NO LATER than base_date to the close
    horizon_days trading days later. None if there is not enough data — including
    when the horizon has not arrived yet (the sweep is fresher than the horizon).
    """
    if not series:
        return None
    dates = sorted(series)
    base = [d for d in dates if d <= base_date]
    if not base:
        return None
    b_idx = dates.index(base[-1])
    t_idx = b_idx + horizon_days
    if t_idx >= len(dates):
        return None
    b, t = series[dates[b_idx]], series[dates[t_idx]]
    if b <= 0:
        return None
    return (t / b - 1) * 100


def measure_sweep(results, when, horizon, cache):
    """One sweep: for each set — excess return vs. SPY."""
    base_date = when.strftime("%Y-%m-%d")

    selections, universe = {}, set()
    for name, spec in PRESETS.items():
        # A regime rule is a function of the sweep; a static set is just a dict.
        overrides = spec(results, when) if callable(spec) else spec
        ranked, _ = rank_long_candidates(results, overrides=overrides or None)
        picked = [c["ticker"] for c in ranked[:TOP_N]]
        selections[name] = picked
        universe.update(picked)

    # Candidate sources (see SELECTORS): return a list of tickers directly,
    # not thresholds. An empty list = the source gave nothing on this sweep
    # (e.g. the earnings cache is not filled yet) — such a set simply
    # does not take part in this sweep.
    for name, fn in SELECTORS.items():
        picked = fn(results, when)
        if picked:
            selections[name] = picked
            universe.update(picked)

    # Control: LONG candidates that passed the ENTRY filter but did NOT pass
    # the default set's thresholds. Exactly what the filter must differ from.
    passed = set(selections["default"])
    rejected = [r["ticker"] for r in results
                if r.get("direction") == "LONG" and r.get("entry_ok")
                and r["ticker"] not in passed][:TOP_N]
    selections["CONTROL (rejected)"] = rejected
    universe.update(rejected)

    closes = fetch_closes(sorted(universe) + [BENCHMARK],
                          when - timedelta(days=10),
                          when + timedelta(days=horizon * 3 + 14), cache)
    bench = _return_pct(closes.get(BENCHMARK, {}), base_date, horizon)
    if bench is None:
        return None, None

    out = {}
    for name, tickers in selections.items():
        vals = []
        for t in tickers:
            r = _return_pct(closes.get(t, {}), base_date, horizon)
            if r is not None:
                vals.append(r - bench)
        if len(vals) >= 5:
            s = pd.Series(vals)
            out[name] = {"n": len(vals), "median": float(s.median()),
                         "mean": float(s.mean()),
                         "beat": float((s > 0).mean() * 100)}
    return out, bench


def report(horizon):
    """
    Report for one horizon. Returns TEXT rather than printing — the same
    report is needed both in the console and in Telegram (/lab), and duplicating the construction
    in two places means one day fixing only one of them.
    """
    L = []
    add = L.append
    sweeps = list_sweeps()
    if not sweeps:
        return (f"No archived sweeps in {SWEEP_ARCHIVE_DIR}.\n"
                f"The archive fills automatically with every /autoscan.")

    cache = _load_cache()
    add(f"HORIZON {horizon} trading days")
    add("=" * 62)

    per_sweep, benches, regime_log = [], [], []
    for path, when, results in sweeps:
        res, bench = measure_sweep(results, when, horizon, cache)
        if not res:
            add(f"  {when:%Y-%m-%d} — horizon not reached yet, skipping")
            continue
        per_sweep.append(res)
        benches.append(bench)
        # Which branch each regime rule chose on this sweep — without
        # this, a rule that always goes to one branch would silently look
        # like an ordinary static set, and the difference would go unnoticed.
        regime_log.append({
            name: _branch_label(spec(results, when))
            for name, spec in PRESETS.items() if callable(spec)
        })
        kind = "synth" if os.path.basename(path).startswith("synthetic_") else "live"
        ctrl = res.get("CONTROL (rejected)", {}).get("median")
        add("")
        add(f"  Sweep {when:%Y-%m-%d} [{kind}]  (SPY {bench:+.2f}%)")
        for name, m in res.items():
            mark = ""
            if ctrl is not None and name != "CONTROL (rejected)":
                mark = "  ✔ above control" if m["median"] > ctrl else "  ✘ below"
            add(f"    {name:<24} n={m['n']:<4} exc. {m['median']:>+6.2f}%  "
                f"beat SPY {m['beat']:>5.1f}%{mark}")

    if not per_sweep:
        add("")
        add("No sweep has reached the horizon yet.")
        return "\n".join(L)

    add("")
    add("-" * 62)
    add(f"MAIN TABLE — unit of observation is the SWEEP ({len(per_sweep)} of them)")
    add("-" * 62)
    add(f"{'set':<32}{'above control':<16}{'sign test':<14}{'med. of medians'}")
    for name in all_set_names():
        wins, medians = 0, []
        for res in per_sweep:
            if name not in res or "CONTROL (rejected)" not in res:
                continue
            medians.append(res[name]["median"])
            wins += res[name]["median"] > res["CONTROL (rejected)"]["median"]
        n = len(medians)
        if not n:
            continue
        add(f"{name:<26}{f'{wins}/{n}':<16}p={binom_two_sided(wins, n):<12.4f}"
            f"{pd.Series(medians).median():>+7.2f}%")

    if regime_log:
        add("")
        add("-" * 62)
        add("HOW THE REGIME RULES SPLIT THE SWEEPS")
        add("-" * 62)
        for name in (n for n, s in PRESETS.items() if callable(s)):
            counts = {}
            for entry in regime_log:
                counts[entry[name]] = counts.get(entry[name], 0) + 1
            split = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
            warn = "  ⚠️ only one branch — the rule switches nothing" if len(counts) < 2 else ""
            add(f"  {name:<28}{split}{warn}")

    add("")
    add("  p<0.05 needs at least 6 sweeps (5 of 5 gives p=0.0625).")
    add(f"  Sweeps with data now: {len(per_sweep)}"
        + (" — TOO EARLY FOR CONCLUSIONS." if len(per_sweep) < 6 else ""))

    # THE DECISIVE SPLIT. The filter selects by volume and momentum, i.e. it pulls toward
    # high-beta names. In a rising market they will outrun anything without any
    # selection. The only way to tell selection from leverage is to see whether
    # the edge survives on days when the market FELL.
    up_idx = [i for i, b in enumerate(benches) if b > 0]
    dn_idx = [i for i, b in enumerate(benches) if b <= 0]
    if up_idx and dn_idx:
        add("")
        add("-" * 62)
        add(f"SPLIT BY MARKET — selection or just beta?")
        add(f"  market rose: {len(up_idx)} sweeps, fell: {len(dn_idx)}")
        add("-" * 62)
        add(f"{'set':<26}{'on up days':<16}{'on down days':<16}{'verdict'}")
        for name in all_set_names():
            cells = []
            for idxs in (up_idx, dn_idx):
                w = n = 0
                for i in idxs:
                    r = per_sweep[i]
                    if name not in r or "CONTROL (rejected)" not in r:
                        continue
                    n += 1
                    w += r[name]["median"] > r["CONTROL (rejected)"]["median"]
                cells.append((w, n))
            (uw, un), (dw, dn_) = cells
            if not un or not dn_:
                continue
            # An edge only on up days is beta. If it holds on down days too,
            # it starts to look like selection.
            verdict = ("looks like selection" if dw / dn_ > 0.5
                       else "looks like beta" if uw / un > 0.5
                       else "no edge")
            add(f"{name:<26}{f'{uw}/{un}':<16}{f'{dw}/{dn_}':<16}{verdict}")
        add("")
        add("  How to read: a set that wins on up days and loses on")
        add("  down days is not selecting stocks — it is just holding more risk.")
    elif len(per_sweep) >= 3:
        add("")
        add("  ⚠️ All sweeps fell into one market regime "
            f"({'up' if up_idx else 'down'}) — telling selection from beta")
        add("     on such data is NOT possible, whatever the p above.")
    add("")
    add("  For reference, by ticker (NOT independent observations: all tickers")
    add("  of one sweep lived through the same day):")
    for name in all_set_names() + ["CONTROL (rejected)"]:
        vals = [r[name]["median"] for r in per_sweep if name in r]
        if vals:
            add(f"    {name:<26} {pd.Series(vals).median():+.2f}%")
    return "\n".join(L)


def main():
    args = sys.argv[1:]
    if "--list" in args:
        sweeps = list_sweeps()
        print(f"Archived sweeps: {len(sweeps)} in {SWEEP_ARCHIVE_DIR}")
        for path, when, results in sweeps:
            print(f"  {when:%Y-%m-%d %H:%M}  {len(results):>5} tickers  "
                  f"{os.path.basename(path)}")
        return 0

    horizons = [int(args[i + 1]) for i, a in enumerate(args) if a == "--days"] \
        or DEFAULT_HORIZONS
    for h in horizons:
        print("\n" + report(h))
    return 0


if __name__ == "__main__":
    sys.exit(main())
