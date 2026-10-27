# -*- coding: utf-8 -*-
"""
FIVE TARGETED CHECKS on top of the existing historical_backtest.json — they do NOT
repeat the expensive run (price download + STUMPY match across 25 tickers),
they only extend it with data that was not there yet. A one-off tool,
the result is reports/historical_theories.json (also a "historical data
check", not the bot's real statistics — see the general disclaimer below).

    python -m agents.historical_theories

TEN TASKS (1-5 — first wave, 6-10 — second, prompted by "what did we
miss as a financial analyst"):
1. Old (two-step) STUMPY vs. the new one (flat2, see utils/signals.py
   SCORING_VERSION v4_stumpy_flat2) — on an INDEPENDENT dataset (not the one
   the decision was made on). The data is already in records[].raw — just
   a recompute, no network.
2. FINRA short volume IN HINDSIGHT — the same utils/short_volume.py, but
   with an as_of parameter set to each record's date instead of "now". Tests
   whether a high short share predicts a further decline — the only one
   of the new diagnostics (real_flow/short_volume) that can be
   tested on history at all (Alpaca only gives current trades).
3. Validity of the entry filters (ATR% and volume_ratio) from
   utils.signals.check_entry_filters — do they actually cut the worst
   cases, or cut at random.
4. Sensitivity to the outcome horizon (1/5/10 days instead of a fixed 3).
5. Does the formula work differently depending on market cap/sector.
6. EXPECTANCY (size, not just frequency). Everything above measured ONLY the win rate
   — while profitability depends on the ratio of win size to
   loss size. You can call the direction right 50% of the time and be profitable
   or unprofitable at the same 50% — these are different things, and we had not
   distinguished them until now. No network — recomputed from the collected outcome_return_pct.
7. MULTIPLE-COMPARISONS CORRECTION. We ran ~20 different slices on
   more or less one dataset — with that many tests some "interesting"
   numbers inevitably surface from pure noise. No network — a meta-analysis of the already
   computed results.
8. EXCESS RETURN RELATIVE TO THE MARKET (SPY). The base rate wandered from
   42% to 49% between samples — suspiciously like beta to the market rather than
   selection skill. One downloaded ticker (SPY), all records recomputed.
9. OPENING GAP. Specific to day trading: does a gap continue
   (gap-and-go) or close (gap-fill) over the next days — never tested
   at all. 25 downloads (per ticker), using Open, which was not in the
   original records.
10. EARNINGS PROXIMITY. The live bot hard-blocks entry when earnings are
    ≤3 days away (EARNINGS_BLOCK_DAYS) — justified by risk asymmetry, but
    NEVER tested on history directly. 25 earnings-date requests.

Same disclaimer as historical_backtest.py: daily bars only, no
VWAP or news, close-to-close outcome, not a survivorship-free retro universe. See
the agents/historical_backtest.py docstring for the full text.
"""
import json
from datetime import datetime, timedelta

import yfinance as yf

from config import HISTORICAL_BACKTEST_FILE, HISTORICAL_THEORIES_FILE
from utils.short_volume import get_short_volume
from agents.historical_backtest_llm import _fetch_firmographics, FAMOUS_TICKERS

# Same thresholds as utils.signals.check_entry_filters — kept as numbers
# here rather than imported, because there they are not module constants
# (they are baked into the function body); the duplication is minimal (2 numbers) and explicitly
# commented so it does not silently drift apart.
ENTRY_ATR_MIN_PCT = 0.2
ENTRY_ATR_MAX_PCT = 8.0
ENTRY_VOLUME_RATIO_MIN = 0.4

HORIZONS = [1, 5, 10]  # 3 days are already in the original records


def wilson(wins, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)
    return (round((center - margin) / denom * 100, 1), round((center + margin) / denom * 100, 1))


def _hit_rate(pairs):
    """pairs: list of (predicted_sign, actual_return_pct). predicted_sign
    in {1,-1}. Returns (n, hit_rate_pct, wilson95)."""
    n = len(pairs)
    if n == 0:
        return {"n": 0, "hit_rate_pct": None, "wilson95_pct": None}
    wins = sum(1 for pred, ret in pairs if (pred > 0) == (ret > 0))
    lo, hi = wilson(wins, n)
    return {"n": n, "hit_rate_pct": round(wins / n * 100, 1), "wilson95_pct": [lo, hi]}


# ═══════════════════════════════════════════════════════════════════════
# TASK 1: old STUMPY vs. new (flat2) on an independent dataset
# ═══════════════════════════════════════════════════════════════════════
def old_stumpy_score(win_rate, avg_return, n_samples):
    if n_samples is None or n_samples < 5 or win_rate is None or avg_return is None:
        return 0
    if win_rate >= 65 and avg_return > 0.5:
        raw = 2
    elif win_rate >= 55 and avg_return > 0:
        raw = 1
    elif win_rate <= 35 and avg_return < -0.5:
        raw = -2
    elif win_rate <= 45 and avg_return < 0:
        raw = -1
    else:
        raw = 0
    return round(raw * min(1.0, n_samples / 15))


def new_stumpy_score(win_rate, avg_return, n_samples):
    if n_samples is None or n_samples < 5 or win_rate is None or avg_return is None:
        return 0
    if win_rate >= 55 and avg_return > 0:
        raw = 2
    elif win_rate <= 45 and avg_return < 0:
        raw = -2
    else:
        raw = 0
    return round(raw * min(1.0, n_samples / 15))


def task1_stumpy_comparison(records):
    old_pairs, new_pairs = [], []
    old_mag = {1: [], 2: []}
    new_mag = {1: [], 2: []}
    old_ge3 = new_ge3 = 0
    old_taken_pairs, new_taken_pairs = [], []

    for r in records:
        raw = r["raw"]
        wr, ar, ns = raw.get("stumpy_win_rate"), raw.get("stumpy_avg_return"), raw.get("stumpy_n")
        old_s = old_stumpy_score(wr, ar, ns)
        new_s = new_stumpy_score(wr, ar, ns)
        ret = r["outcome_return_pct"]

        if old_s != 0:
            old_pairs.append((old_s, ret))
            old_mag[abs(old_s)].append((old_s, ret))
        if new_s != 0:
            new_pairs.append((new_s, ret))
            new_mag[abs(new_s)].append((new_s, ret))

        # Full formula: momentum (from scores, no longer contains orderflow) +
        # technical + news(=0) + stumpy(old/new) — total is recomputed, not
        # taken from records[].total_score (that one is already computed with the NEW stumpy).
        base = r["scores"]["momentum"] + r["scores"]["technical"] + r["scores"]["news"]
        old_total = base + old_s
        new_total = base + new_s
        if abs(old_total) >= 3:
            old_ge3 += 1
            old_taken_pairs.append((1 if old_total > 0 else -1, ret))
        if abs(new_total) >= 3:
            new_ge3 += 1
            new_taken_pairs.append((1 if new_total > 0 else -1, ret))

    return {
        "note": ("Recomputed on historical_backtest.json (2350 observations) — "
                "a dataset the decision was NOT made on (that was the bot's live "
                "history, n=256)."),
        "stumpy_alone": {
            "old_formula": _hit_rate(old_pairs),
            "new_formula": _hit_rate(new_pairs),
            "old_by_magnitude": {str(k): _hit_rate(v) for k, v in old_mag.items()},
            "new_by_magnitude": {str(k): _hit_rate(v) for k, v in new_mag.items()},
        },
        "full_formula_taken_signals": {
            "old_stumpy_ge3_share_pct": round(old_ge3 / len(records) * 100, 1),
            "new_stumpy_ge3_share_pct": round(new_ge3 / len(records) * 100, 1),
            "old_formula_hit_rate": _hit_rate(old_taken_pairs),
            "new_formula_hit_rate": _hit_rate(new_taken_pairs),
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# TASK 2: FINRA short volume in hindsight
# ═══════════════════════════════════════════════════════════════════════
def task2_short_volume(records):
    pairs_high, pairs_low = [], []
    correl_pairs = []
    fetched, missing = 0, 0

    for r in records:
        try:
            as_of = datetime.strptime(r["date"], "%Y-%m-%d")
        except ValueError:
            continue
        sv = get_short_volume(r["ticker"], as_of=as_of)
        ratio = sv.get("short_ratio") if sv else None
        if ratio is None:
            missing += 1
            continue
        fetched += 1
        ret = r["outcome_return_pct"]
        correl_pairs.append((ratio, ret))
        # Hypothesis: high short share -> further decline (bearish signal),
        # i.e. "predicted_sign" = -1 at a high ratio.
        if ratio >= 0.5:
            pairs_high.append((-1, ret))
        else:
            pairs_low.append((-1, ret))  # same hypothesis sign for a fair comparison of shares

    # Pearson correlation without numpy/scipy — by hand, as before in the project.
    n = len(correl_pairs)
    corr = None
    if n >= 10:
        xs = [p[0] for p in correl_pairs]
        ys = [p[1] for p in correl_pairs]
        mx, my = sum(xs) / n, sum(ys) / n
        cov = sum((x - mx) * (y - my) for x, y in correl_pairs)
        varx = sum((x - mx) ** 2 for x in xs)
        vary = sum((y - my) ** 2 for y in ys)
        if varx > 0 and vary > 0:
            corr = round(cov / (varx * vary) ** 0.5, 3)

    return {
        "note": ("Hypothesis: a high short share (>=50% of daily volume) "
                "predicts a further price decline. Below is the share of down "
                "days in each group, NOT a 'hypothesis win' in the sense of "
                "a directional trade (short volume itself is not part of the bot's "
                "LONG/SHORT entry)."),
        "n_fetched": fetched, "n_missing": missing,
        "pearson_r_ratio_vs_return": corr,
        "high_short_ratio_group": {
            "n": len(pairs_high),
            "share_falling_pct": round(sum(1 for _, ret in pairs_high if ret < 0) / len(pairs_high) * 100, 1) if pairs_high else None,
        },
        "low_short_ratio_group": {
            "n": len(pairs_low),
            "share_falling_pct": round(sum(1 for _, ret in pairs_low if ret < 0) / len(pairs_low) * 100, 1) if pairs_low else None,
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# TASK 3: validity of entry filters (ATR%, volume_ratio)
# ═══════════════════════════════════════════════════════════════════════
def task3_entry_filters(records):
    taken = [r for r in records if r["direction"] != "NO_TRADE"]

    def would_pass(r):
        atr = r["raw"].get("atr_pct")
        vol = r["raw"].get("volume_ratio")
        if atr is None or vol is None:
            return None
        return (ENTRY_ATR_MIN_PCT <= atr <= ENTRY_ATR_MAX_PCT) and (vol >= ENTRY_VOLUME_RATIO_MIN)

    def hyp_pair(r):
        sign = 1 if r["direction"] in ("LONG", "STRONG_LONG") else -1
        return (sign, r["outcome_return_pct"])

    passed = [hyp_pair(r) for r in taken if would_pass(r) is True]
    failed = [hyp_pair(r) for r in taken if would_pass(r) is False]

    return {
        "note": ("Only among 'taken' (LONG/SHORT) observations — checks "
                "whether what the filter WOULD HAVE CUT is really worse than what "
                "passed. The RR filter is not tested: entry/stop levels are not "
                "computed in this backtest (no calculate_entry_levels)."),
        "would_pass_filter": _hit_rate(passed),
        "would_be_rejected": _hit_rate(failed),
    }


# ═══════════════════════════════════════════════════════════════════════
# TASK 4: sensitivity to the outcome horizon
# ═══════════════════════════════════════════════════════════════════════
def task4_horizons(records):
    by_ticker = {}
    for r in records:
        by_ticker.setdefault(r["ticker"], []).append(r)

    results_by_horizon = {h: [] for h in HORIZONS}

    for ticker, recs in by_ticker.items():
        df = yf.download(ticker, period="3y", interval="1d",
                         auto_adjust=False, progress=False)
        if df is None or df.empty:
            continue
        if hasattr(df.columns, "get_level_values"):
            import pandas as pd
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
        close = df["Close"].values.astype(float)
        dates = [d.strftime("%Y-%m-%d") for d in df.index]
        date_to_idx = {d: i for i, d in enumerate(dates)}

        for r in recs:
            idx = date_to_idx.get(r["date"])
            if idx is None:
                continue
            sign = 1 if r["direction"] in ("LONG", "STRONG_LONG") else (
                   -1 if r["direction"] in ("SHORT", "STRONG_SHORT") else
                   (1 if r["total_score"] >= 0 else -1))
            for h in HORIZONS:
                if idx + h >= len(close):
                    continue
                entry = float(close[idx])
                exit_ = float(close[idx + h])
                ret = (exit_ - entry) / entry * 100 if entry else 0.0
                results_by_horizon[h].append((sign, ret))

    return {
        "note": ("Same hypothesis sign (LONG/SHORT from the direction, NO_TRADE "
                "via the sign of total_score) as in the main backtest, only "
                "the outcome is measured a different number of days ahead."),
        "3d_original": None,  # see the main file's summary — same set, for comparison
        **{f"{h}d": _hit_rate(pairs) for h, pairs in results_by_horizon.items()},
    }


# ═══════════════════════════════════════════════════════════════════════
# TASK 5: market cap / sector
# ═══════════════════════════════════════════════════════════════════════
def task5_cap_sector(records):
    tickers = sorted(set(r["ticker"] for r in records))
    firm = {t: _fetch_firmographics(t) for t in tickers}

    taken = [r for r in records if r["direction"] != "NO_TRADE"]

    def hyp_pair(r):
        sign = 1 if r["direction"] in ("LONG", "STRONG_LONG") else -1
        return (sign, r["outcome_return_pct"])

    by_cap, by_sector = {}, {}
    for r in taken:
        f = firm[r["ticker"]]
        by_cap.setdefault(f["market_cap_tier"], []).append(hyp_pair(r))
        by_sector.setdefault(f["sector"], []).append(hyp_pair(r))

    return {
        "note": "Only 'taken' (LONG/SHORT) observations; market cap/sector are a current snapshot (see the general disclaimer).",
        "by_market_cap_tier": {k: _hit_rate(v) for k, v in by_cap.items()},
        "by_sector": {k: _hit_rate(v) for k, v in by_sector.items()},
    }


# ═══════════════════════════════════════════════════════════════════════
# TASK 6: EXPECTANCY — size of wins/losses, not just frequency
# ═══════════════════════════════════════════════════════════════════════
def _signed_return(r):
    d = r["direction"]
    if d in ("LONG", "STRONG_LONG"):
        return r["outcome_return_pct"]
    if d in ("SHORT", "STRONG_SHORT"):
        return -r["outcome_return_pct"]
    return r["outcome_return_pct"] if r["total_score"] >= 0 else -r["outcome_return_pct"]


def _expectancy_stats(signed_rets):
    n = len(signed_rets)
    if n == 0:
        return {"n": 0}
    wins = [x for x in signed_rets if x > 0]
    losses = [x for x in signed_rets if x < 0]
    avg_win = round(sum(wins) / len(wins), 3) if wins else None
    avg_loss = round(sum(losses) / len(losses), 3) if losses else None
    return {
        "n": n,
        "expectancy_pct_per_trade": round(sum(signed_rets) / n, 4),
        "win_rate_pct": round(len(wins) / n * 100, 1),
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "reward_risk_ratio": round(abs(avg_win / avg_loss), 2) if avg_win and avg_loss else None,
    }


def task6_expectancy(records):
    taken = [r for r in records if r["direction"] != "NO_TRADE"]
    long_only = [r for r in taken if r["direction"] in ("LONG", "STRONG_LONG")]
    short_only = [r for r in taken if r["direction"] in ("SHORT", "STRONG_SHORT")]
    no_trade = [r for r in records if r["direction"] == "NO_TRADE"]

    return {
        "note": ("Expectancy = mean signed return per 'trade' (equal stakes, "
                "NO commissions/slippage). A positive value means "
                "profitable on average even at winrate<=50% if wins are larger than "
                "losses; negative means unprofitable even at winrate>50% if it is "
                "the other way round. This is exactly what none of the hit rates above showed. "
                "NO_TRADE — counterfactual by the sign of total_score, the same assumption "
                "as in the other tasks."),
        "taken_all": _expectancy_stats([_signed_return(r) for r in taken]),
        "taken_long_only": _expectancy_stats([_signed_return(r) for r in long_only]),
        "taken_short_only": _expectancy_stats([_signed_return(r) for r in short_only]),
        "no_trade_counterfactual": _expectancy_stats([_signed_return(r) for r in no_trade]),
    }


# ═══════════════════════════════════════════════════════════════════════
# TASK 7: multiple-comparisons correction (meta-analysis of what has been done)
# ═══════════════════════════════════════════════════════════════════════
def _collect_tests(obj, path=""):
    found = []
    if isinstance(obj, dict):
        if "wilson95_pct" in obj and obj.get("wilson95_pct") and obj.get("n"):
            found.append((path or "root", obj["n"], obj["wilson95_pct"]))
        for k, v in obj.items():
            found.extend(_collect_tests(v, f"{path}.{k}" if path else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found.extend(_collect_tests(v, f"{path}[{i}]"))
    return found


def task7_multiple_comparisons(combined_results):
    tests = _collect_tests(combined_results)
    n_tests = len(tests)
    sig = [{"test": path, "n": n, "wilson95_pct": ci}
           for path, n, ci in tests if ci[0] > 50 or ci[1] < 50]
    expected = round(n_tests * 0.05, 1)

    return {
        "note": ("Collected from all Wilson CIs (n + bounds) in all three files "
                "(historical_backtest.json, historical_backtest_llm.json, "
                "the current historical_theories.json). 'Significant' here = the CI does not "
                "cross 50% — NOT a formal Bonferroni correction (that would give "
                "unjustifiably narrow thresholds for such a heterogeneous set of "
                "tests), but an estimate: how many 'interesting' findings would be expected "
                "from pure noise at alpha=5%, and how many there actually are."),
        "n_tests_total": n_tests,
        "n_significant_cis_excluding_50pct": len(sig),
        "expected_by_chance_alone_at_5pct": expected,
        "significant_tests": sig,
    }


# ═══════════════════════════════════════════════════════════════════════
# TASK 8: excess return relative to SPY
# ═══════════════════════════════════════════════════════════════════════
def task8_excess_return_vs_spy(records):
    from agents.historical_backtest import OUTCOME_HORIZON_DAYS
    import pandas as pd

    spy = yf.download("SPY", period="3y", interval="1d", auto_adjust=False, progress=False)
    if spy is None or spy.empty:
        return {"note": "SPY failed to download", "error": True}
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy_arr = spy["Close"].values.astype(float)
    spy_dates = [d.strftime("%Y-%m-%d") for d in spy.index]
    spy_idx = {d: i for i, d in enumerate(spy_dates)}

    pairs_raw, pairs_excess = [], []
    for r in records:
        idx = spy_idx.get(r["date"])
        if idx is None or idx + OUTCOME_HORIZON_DAYS >= len(spy_arr):
            continue
        spy_entry, spy_exit = spy_arr[idx], spy_arr[idx + OUTCOME_HORIZON_DAYS]
        spy_ret = (spy_exit - spy_entry) / spy_entry * 100 if spy_entry else 0.0
        excess = r["outcome_return_pct"] - spy_ret

        sign = (1 if r["direction"] in ("LONG", "STRONG_LONG") else
               -1 if r["direction"] in ("SHORT", "STRONG_SHORT") else
               (1 if r["total_score"] >= 0 else -1))
        pairs_raw.append((sign, r["outcome_return_pct"]))
        pairs_excess.append((sign, excess))

    return {
        "note": ("excess = stock return minus SPY return over the same window "
                "(same horizon=3 days). If hit_rate_excess differs noticeably "
                "from hit_rate_raw, then raw was mostly beta to the market rather than "
                "selection skill."),
        "n_matched": len(pairs_raw),
        "hit_rate_raw": _hit_rate(pairs_raw),
        "hit_rate_excess_vs_spy": _hit_rate(pairs_excess),
    }


# ═══════════════════════════════════════════════════════════════════════
# TASK 9: opening gap (gap-and-go / gap-fill)
# ═══════════════════════════════════════════════════════════════════════
def task9_gap_analysis(records):
    import pandas as pd

    by_ticker = {}
    for r in records:
        by_ticker.setdefault(r["ticker"], []).append(r)

    gap_up, gap_down = [], []
    correl_pairs = []

    for ticker, recs in by_ticker.items():
        df = yf.download(ticker, period="3y", interval="1d", auto_adjust=False, progress=False)
        if df is None or df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        open_arr = df["Open"].values.astype(float)
        close_arr = df["Close"].values.astype(float)
        date_to_idx = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(df.index)}

        for r in recs:
            idx = date_to_idx.get(r["date"])
            if idx is None or idx == 0:
                continue
            prev_close = close_arr[idx - 1]
            if not prev_close:
                continue
            gap_pct = (open_arr[idx] - prev_close) / prev_close * 100
            ret = r["outcome_return_pct"]
            correl_pairs.append((gap_pct, ret))
            if gap_pct >= 2.0:
                gap_up.append((1, ret))     # gap-and-go hypothesis: continuation up
            elif gap_pct <= -2.0:
                gap_down.append((-1, ret))  # continuation down

    n = len(correl_pairs)
    corr = None
    if n >= 10:
        xs, ys = [p[0] for p in correl_pairs], [p[1] for p in correl_pairs]
        mx, my = sum(xs) / n, sum(ys) / n
        cov = sum((x - mx) * (y - my) for x, y in correl_pairs)
        varx = sum((x - mx) ** 2 for x in xs)
        vary = sum((y - my) ** 2 for y in ys)
        if varx > 0 and vary > 0:
            corr = round(cov / (varx * vary) ** 0.5, 3)

    return {
        "note": ("Gap-and-go hypothesis: an opening gap >=2% (up or down) "
                "continues in the same direction over the next 3 trading days. "
                "pearson_r — gap(%) vs. return over ALL observations "
                "with a gap of any size."),
        "pearson_r_gap_vs_return": corr,
        "gap_up_ge2pct_continuation": _hit_rate(gap_up),
        "gap_down_le_minus2pct_continuation": _hit_rate(gap_down),
    }


# ═══════════════════════════════════════════════════════════════════════
# TASK 10: earnings proximity (EARNINGS_BLOCK_DAYS on history)
# ═══════════════════════════════════════════════════════════════════════
def task10_earnings_proximity(records):
    by_ticker = {}
    for r in records:
        by_ticker.setdefault(r["ticker"], []).append(r)

    near_pairs, far_pairs = [], []

    for ticker, recs in by_ticker.items():
        earn_dates = []
        try:
            edates = yf.Ticker(ticker).get_earnings_dates(limit=40)
            if edates is not None and not edates.empty:
                idx = edates.index
                if idx.tz is not None:
                    idx = idx.tz_localize(None)
                earn_dates = [d.date() for d in idx]
        except Exception:
            pass

        for r in recs:
            rdate = datetime.strptime(r["date"], "%Y-%m-%d").date()
            sign = (1 if r["direction"] in ("LONG", "STRONG_LONG") else
                   -1 if r["direction"] in ("SHORT", "STRONG_SHORT") else
                   (1 if r["total_score"] >= 0 else -1))
            pair = (sign, r["outcome_return_pct"])
            days_to_nearest = min((abs((rdate - ed).days) for ed in earn_dates), default=None)
            if days_to_nearest is not None and days_to_nearest <= 3:
                near_pairs.append(pair)
            else:
                far_pairs.append(pair)

    return {
        "note": ("near = observation within ±3 calendar days of the REAL "
                "historical earnings date (the same EARNINGS_BLOCK_DAYS as "
                "in the live check_entry_filters). far = everything else. Checks "
                "whether the filter really cuts what is worse."),
        "near_earnings": _hit_rate(near_pairs),
        "far_from_earnings": _hit_rate(far_pairs),
    }


def main():
    data = json.load(open(HISTORICAL_BACKTEST_FILE, encoding="utf-8"))
    records = data["records"]
    print(f"Loaded {len(records)} observations from {HISTORICAL_BACKTEST_FILE}")

    print("Task 1/10: old vs. new STUMPY...", flush=True)
    t1 = task1_stumpy_comparison(records)

    print("Task 2/10: FINRA short volume in hindsight (may take a while — network)...", flush=True)
    t2 = task2_short_volume(records)

    print("Task 3/10: entry filter validity...", flush=True)
    t3 = task3_entry_filters(records)

    print("Task 4/10: outcome horizon sensitivity (downloading prices)...", flush=True)
    t4 = task4_horizons(records)

    print("Task 5/10: market cap/sector...", flush=True)
    t5 = task5_cap_sector(records)

    print("Task 6/10: expectancy (size, not just frequency)...", flush=True)
    t6 = task6_expectancy(records)

    print("Task 8/10: excess return vs. SPY (before task 7 — that one is a meta-analysis and runs last)...", flush=True)
    t8 = task8_excess_return_vs_spy(records)

    print("Task 9/10: opening gap (downloading prices)...", flush=True)
    t9 = task9_gap_analysis(records)

    print("Task 10/10: earnings proximity (earnings-date requests)...", flush=True)
    t10 = task10_earnings_proximity(records)

    print("Task 7/10: multiple-comparisons correction (meta-analysis of everything collected)...", flush=True)
    # Mix in the summaries from the ready first-wave files and the LLM experiment —
    # the meta-analysis must see ALL tests we have run, not only
    # those from this file.
    prior_summaries = {}
    try:
        prior_summaries["historical_backtest_summary"] = data.get("summary")
    except Exception:
        pass
    try:
        from config import HISTORICAL_BACKTEST_LLM_FILE
        llm_data = json.load(open(HISTORICAL_BACKTEST_LLM_FILE, encoding="utf-8"))
        prior_summaries["historical_backtest_llm_summary"] = llm_data.get("summary")
    except Exception as e:
        print(f"   (historical_backtest_llm.json not mixed in: {e})")

    combined_for_meta = {
        "task1": t1, "task2": t2, "task3": t3, "task4": t4, "task5": t5,
        "task8": t8, "task9": t9, "task10": t10,
        **prior_summaries,
    }
    t7 = task7_multiple_comparisons(combined_for_meta)

    out = {
        "IS_HISTORICAL_BACKTEST": True,
        "disclaimer": ("TEN TARGETED CHECKS ON HISTORICAL DATA on top of "
                      "historical_backtest.json — not real bot signals. "
                      "Methodology limitations are the same as in historical_backtest.py "
                      "(daily bars, no VWAP/news, close-to-close outcome, "
                      "not a survivorship-free retro universe; market cap/sector/earnings "
                      "dates outside the window are a current snapshot or the best available "
                      "approximation, see the notes inside each task)."),
        "generated_at": datetime.now().isoformat(),
        "source_file": HISTORICAL_BACKTEST_FILE,
        "n_source_records": len(records),
        "task1_stumpy_old_vs_new": t1,
        "task2_short_volume_historical": t2,
        "task3_entry_filter_validity": t3,
        "task4_outcome_horizon_sensitivity": t4,
        "task5_cap_and_sector": t5,
        "task6_expectancy": t6,
        "task7_multiple_comparisons_check": t7,
        "task8_excess_return_vs_spy": t8,
        "task9_gap_analysis": t9,
        "task10_earnings_proximity": t10,
    }

    with open(HISTORICAL_THEORIES_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Written: {HISTORICAL_THEORIES_FILE}")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
