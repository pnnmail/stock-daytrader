# -*- coding: utf-8 -*-
"""
TESTING THE CURRENT FORMULA ON HISTORICAL DATA. A one-off tool, not part of
the bot's live pipeline. The result is written to a separate file
(config.HISTORICAL_BACKTEST_FILE), NOT to signal_history.json/no_trade_history.json
— these are not real signals but a replay of the current formula (utils.signals.compute_factor_scores,
see SCORING_VERSION) on past prices.

RUN:
    python -m agents.historical_backtest

HONEST ABOUT THE LIMITATIONS (otherwise the result is easily mistaken for something it is not):

1. DAILY BARS ONLY. yfinance returns hourly candles no deeper than ~730 days, and
   a multi-year backtest cannot be built on them. Two direct consequences:
   - VWAP is unavailable (it is an intraday quantity) -> compute_factor_scores
     receives intraday_levels={} -> vwap_diff is always 0 -> the momentum group
     effectively equals the sign of MACD (the VWAP term is neutral, as the function
     itself provides for missing data, not through a separate hack).
   - STUMPY here works on a WINDOW OF 20 DAILY bars (~a month-long pattern), not
     20 hourly ones (~1-2 trading days) as in the live pipeline (data/stumpy_finder.py).
     It is the same algorithm (stumpy.match, same max_distance), but a different pattern
     scale — not identical to production.
2. NEWS IS NOT INCLUDED (news_analysis=None -> news=0 always). The bot has no news archive
   for an arbitrary historical date, and the factor itself in the latest measurement
   (2026-09-11, n=175) showed no effect different from the base rate anyway —
   its absence here does not change what is being tested.
3. THE OUTCOME is close-to-close 3 TRADING DAYS after the signal day's close,
   not the intraday stop/target rule that actually labels
   signal_history.json. This is comparable to what `stumpy_data` already gives
   (win_rate_3d), but NOT identical to a live trade outcome.
4. UNIVERSE SURVIVORSHIP. The ticker list below is today's, fixed
   by hand (see UNIVERSE) for diversity (mega-caps + names that actually
   appeared in the live no_trade_history/signal_history) — not a retroactive
   sample of what traded 3 years ago, so not everything delisted
   is accounted for (see the same caveat in agents/replay_lab.py).

That is exactly why this is a SEPARATE file with its own "is_historical_backtest" flag,
and not an addition to the live statistics.
"""
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import stumpy
import yfinance as yf

from config import HISTORICAL_BACKTEST_FILE
from data.indicators import compute_indicators
from utils.signals import (compute_factor_scores, determine_direction,
                           SCORING_VERSION, LONG_THRESHOLD, SHORT_THRESHOLD)

# Mega-caps (reliable long history, no liquidity surprises) + names
# that actually appeared in the live signal_history/no_trade_history.json as of
# the date the script was written (2026-09-12) — for diversity across market cap/sector,
# not only "convenient" tickers.
UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META",
    "ORCL", "CMCSA", "BUD", "BILI", "OMC", "PCG", "EIX",
    "VERA", "BEAM", "SFL", "RLGT", "CHCI", "DFIN", "PFBC",
    "SYBT", "GBLI", "UI", "VCTR", "INDO",
]

PERIOD = "3y"
STEP_DAYS = 5          # same step as in agents/replay_lab.py — reduces overlap of neighbouring dates
WARMUP_BARS = 280      # 252 for the 52W range + margin for RSI/MACD/BB
OUTCOME_HORIZON_DAYS = 3
STUMPY_WINDOW = 20
STUMPY_TOP_K = 20
STUMPY_MIN_SAMPLES = 5  # same gate as in utils/signals.py

DOWNLOAD_BACKOFF = [10, 20, 40]


def _download(ticker):
    for attempt, wait in enumerate([0] + DOWNLOAD_BACKOFF):
        if wait:
            print(f"   rate limit on {ticker}, waiting {wait}s...", flush=True)
            time.sleep(wait)
        try:
            df = yf.download(ticker, period=PERIOD, interval="1d",
                             auto_adjust=False, progress=False)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df
        except Exception as e:
            print(f"   ⚠️ {ticker}: {str(e)[:120]}", flush=True)
    return None


def _stumpy_asof(close_prices, as_of_idx):
    """Same algorithm as data/stumpy_finder.py::find_similar_patterns
    (query = last STUMPY_WINDOW bars, search in the history BEFORE the window,
    max_distance = mean-1.5*std), but on the slice close_prices[:as_of_idx+1] —
    neither the query, nor the search, nor the evaluation of matches sees anything after
    as_of_idx."""
    series = close_prices[:as_of_idx + 1]
    if len(series) < STUMPY_WINDOW * 3:
        return None
    query = series[-STUMPY_WINDOW:]
    cutoff = STUMPY_WINDOW * 2
    history = series[:-cutoff] if len(series) > cutoff else series[:0]
    if len(history) < STUMPY_WINDOW:
        return None
    try:
        matches = stumpy.match(
            query, history, max_matches=STUMPY_TOP_K,
            max_distance=lambda D: max(np.nanmean(D) - 1.5 * np.nanstd(D), np.nanmin(D)),
        )
    except Exception:
        return None
    if matches is None or len(matches) == 0:
        return None

    rets = []
    for _distance, idx in matches:
        idx = int(idx)
        match_end = idx + STUMPY_WINDOW - 1
        target = match_end + OUTCOME_HORIZON_DAYS
        if target >= len(series):
            continue
        entry = float(series[match_end])
        exit_ = float(series[target])
        if entry:
            rets.append((exit_ - entry) / entry * 100)

    if len(rets) < 3:
        return None
    n = len(rets)
    wins = sum(1 for r in rets if r > 0)
    return {
        "n_samples": n,
        "win_rate": round(wins / n * 100, 1),
        "avg_return": round(sum(rets) / n, 2),
    }


def _relative_context_asof(daily_df, as_of_idx):
    vol = daily_df["Volume"].values.astype(float)[:as_of_idx + 1]
    close = daily_df["Close"].values.astype(float)[:as_of_idx + 1]
    if len(vol) < 21:
        return {"volume_ratio": None, "position_in_range_pct": None}

    avg_vol = float(np.mean(vol[-21:-1]))
    cur_vol = float(vol[-1])
    volume_ratio = cur_vol / avg_vol if avg_vol else None

    window = close[-252:] if len(close) >= 252 else close
    h52, l52, p = float(window.max()), float(window.min()), float(close[-1])
    position = ((p - l52) / (h52 - l52) * 100) if h52 != l52 else None

    return {"volume_ratio": volume_ratio, "position_in_range_pct": position}


def _atr_pct_asof(df, as_of_idx, period=14):
    """ATR% on daily bars (classic True Range) — a rough
    analogue of intraday_levels['atr_hourly'] from the live pipeline, only on
    the daily rather than hourly scale (same caveat about VWAP/intraday)."""
    hi = df["High"].values.astype(float)[:as_of_idx + 1]
    lo = df["Low"].values.astype(float)[:as_of_idx + 1]
    close = df["Close"].values.astype(float)[:as_of_idx + 1]
    if len(close) < period + 1:
        return None
    prev_close = close[:-1]
    tr = np.maximum(hi[1:] - lo[1:],
                    np.maximum(np.abs(hi[1:] - prev_close), np.abs(lo[1:] - prev_close)))
    atr = float(np.mean(tr[-period:]))
    price = float(close[-1])
    return round(atr / price * 100, 2) if price else None


def run_backtest():
    all_records = []
    per_ticker_counts = {}

    for ti, ticker in enumerate(UNIVERSE, 1):
        print(f"[{ti}/{len(UNIVERSE)}] {ticker}: downloading {PERIOD} of daily bars...", flush=True)
        df = _download(ticker)
        if df is None or len(df) < WARMUP_BARS + STEP_DAYS:
            print(f"   ⚠️ {ticker}: not enough data, skipping", flush=True)
            continue

        close_arr = df["Close"].values.astype(float)
        n_bars = len(df)
        as_of_indices = list(range(WARMUP_BARS, n_bars - OUTCOME_HORIZON_DAYS, STEP_DAYS))
        count = 0

        for as_of_idx in as_of_indices:
            daily_slice = df.iloc[:as_of_idx + 1]
            indicators = compute_indicators(daily_slice)
            if not indicators.get("price"):
                continue

            relctx = _relative_context_asof(df, as_of_idx)
            atr_pct = _atr_pct_asof(df, as_of_idx)
            stumpy_res = _stumpy_asof(close_arr, as_of_idx)
            stumpy_data = {}
            if stumpy_res and stumpy_res["n_samples"] >= STUMPY_MIN_SAMPLES:
                stumpy_data = stumpy_res

            scores, reasons = compute_factor_scores(
                indicators=indicators,
                intraday_levels={},          # VWAP unavailable -> neutral (see disclaimer)
                news_analysis=None,          # no historical news
                stumpy_data=stumpy_data,
                relative_context=relctx,
            )
            direction, total_score = determine_direction(scores)

            entry_price = float(close_arr[as_of_idx])
            exit_price = float(close_arr[as_of_idx + OUTCOME_HORIZON_DAYS])
            # float() — otherwise numpy.float64 from close_arr does not serialize to JSON
            # directly (json.dump fails with Object of type float64).
            outcome_return_pct = float(round((exit_price - entry_price) / entry_price * 100, 3))
            raw_sign = 1 if outcome_return_pct > 0 else (-1 if outcome_return_pct < 0 else 0)

            record = {
                "ticker": ticker,
                "date": str(daily_slice.index[-1].date()),
                "direction": direction,
                "total_score": total_score,
                "scores": scores,
                "outcome_return_pct": outcome_return_pct,
                "raw_sign": raw_sign,
                # Raw inputs — not just derived scores. Needed separately
                # for agents/historical_backtest_llm.py (the LLM gets a description
                # of the state, not a ready-made python score, otherwise the test would be
                # not about the LLM but about whether it can read our own score).
                "raw": {
                    "price": indicators.get("price"),
                    "rsi": indicators.get("rsi"),
                    "macd": indicators.get("macd"),
                    "macd_signal": indicators.get("macd_signal"),
                    "bb_upper": indicators.get("bb_upper"),
                    "bb_lower": indicators.get("bb_lower"),
                    "bb_position_pct": indicators.get("bb_position_pct"),
                    "volume_ratio": relctx.get("volume_ratio"),
                    "position_in_range_pct": relctx.get("position_in_range_pct"),
                    "atr_pct": atr_pct,
                    "stumpy_win_rate": stumpy_data.get("win_rate"),
                    "stumpy_avg_return": stumpy_data.get("avg_return"),
                    "stumpy_n": stumpy_data.get("n_samples"),
                },
            }
            all_records.append(record)
            count += 1

        per_ticker_counts[ticker] = count
        print(f"   ✅ {ticker}: {count} observations", flush=True)

    return all_records, per_ticker_counts


def wilson(wins, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)
    return ((center - margin) / denom, (center + margin) / denom)


def summarize(records):
    n = len(records)
    base_wins = sum(1 for r in records if r["raw_sign"] > 0)
    base_lo, base_hi = wilson(base_wins, n)

    taken = [r for r in records if r["direction"] not in ("NO_TRADE",)]
    no_trade = [r for r in records if r["direction"] == "NO_TRADE"]

    def taken_hit(r):
        if r["direction"] in ("LONG", "STRONG_LONG"):
            return r["outcome_return_pct"] > 0
        return r["outcome_return_pct"] < 0

    taken_wins = sum(1 for r in taken if taken_hit(r))
    taken_lo, taken_hi = wilson(taken_wins, len(taken)) if taken else (0, 0)

    def cf_would_win(r):
        hyp_long = r["total_score"] >= 0
        cf_ret = r["outcome_return_pct"] if hyp_long else -r["outcome_return_pct"]
        return cf_ret > 0

    nt_would_win = sum(1 for r in no_trade if cf_would_win(r))
    nt_lo, nt_hi = wilson(nt_would_win, len(no_trade)) if no_trade else (0, 0)

    per_factor = {}
    for factor in ["news", "technical", "stumpy", "macd", "vwap"]:
        sub = [r for r in records if r["scores"].get(factor, 0) != 0]
        if not sub:
            continue
        agree = sum(1 for r in sub
                    if (1 if r["scores"][factor] > 0 else -1) == r["raw_sign"])
        lo, hi = wilson(agree, len(sub))
        per_factor[factor] = {
            "n": len(sub), "hit_rate_pct": round(agree / len(sub) * 100, 1),
            "wilson95_low_pct": round(lo * 100, 1), "wilson95_high_pct": round(hi * 100, 1),
        }

    return {
        "n_total": n,
        "base_rate_up_pct": round(base_wins / n * 100, 1) if n else None,
        "base_rate_wilson95_pct": [round(base_lo * 100, 1), round(base_hi * 100, 1)],
        "taken": {
            "n": len(taken),
            "win_rate_pct": round(taken_wins / len(taken) * 100, 1) if taken else None,
            "wilson95_pct": [round(taken_lo * 100, 1), round(taken_hi * 100, 1)] if taken else None,
        },
        "no_trade_counterfactual": {
            "n": len(no_trade),
            "would_win_rate_pct": round(nt_would_win / len(no_trade) * 100, 1) if no_trade else None,
            "wilson95_pct": [round(nt_lo * 100, 1), round(nt_hi * 100, 1)] if no_trade else None,
        },
        "per_factor": per_factor,
    }


def main():
    started = datetime.now()
    records, per_ticker_counts = run_backtest()

    summary = summarize(records) if records else {}

    out = {
        "IS_HISTORICAL_BACKTEST": True,
        "disclaimer": ("HISTORICAL DATA CHECK — a replay of the current "
                      "compute_factor_scores formula on past prices, NOT real bot "
                      "signals. Do not confuse with signal_history.json / no_trade_history.json. "
                      "Methodology limitations — see the agents/historical_backtest.py docstring "
                      "(no VWAP or news, outcome by daily closes rather than "
                      "the intraday rule)."),
        "generated_at": started.isoformat(),
        "elapsed_sec": int((datetime.now() - started).total_seconds()),
        "scoring_version": SCORING_VERSION,
        "long_threshold": LONG_THRESHOLD,
        "short_threshold": SHORT_THRESHOLD,
        "universe": UNIVERSE,
        "per_ticker_observations": per_ticker_counts,
        "summary": summary,
        "records": records,
    }

    os.makedirs(os.path.dirname(HISTORICAL_BACKTEST_FILE), exist_ok=True)
    with open(HISTORICAL_BACKTEST_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Written: {HISTORICAL_BACKTEST_FILE}")
    print(f"Total observations: {len(records)}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
