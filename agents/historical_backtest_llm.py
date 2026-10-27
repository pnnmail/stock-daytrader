# -*- coding: utf-8 -*-
"""
EXPERIMENT: "IS THERE ANY POINT TO AN LLM VERDICT AT ALL" — on top of the existing
agents/historical_backtest.py. A one-off tool, not part of the live pipeline.
The result is a SEPARATE file (config.HISTORICAL_BACKTEST_LLM_FILE), not
signal_history.json/no_trade_history.json and not historical_backtest.json.

    python -m agents.historical_backtest_llm

WHAT IS TESTED. The project's architectural principle is "Python computes, the LLM only
explains" (see the project_architecture_principle note). Here that principle is
DELIBERATELY broken: the model (qwen3.5:4b, the same as TRADER_MODEL) is allowed
to give an INDEPENDENT LONG/SHORT/NO_TRADE verdict on the same historical
snapshots already run through the python formula (utils.signals.compute_factor_scores,
see historical_backtest.py) — and we compare: (1) the LLM verdict itself vs.
the outcome, (2) LLM vs. python ON THE SAME observations, (3) WHO IS RIGHT
when they disagree — that is the answer to "is there any point", not the fact
of agreement/disagreement by itself.

ANONYMISATION. The model does NOT see the ticker or the company name — only
a label (STOCK_A, STOCK_B...) and a QUANTITATIVE profile: market-cap band,
sector, daily dollar-volume band — what was asked to be kept,
but not as an exact number (an exact $3.4T market cap by itself
names the company unambiguously). WHY IT MATTERS: a local model may have
seen real news/prices AFTER the snapshot date in training — then the answer would be
not pattern analysis but leakage of knowledge about the future. Anonymisation does not remove
this risk entirely (numbers alone are sometimes recognisable), but it closes the
cheapest leakage channel — direct recognition by name/ticker.

MARKET CAP/SECTOR/VOLUME ARE A CURRENT SNAPSHOT, NOT AS OF THE SNAPSHOT DATE. yfinance does not
provide historical market cap for free; companies almost never change sector,
and market cap and volume are rounded into wide bands
precisely so that a small drift between "now" and "then" does not move a ticker into another
bucket in the vast majority of cases. This is an approximation, not an exact
historical fact — as in historical_backtest.py, but here the cost of the inaccuracy
is lower: a band, not a number.

THE "ALWAYS NO_TRADE" TRAP (see project_crypto_llm_experiment). A model that
always refuses shows zero losses and looks perfect. So for
EACH verdict (including NO_TRADE) lean_if_forced is asked separately —
the hypothetical direction if refusing were not allowed — and that is what
is used for counterfactual scoring of NO_TRADE verdicts, with the same logic
as unified in utils/signal_history.py for python NO_TRADE.

SAMPLE: 300 observations from HISTORICAL_BACKTEST_FILE — ALL "taken" python
signals (LONG/SHORT/STRONG_*) plus random NO_TRADE up to 300 (fixed
seed — reproducible). One model (qwen3.5:4b), temperature=0 — no
cross-model robustness check and no repeats of the same input in
this run (deliberately left for later).
"""
import json
import os
import random

import yfinance as yf

from config import (HISTORICAL_BACKTEST_FILE, HISTORICAL_BACKTEST_LLM_FILE,
                    TRADER_MODEL)
from utils.ollama import call_ollama_json

N_SAMPLE = 300
SEED = 20260912
SAVE_EVERY = 20

FAMOUS_TICKERS = {"AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META"}

BACKTEST_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "direction": {"enum": ["LONG", "SHORT", "NO_TRADE"]},
        # Hypothetical direction even if the verdict is NO_TRADE — see the file
        # header about the "always NO_TRADE" trap.
        "lean_if_forced": {"enum": ["LONG", "SHORT"]},
        "confidence": {"type": "integer"},
        "reasoning": {"type": "string"},
    },
    "required": ["direction", "lean_if_forced", "confidence", "reasoning"],
}


def _market_cap_tier(cap):
    if not cap:
        return "unknown"
    if cap >= 200e9:
        return "mega (>$200B)"
    if cap >= 10e9:
        return "large ($10-200B)"
    if cap >= 2e9:
        return "mid ($2-10B)"
    if cap >= 300e6:
        return "small ($300M - $2B)"
    return "micro (<$300M)"


def _dollar_volume_tier(avg_vol, price):
    if not avg_vol or not price:
        return "unknown"
    dv = avg_vol * price
    if dv >= 100e6:
        return "high (>$100M/day)"
    if dv >= 10e6:
        return "medium ($10-100M/day)"
    return "low (<$10M/day)"


def _fetch_firmographics(ticker):
    """One info call per ticker — market cap/sector/volume as a current
    snapshot (see the disclaimer in the file header)."""
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        info = {}
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    avg_vol = info.get("averageVolume") or info.get("averageDailyVolume10Day")
    return {
        "market_cap_tier": _market_cap_tier(info.get("marketCap")),
        "sector": info.get("sector") or "unknown",
        "dollar_volume_tier": _dollar_volume_tier(avg_vol, price),
    }


def _sample_records(records, n, seed):
    taken = [r for r in records if r["direction"] != "NO_TRADE"]
    no_trade = [r for r in records if r["direction"] == "NO_TRADE"]
    rnd = random.Random(seed)
    need_no_trade = max(0, n - len(taken))
    sampled_no_trade = rnd.sample(no_trade, min(need_no_trade, len(no_trade)))
    sample = taken + sampled_no_trade
    rnd.shuffle(sample)
    return sample[:n]


def _build_prompt(label, firm, raw):
    def f(v, suffix=""):
        return f"{v}{suffix}" if v is not None else "no data"

    return f"""You are analysing a snapshot of a stock's state at a specific moment. This
is a HISTORICAL anonymous snapshot — the ticker and company name are hidden on purpose,
you have no access to what happened to this particular company. Judge
ONLY by the numbers below, not by trying to guess which company it is.

OBJECT: {label}
Market cap: {firm['market_cap_tier']}
Sector: {firm['sector']}
Daily dollar volume: {firm['dollar_volume_tier']}

STATE AT DECISION TIME:
Price: ${f(raw['price'])}
RSI(14): {f(raw['rsi'])}
MACD: {f(raw['macd'])} / signal line: {f(raw['macd_signal'])}
Bollinger Bands: price at {f(raw['bb_position_pct'], '%')} of the band height (0%=lower, 100%=upper)
Volume: {f(raw['volume_ratio'], 'x')} of the 20-day average
Position in the 52-week range: {f(raw['position_in_range_pct'], '%')}
Volatility (ATR): {f(raw['atr_pct'], '%')} of price
Similar historical patterns (STUMPY, search for similar price moves): {f(raw['stumpy_n'])} found, of which {f(raw['stumpy_win_rate'], '%')} went up, average return {f(raw['stumpy_avg_return'], '%')}

Give a decision: direction for the next 3 trading days (LONG/SHORT/NO_TRADE),
and SEPARATELY — which way you would lean (lean_if_forced), even if the main verdict
is NO_TRADE (this is needed for an honest assessment of your caution, not for a trade).
Confidence 0-100. Explain briefly in English.

ANSWER WITH JSON ONLY."""


def run_experiment():
    data = json.load(open(HISTORICAL_BACKTEST_FILE, encoding="utf-8"))
    records = data["records"]
    sample = _sample_records(records, N_SAMPLE, SEED)
    print(f"Sample: {len(sample)} observations "
          f"(taken python signals: {sum(1 for r in sample if r['direction'] != 'NO_TRADE')}, "
          f"NO_TRADE: {sum(1 for r in sample if r['direction'] == 'NO_TRADE')})")

    unique_tickers = sorted(set(r["ticker"] for r in sample))
    labels = {t: f"STOCK_{i}" for i, t in enumerate(unique_tickers)}
    print(f"Unique tickers: {len(unique_tickers)} -> anonymised as {list(labels.values())[:5]}...")

    firmographics = {}
    for t in unique_tickers:
        firmographics[t] = _fetch_firmographics(t)
        print(f"   {labels[t]} ({t}): {firmographics[t]['market_cap_tier']}, "
              f"{firmographics[t]['sector']}, {firmographics[t]['dollar_volume_tier']}")

    results = []
    out_path = HISTORICAL_BACKTEST_LLM_FILE
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    for i, r in enumerate(sample, 1):
        ticker = r["ticker"]
        label = labels[ticker]
        firm = firmographics[ticker]
        prompt = _build_prompt(label, firm, r["raw"])

        try:
            verdict = call_ollama_json(prompt, TRADER_MODEL,
                                       json_schema=BACKTEST_VERDICT_SCHEMA, temperature=0)
        except Exception as e:
            print(f"   ⚠️ [{i}/{len(sample)}] {label} ({ticker} {r['date']}): {e}", flush=True)
            continue

        row = dict(r)  # ticker/date/direction/total_score/scores/outcome_return_pct/raw_sign/raw
        row["label"] = label
        row["firmographics"] = firm
        row["llm_direction"] = verdict.get("direction")
        row["llm_lean_if_forced"] = verdict.get("lean_if_forced")
        row["llm_confidence"] = verdict.get("confidence")
        row["llm_reasoning"] = verdict.get("reasoning")
        results.append(row)

        print(f"   [{i}/{len(sample)}] {label} ({ticker} {r['date']}): "
              f"python={r['direction']} llm={row['llm_direction']}"
              f"(lean {row['llm_lean_if_forced']}, conf {row['llm_confidence']})", flush=True)

        if i % SAVE_EVERY == 0:
            _save(out_path, results, sample_size=len(sample), done=i, total=len(sample))

    _save(out_path, results, sample_size=len(sample), done=len(results), total=len(sample))
    return results


def wilson(wins, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)
    return ((center - margin) / denom, (center + margin) / denom)


def _hyp_direction_and_hit(direction, lean_if_forced, total_score_or_none, outcome_return_pct):
    """Returns (hypothetical_direction, was_correct) — the same logic for
    python (total_score sets the direction on NO_TRADE) and the LLM (lean_if_forced
    sets the direction on NO_TRADE)."""
    if direction in ("LONG", "STRONG_LONG"):
        hyp = "LONG"
    elif direction in ("SHORT", "STRONG_SHORT"):
        hyp = "SHORT"
    elif lean_if_forced in ("LONG", "SHORT"):
        hyp = lean_if_forced
    elif total_score_or_none is not None:
        hyp = "LONG" if total_score_or_none >= 0 else "SHORT"
    else:
        return None, None
    hit = (outcome_return_pct > 0) if hyp == "LONG" else (outcome_return_pct < 0)
    return hyp, hit


def summarize(results):
    n = len(results)
    base_wins = sum(1 for r in results if r["raw_sign"] > 0)
    base_lo, base_hi = wilson(base_wins, n)

    py_hits, llm_hits = [], []
    agree = 0
    disagree_py_right, disagree_llm_right, disagree_both_wrong, disagree_both_right = 0, 0, 0, 0
    llm_no_trade_n = 0

    for r in results:
        py_hyp, py_hit = _hyp_direction_and_hit(
            r["direction"], None, r["total_score"], r["outcome_return_pct"])
        llm_hyp, llm_hit = _hyp_direction_and_hit(
            r["llm_direction"], r["llm_lean_if_forced"], None, r["outcome_return_pct"])

        if py_hit is not None:
            py_hits.append(py_hit)
        if llm_hit is not None:
            llm_hits.append(llm_hit)
        if r["llm_direction"] == "NO_TRADE":
            llm_no_trade_n += 1

        if py_hyp is not None and llm_hyp is not None:
            if py_hyp == llm_hyp:
                agree += 1
            else:
                if py_hit and not llm_hit:
                    disagree_py_right += 1
                elif llm_hit and not py_hit:
                    disagree_llm_right += 1
                elif py_hit and llm_hit:
                    disagree_both_right += 1
                else:
                    disagree_both_wrong += 1

    py_wins = sum(1 for h in py_hits if h)
    llm_wins = sum(1 for h in llm_hits if h)
    py_lo, py_hi = wilson(py_wins, len(py_hits))
    llm_lo, llm_hi = wilson(llm_wins, len(llm_hits))

    # LLM confidence calibration
    conf_buckets = [(0, 50), (50, 70), (70, 90), (90, 101)]
    calibration = {}
    for lo_c, hi_c in conf_buckets:
        sub = [r for r in results if r.get("llm_confidence") is not None
               and lo_c <= r["llm_confidence"] < hi_c]
        if not sub:
            continue
        hyps_hits = [_hyp_direction_and_hit(r["llm_direction"], r["llm_lean_if_forced"],
                                            None, r["outcome_return_pct"]) for r in sub]
        hits = [h for _, h in hyps_hits if h is not None]
        if hits:
            calibration[f"{lo_c}-{hi_c}"] = {
                "n": len(hits), "hit_rate_pct": round(sum(hits) / len(hits) * 100, 1)
            }

    # Well-known vs. obscure names (post hoc, for leakage — the model
    # did not see the ticker, but numbers alone are theoretically recognisable)
    fame_split = {}
    for fame_label, cond in [("famous", lambda r: r["ticker"] in FAMOUS_TICKERS),
                             ("obscure", lambda r: r["ticker"] not in FAMOUS_TICKERS)]:
        sub = [r for r in results if cond(r)]
        hyps_hits = [_hyp_direction_and_hit(r["llm_direction"], r["llm_lean_if_forced"],
                                            None, r["outcome_return_pct"]) for r in sub]
        hits = [h for _, h in hyps_hits if h is not None]
        if hits:
            lo, hi = wilson(sum(hits), len(hits))
            fame_split[fame_label] = {
                "n": len(hits), "hit_rate_pct": round(sum(hits) / len(hits) * 100, 1),
                "wilson95_pct": [round(lo * 100, 1), round(hi * 100, 1)],
            }

    return {
        "n_total": n,
        "base_rate_up_pct": round(base_wins / n * 100, 1) if n else None,
        "base_rate_wilson95_pct": [round(base_lo * 100, 1), round(base_hi * 100, 1)],
        "python_hit_rate": {
            "n": len(py_hits),
            "hit_rate_pct": round(py_wins / len(py_hits) * 100, 1) if py_hits else None,
            "wilson95_pct": [round(py_lo * 100, 1), round(py_hi * 100, 1)] if py_hits else None,
        },
        "llm_hit_rate": {
            "n": len(llm_hits),
            "hit_rate_pct": round(llm_wins / len(llm_hits) * 100, 1) if llm_hits else None,
            "wilson95_pct": [round(llm_lo * 100, 1), round(llm_hi * 100, 1)] if llm_hits else None,
        },
        "llm_no_trade_share_pct": round(llm_no_trade_n / n * 100, 1) if n else None,
        "agreement": {
            "agree_n": agree,
            "disagree_n": disagree_py_right + disagree_llm_right + disagree_both_right + disagree_both_wrong,
            "disagree_python_right": disagree_py_right,
            "disagree_llm_right": disagree_llm_right,
            "disagree_both_right": disagree_both_right,
            "disagree_both_wrong": disagree_both_wrong,
        },
        "confidence_calibration": calibration,
        "famous_vs_obscure": fame_split,
    }


def _save(path, results, sample_size, done, total):
    summary = summarize(results) if results else {}
    out = {
        "IS_HISTORICAL_BACKTEST": True,
        "IS_LLM_EXPERIMENT": True,
        "disclaimer": ("LLM-VERDICT EXPERIMENT ON ANONYMISED HISTORICAL DATA — "
                      "not real bot signals, do not mix with signal_history.json / "
                      "no_trade_history.json / historical_backtest.json. The model "
                      "saw market cap/sector/volume as a CURRENT snapshot (not as of the "
                      "snapshot date) and did not see the ticker/company name — see the "
                      "agents/historical_backtest_llm.py docstring on why and what "
                      "risks of future leakage through model memory remain."),
        "model": TRADER_MODEL,
        "progress": f"{done}/{total}",
        "sample_size": sample_size,
        "summary": summary,
        "results": results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    res = run_experiment()
    print("\n" + json.dumps(summarize(res), ensure_ascii=False, indent=2))
