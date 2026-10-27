"""
Earnings event study: does the event produce a systematic excess
return, and does it differ by the stock's liquidity.

    python -m agents.earnings_event_study build    # compute events
    python -m agents.earnings_event_study report   # breakdown by tercile

WHAT EXACTLY IS MEASURED. For each event: the return from the close of the LAST
trading session BEFORE the announcement to the close of the FIRST session AFTER it, minus
the same quantity for the benchmark. The window is chosen deliberately: it covers
the announcement regardless of whether it came before the open or after the close
(BMO/AMC), and we don't have that label — the cache stores only the date.

WHY LIQUIDITY AND NOT MARKET CAP. Only today's market cap is available
(`.info`), and applying it to an event three years ago is
outright look-ahead: the company could have been worth an order of magnitude more and
fallen, and the fall correlates with the outcome. Dollar volume is computed
RIGHT HERE from the same quotes over the 20 sessions ending BEFORE the event, so
it is contemporaneous. It is also closer to what actually makes small names
hard to trade — spread and slippage.

THE UNIT OF OBSERVATION IS THE QUARTER, NOT THE EVENT. Same lesson as in
filter_lab: earnings cluster in season, events of one week lived through
the same market, and treating 34 thousand events as independent is the
shortest path to a "significant" result out of nothing. So the main line
of the report is by quarter: in how many of N quarters the illiquid lagged the
liquid, with a sign test over quarters.

WHAT THIS MEASUREMENT DOES NOT FIX. The ticker set is today's (top candidates
of archived sweeps), so delisted and bankrupt companies are not
in it. The bias points upward and hits the bottom tercile
hardest — i.e. the real underperformance of illiquid names, if any, is
more likely UNDERESTIMATED here. For an "avoid" conclusion this is the safe side of
the error, for a "buy" conclusion it is not.
"""
import json
import statistics
import sys
import time
from math import comb

from config import EARNINGS_EVENTS_FILE

BENCHMARK = "SPY"
LOOKBACK_DV = 20        # sessions for estimating dollar volume (before the event)
MIN_HISTORY_YEARS = "6y"
BATCH = 100
BACKOFF = [15, 30, 60, 120, 300]


def _load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def binom_two_sided(k, n, p=0.5):
    """Exact two-sided binomial test (as in filter_lab)."""
    if n == 0:
        return float("nan")
    probs = [comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    return float(min(1.0, sum(pr for pr in probs if pr <= probs[k] * (1 + 1e-9))))


def _download(tickers):
    """Daily bars in a batch. Empty dict if the rate limit did not lift."""
    import yfinance as yf

    for attempt, pause in enumerate([0] + BACKOFF):
        if pause:
            time.sleep(pause)
        try:
            return yf.download(tickers, period=MIN_HISTORY_YEARS, interval="1d",
                               progress=False, auto_adjust=False, threads=False,
                               group_by="ticker")
        except Exception as exc:
            if attempt == len(BACKOFF):
                print(f"   ⚠️ Batch failed to download: {str(exc)[:90]}")
                return None
    return None


def _series(df, ticker):
    """(dates, closes, volumes) for a ticker from the shared dataframe."""
    import pandas as pd

    try:
        sub = df[ticker] if isinstance(df.columns, pd.MultiIndex) else df
        sub = sub[["Close", "Volume"]].dropna()
    except (KeyError, TypeError):
        return [], [], []
    dates = [d.date().isoformat() for d in sub.index]
    return dates, list(sub["Close"]), list(sub["Volume"])


def _bench_window(bench, d_pre, d_post):
    """Benchmark return on exactly the same dates as the stock."""
    try:
        return bench[d_post] / bench[d_pre] - 1
    except (KeyError, ZeroDivisionError, TypeError):
        return None


def _events_for(dates, closes, volumes, earnings, bench):
    """Computes event records for one ticker."""
    out = []
    for ed, surprise in earnings.items():
        # The last session STRICTLY before the event and the first STRICTLY after — the window
        # covers the announcement whatever its timing (BMO/AMC).
        before = [i for i, d in enumerate(dates) if d < ed]
        after = [i for i, d in enumerate(dates) if d > ed]
        if not before or not after:
            continue
        j, k = before[-1], after[0]
        if j < LOOKBACK_DV or closes[j] <= 0:
            continue
        ret = closes[k] / closes[j] - 1
        b = _bench_window(bench, dates[j], dates[k])
        if b is None:
            continue
        dv_window = [closes[i] * volumes[i] for i in range(j - LOOKBACK_DV, j)]
        dv = statistics.median([v for v in dv_window if v and v > 0] or [0])
        if dv <= 0:
            continue
        out.append({"d": ed, "s": surprise,
                    "exc": round((ret - b) * 100, 3), "dv": round(dv)})
    return out


def build():
    from agents.earnings_calendar import load_cache

    calendar = load_cache()
    if not calendar:
        print("Earnings date cache is empty — first run python -m agents.earnings_calendar fill")
        return 1

    done = _load(EARNINGS_EVENTS_FILE)
    todo = [t for t, v in calendar.items() if v and t not in done]
    print(f"Tickers with dates: {sum(1 for v in calendar.values() if v)}, "
          f"already computed: {len(done)}, remaining: {len(todo)}")
    if not todo:
        return 0

    bench_df = _download([BENCHMARK])
    if bench_df is None:
        print("Benchmark failed to download — aborting.")
        return 1
    b_dates, b_closes, _ = _series(bench_df, BENCHMARK)
    bench = dict(zip(b_dates, b_closes))
    print(f"Benchmark {BENCHMARK}: {len(bench)} sessions")

    for start in range(0, len(todo), BATCH):
        chunk = todo[start:start + BATCH]
        df = _download(chunk)
        if df is None:
            continue
        for t in chunk:
            dates, closes, volumes = _series(df, t)
            if len(dates) < LOOKBACK_DV + 2:
                done[t] = []
                continue
            done[t] = _events_for(dates, closes, volumes, calendar[t], bench)
        _save(EARNINGS_EVENTS_FILE, done)
        total = sum(len(v) for v in done.values())
        print(f"   {min(start + BATCH, len(todo))}/{len(todo)} tickers, "
              f"events: {total}", flush=True)

    print(f"Done. Total events: {sum(len(v) for v in done.values())}")
    return 0


def _quarter(date_iso):
    y, m, _ = date_iso.split("-")
    return f"{y}Q{(int(m) - 1) // 3 + 1}"


def report():
    data = _load(EARNINGS_EVENTS_FILE)
    events = [dict(e, t=t) for t, evs in data.items() for e in evs]
    if not events:
        print("No events — first run python -m agents.earnings_event_study build")
        return 1

    L = []
    add = L.append
    add(f"EARNINGS EVENT STUDY — {len(events)} events, "
        f"{len({e['t'] for e in events})} tickers")
    add("window: close before announcement -> close after, in excess of " + BENCHMARK)
    add("=" * 66)

    dvs = sorted(e["dv"] for e in events)
    lo_cut = dvs[len(dvs) // 3]
    hi_cut = dvs[2 * len(dvs) // 3]

    def tercile(e):
        if e["dv"] <= lo_cut:
            return "low liq."
        return "high liq." if e["dv"] > hi_cut else "medium"

    add("")
    add(f"Tercile cut-offs by dollar volume: "
        f"${lo_cut/1e6:.1f}M / ${hi_cut/1e6:.1f}M per day")
    add("")
    add(f"{'tercile':<16}{'n':>7}{'median':>11}{'mean':>11}{'>0':>8}")
    for name in ("low liq.", "medium", "high liq."):
        vals = [e["exc"] for e in events if tercile(e) == name]
        if not vals:
            continue
        beat = sum(1 for v in vals if v > 0) / len(vals) * 100
        add(f"{name:<16}{len(vals):>7}{statistics.median(vals):>+10.2f}%"
            f"{statistics.mean(vals):>+10.2f}%{beat:>7.1f}%")

    add("")
    add("  For reference and NOT independent: events cluster in earnings season,")
    add("  events of one week lived through the same market. The main")
    add("  table is below, by quarter.")

    # MAIN TABLE: the unit of observation is the quarter.
    by_q = {}
    for e in events:
        by_q.setdefault(_quarter(e["d"]), []).append(e)

    quarters = sorted(q for q, evs in by_q.items() if len(evs) >= 30)
    wins = 0
    rows = []
    for q in quarters:
        evs = by_q[q]
        lo = [e["exc"] for e in evs if tercile(e) == "low liq."]
        hi = [e["exc"] for e in evs if tercile(e) == "high liq."]
        if len(lo) < 5 or len(hi) < 5:
            continue
        m_lo, m_hi = statistics.median(lo), statistics.median(hi)
        worse = m_lo < m_hi
        wins += worse
        rows.append((q, len(lo), m_lo, len(hi), m_hi, worse))

    add("")
    add("-" * 66)
    add(f"MAIN TABLE — unit of observation is the QUARTER ({len(rows)} of them)")
    add("-" * 66)
    add(f"{'quarter':<10}{'n low':>8}{'median':>10}{'n high':>8}"
        f"{'median':>10}{'  low worse?'}")
    for q, n_lo, m_lo, n_hi, m_hi, worse in rows:
        add(f"{q:<10}{n_lo:>8}{m_lo:>+9.2f}%{n_hi:>8}{m_hi:>+9.2f}%"
            f"{'      yes' if worse else '      no'}")

    if rows:
        p = binom_two_sided(wins, len(rows))
        add("")
        add(f"Illiquid names lagged in {wins} of {len(rows)} quarters, "
            f"sign test p={p:.4f}")
        if p < 0.05:
            add("  -> the liquidity difference is systematic")
        else:
            add("  -> indistinguishable from chance; a liquidity filter")
            add("     before earnings is not justified on this data")

    print("\n".join(L))
    return 0


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    return build() if cmd == "build" else report()


if __name__ == "__main__":
    sys.exit(main())
