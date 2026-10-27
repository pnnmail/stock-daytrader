"""
STUMPY Finder — searching for similar historical patterns.
Uses stumpy.match to search for a specific segment (query) in history —
a fundamentally different operation from stumpy.stump (self-join), which looks for
self-repeating patterns inside a series, not tied to the current moment.
"""
import numpy as np
import pandas as pd
import stumpy
import yfinance as yf
from scipy import stats
from config import EXTENDED_HOURS_ENABLED

# Bars per trading day: 7 in the regular session, ~17 with pre-market and after-hours
BARS_PER_DAY = 17 if EXTENDED_HOURS_ENABLED else 7


def wilson_score_interval(wins, n, confidence=0.95):
    """
    Wilson confidence interval for a proportion of successes.
    Robust at small n (unlike the normal approximation).
    Returns (lower, upper) as fractions (0..1).
    """
    if n == 0:
        return 0.0, 1.0
    
    z = stats.norm.ppf(1 - (1 - confidence) / 2)  # 1.96 for 95%
    p = wins / n
    
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    
    lower = centre - margin
    upper = centre + margin
    
    return max(0.0, lower), min(1.0, upper)


def fmt(value, format_str="+.2f", fallback="n/a"):
    """Formats a number with protection against None."""
    if value is None:
        return fallback
    return f"{value:{format_str}}"


def find_similar_patterns(ticker, df_hourly, window=20, top_k=20, max_history_days=365):
    """
    Finds segments in the ticker's history similar to the last `window` candles.

    Args:
        ticker: stock ticker
        df_hourly: DataFrame with hourly data (the last 5 days)
        window: search window size (number of candles)
        top_k: how many nearest analogues to return. It was 5, but scoring
            (utils/signals.py) has a gate "n_samples < 8 -> signal ignored",
            i.e. with 5 the factor ALWAYS gave 0 — the search ran for nothing.
            20 makes the gate reachable and also narrows the Wilson
            confidence interval (at n=5 it was 23-88%, i.e. useless).
        max_history_days: how many days back to look at history

    Returns:
        dict with results, or None on error
    """
    try:
        print(f"   🔍 STUMPY: loading {max_history_days} days of history for {ticker}...")
        # prepost must match how the current window was loaded (df_hourly in
        # data/fetcher.py): otherwise a query of ~17 bars/day would be compared with a history
        # of 7 bars/day, i.e. windows of different duration — matches would mean nothing
        hist = yf.download(ticker, period=f"{max_history_days}d", interval="1h",
                           progress=False, prepost=EXTENDED_HOURS_ENABLED)

        if hist.empty or len(hist) < window * 2:
            print(f"   ⚠️ STUMPY: not enough historical data ({len(hist)} candles)")
            return None

        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        # Extended-session bars are NOT dropped: they have zero volume but real
        # prices (Yahoo just does not provide volume outside the regular session). For searching
        # price patterns it is the price that matters.

        close_prices = hist['Close'].values.astype(float)
        timestamps = hist.index

        if len(df_hourly) < window:
            print(f"   ⚠️ STUMPY: not enough current data ({len(df_hourly)} candles)")
            return None

        query = df_hourly['Close'].values.astype(float)[-window:]

        # Search for matches ONLY in the past — exclude the current window (and a small
        # buffer around it) from the search history, so as not to find itself as "similar"
        # with a distance of ~0 and not pick up trivially adjacent, autocorrelated
        # segments right before the current moment.
        cutoff = window * 2
        history_for_search = close_prices[:-cutoff] if len(close_prices) > cutoff else close_prices[:0]
        history_timestamps = timestamps[:-cutoff] if len(timestamps) > cutoff else timestamps[:0]

        if len(history_for_search) < window:
            print(f"   ⚠️ STUMPY: not enough history BEFORE the current window")
            return None

        print(f"   🔍 STUMPY: searching for {top_k} similar patterns (window={window})...")

        # max_distance defaults to mean(D) - 2*std(D), i.e. a very strict cut-off:
        # on real tickers only ~5 matches got through, however many were requested.
        # We relax it to 1.5 sigma — there are more analogues, but they stay
        # statistically close (these are not "any" segments).
        matches = stumpy.match(
            query, history_for_search,
            max_matches=top_k,
            max_distance=lambda D: max(np.nanmean(D) - 1.5 * np.nanstd(D), np.nanmin(D)),
        )

        if matches is None or len(matches) == 0:
            print(f"   ⚠️ STUMPY: no similar patterns found")
            return None

        def price_after(match_end_idx, days):
            """Finds the price `days` calendar days later by the real timestamp,
            not by a fixed candle offset (robust to shortened days
            and gaps in the data)."""
            target_time = history_timestamps[match_end_idx] + pd.Timedelta(days=days)
            future_mask = timestamps >= target_time
            future_slice = close_prices[future_mask]
            return float(future_slice[0]) if len(future_slice) > 0 else None

        results = []
        for distance, idx in matches:
            idx = int(idx)
            match_end_idx = idx + window - 1
            current_match_end = float(history_for_search[match_end_idx])

            future_1d = price_after(match_end_idx, 1)
            future_3d = price_after(match_end_idx, 3)
            future_5d = price_after(match_end_idx, 5)

            results.append({
                'index': idx,
                'distance': float(distance),
                'ret_1d': round((future_1d - current_match_end) / current_match_end * 100, 2) if future_1d else None,
                'ret_3d': round((future_3d - current_match_end) / current_match_end * 100, 2) if future_3d else None,
                'ret_5d': round((future_5d - current_match_end) / current_match_end * 100, 2) if future_5d else None,
            })

        valid_1d = [r['ret_1d'] for r in results if r['ret_1d'] is not None]
        valid_3d = [r['ret_3d'] for r in results if r['ret_3d'] is not None]
        valid_5d = [r['ret_5d'] for r in results if r['ret_5d'] is not None]

        summary = {
            'ticker': ticker,
            'window': window,
            'matches_found': len(results),
            'n_1d': len(valid_1d),
            'n_3d': len(valid_3d),
            'n_5d': len(valid_5d),
            'avg_ret_1d': round(np.mean(valid_1d), 2) if valid_1d else None,
            'avg_ret_3d': round(np.mean(valid_3d), 2) if valid_3d else None,
            'avg_ret_5d': round(np.mean(valid_5d), 2) if valid_5d else None,
            'win_rate_1d': round(sum(1 for r in valid_1d if r > 0) / len(valid_1d) * 100, 1) if valid_1d else None,
            'win_rate_3d': round(sum(1 for r in valid_3d if r > 0) / len(valid_3d) * 100, 1) if valid_3d else None,
            'win_rate_5d': round(sum(1 for r in valid_5d if r > 0) / len(valid_5d) * 100, 1) if valid_5d else None,
            'details': results[:3],
        }
        
        # Wilson confidence intervals
        for horizon, data in [('1d', valid_1d), ('3d', valid_3d), ('5d', valid_5d)]:
            if data:
                wins = sum(1 for r in data if r > 0)
                n = len(data)
                lower, upper = wilson_score_interval(wins, n)
                summary[f'ci_lower_{horizon}'] = round(lower * 100, 1)
                summary[f'ci_upper_{horizon}'] = round(upper * 100, 1)
        
        # Max/min for each horizon
        if valid_1d:
            summary['max_ret_1d'] = round(max(valid_1d), 2)
            summary['min_ret_1d'] = round(min(valid_1d), 2)
        if valid_3d:
            summary['max_ret_3d'] = round(max(valid_3d), 2)
            summary['min_ret_3d'] = round(min(valid_3d), 2)
        if valid_5d:
            summary['max_ret_5d'] = round(max(valid_5d), 2)
            summary['min_ret_5d'] = round(min(valid_5d), 2)

        print(f"   ✅ STUMPY: found {len(results)} matches")
        if valid_3d:
            print(f"   📊 Average return after 3 days: {summary['avg_ret_3d']}% (win rate: {summary['win_rate_3d']}%)")

        return summary

    except Exception as e:
        print(f"   ⚠️ STUMPY error: {str(e)[:200]}")
        return None


def format_stumpy_result(result):
    """
    Formats the STUMPY result for inclusion in a prompt.
    """
    if not result:
        return ""

    lines = []
    lines.append(f"\n📊 *STUMPY — HISTORICAL ANALOGUES ({result['ticker']})*")
    lines.append(f"• Similar patterns found: {result['matches_found']}")
    lines.append(f"• Analysis window: {result['window']} hourly candles (~{result['window']/BARS_PER_DAY:.1f} trading days)")
    lines.append("")
    
    # 1-day horizon
    if result['avg_ret_1d'] is not None:
        n = result.get('n_1d', '?')
        ci_low = result.get('ci_lower_1d', '?')
        ci_high = result.get('ci_upper_1d', '?')
        lines.append(f"• After 1 day (n={n}):")
        lines.append(f"    Average result: {fmt(result['avg_ret_1d'])}%")
        lines.append(f"    Win rate: {result['win_rate_1d']}% (95% CI: {ci_low}-{ci_high}%)")
        lines.append(f"    Max: {fmt(result.get('max_ret_1d'), fallback='n/a')}%, Min: {fmt(result.get('min_ret_1d'), fallback='n/a')}%")
        lines.append("")
    
    # 3-day horizon
    if result['avg_ret_3d'] is not None:
        n = result.get('n_3d', '?')
        ci_low = result.get('ci_lower_3d', '?')
        ci_high = result.get('ci_upper_3d', '?')
        lines.append(f"• After 3 days (n={n}):")
        lines.append(f"    Average result: {fmt(result['avg_ret_3d'])}%")
        lines.append(f"    Win rate: {result['win_rate_3d']}% (95% CI: {ci_low}-{ci_high}%)")
        lines.append(f"    Max: {fmt(result.get('max_ret_3d'), fallback='n/a')}%, Min: {fmt(result.get('min_ret_3d'), fallback='n/a')}%")
        lines.append("")
    
    # 5-day horizon
    if result['avg_ret_5d'] is not None:
        n = result.get('n_5d', '?')
        ci_low = result.get('ci_lower_5d', '?')
        ci_high = result.get('ci_upper_5d', '?')
        lines.append(f"• After 5 days (n={n}):")
        lines.append(f"    Average result: {fmt(result['avg_ret_5d'])}%")
        lines.append(f"    Win rate: {result['win_rate_5d']}% (95% CI: {ci_low}-{ci_high}%)")
        lines.append(f"    Max: {fmt(result.get('max_ret_5d'), fallback='n/a')}%, Min: {fmt(result.get('min_ret_5d'), fallback='n/a')}%")
        lines.append("")
    
    # Details for each match
    if result.get('details'):
        lines.append("📋 *Match details:*")
        for i, match in enumerate(result['details']):
            lines.append(f"  [{i+1}] distance={fmt(match['distance'], '.2f')} | "
                        f"1d: {fmt(match['ret_1d'])}% | "
                        f"3d: {fmt(match['ret_3d'])}% | "
                        f"5d: {fmt(match['ret_5d'])}%")
    
    # Interpretation for the agent — based on the Wilson confidence interval
    if result.get('win_rate_3d') is not None:
        ci_low = result.get('ci_lower_3d', 0)
        ci_high = result.get('ci_upper_3d', 100)
        n = result.get('n_3d', 0)
        win_rate = result['win_rate_3d']
        
        # Explaining the statistics
        lines.append(f"\n📖 *HOW TO READ THE STATISTICS:*")
        lines.append(f"• n={n} — number of historical matches (the more, the more reliable the conclusion)")
        lines.append(f"• Win rate {win_rate}% — the raw share of cases with a rise (may be random at small n)")
        lines.append(f"• 95% CI [{ci_low}% — {ci_high}%] — the Wilson confidence interval:")
        lines.append(f"  This means: if we repeated the experiment many times,")
        lines.append(f"  the true probability of a rise would be in this range 95% of the time.")
        lines.append(f"  A wide interval (as now) = little data, a narrow one = a lot of data.")
        
        # The conclusion itself
        if ci_low > 50:
            lines.append(f"\n💡 *Conclusion:* A strong bullish signal.")
            lines.append(f"  Even the conservative estimate (lower CI bound = {ci_low}%) is above 50%.")
            lines.append(f"  This means: even accounting for the uncertainty of a small n,")
            lines.append(f"  historically this pattern more often led to a rise than to a fall.")
        elif ci_high < 50:
            lines.append(f"\n⚠️ *Conclusion:* A strong bearish signal.")
            lines.append(f"  Even the optimistic estimate (upper CI bound = {ci_high}%) is below 50%.")
            lines.append(f"  This means: even accounting for the uncertainty,")
            lines.append(f"  historically this pattern more often led to a fall.")
        elif n < 10:
            lines.append(f"\n🤷 *Conclusion:* Statistically undetermined (little data).")
            lines.append(f"  At n={n} the confidence interval is too wide [{ci_low}%-{ci_high}%].")
            lines.append(f"  Use this signal only as a weak auxiliary one,")
            lines.append(f"  make the MAIN decision based on indicators, the book and news.")
        else:
            lines.append(f"\n🤷 *Conclusion:* Direction undetermined.")
            lines.append(f"  The confidence interval [{ci_low}%-{ci_high}%] crosses 50%.")
            lines.append(f"  Historically this pattern shows no consistent direction.")

    return "\n".join(lines)