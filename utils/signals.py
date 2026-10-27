"""
Signal system for trading decisions.
All factors are scored in points, then the direction is determined.
"""

from typing import Dict, Any, Tuple
from config import LONG_THRESHOLD, SHORT_THRESHOLD
from utils.helpers import escape_markdown, safe_format_float

# Minimum stop distance from the entry as a fraction of hourly ATR. A closer stop is
# not risk management but a guaranteed shake-out on noise.
MIN_STOP_ATR = 0.5

# Ceiling for a plausible R:R in an intraday trade. Above it, this is not a great
# trade but an artefact of level calculation (a target that is too far).
MAX_REASONABLE_RR = 10.0

# Minimum R:R at which a trade makes sense at all. Extracted into a constant
# because it is checked in TWO places: here at entry (check_entry_filters) and
# when resolving the outcome (utils/signal_history.py) — for an out-of-session signal the entry
# is recomputed to the next session's open, and the R:R approved by the filter on the
# planned entry no longer applies to the actual one. Keeping the threshold as a number in two
# files means one day changing it in one of them and getting statistics on trades
# the system would have rejected at entry.
MIN_ENTRY_RR = 1.0

# Cap on the news contribution. It used to be 3 with an entry threshold of 4 — i.e. news alone
# almost reached a trade. Yet it is the least reliable input: a test of 12 models
# on one ticker gave a news score spread of +1..+3 on THE SAME two articles,
# and that is what decided LONG vs. NO_TRADE. With a cap of 2, news needs support from
# at least one other factor.
NEWS_MAX_SCORE = 2

# The actually reachable maximum of the sum: Momentum ±1 + Technical ±2 (RSI+BB)
# + News ±2 + STUMPY ±2. Used to calibrate position size and
# confidence so they are not computed from a range the system never
# reaches. It was 10 with a flat sum of five factors (see SCORING_VERSION).
MAX_TOTAL_SCORE = 7

# ═══════════════════════════════════════════════════════════════════════════
# SCORE FORMULA VERSION. Observations computed by different versions must not
# be added into one statistic: the score means different things. Written into EVERY record
# (utils/signal_history.py::_evidence_fields), statistics are computed separately —
# the same logic already applied to resolution_rule.
#
# v1_flat_sum      — a flat sum of orderflow + technical + news + stumpy + vwap,
#                    threshold ±4, maximum ±10.
# v2_momentum_cap  — the momentum group (Order Flow + MACD + VWAP) compressed to ±1,
#                    threshold ±3, maximum ±7. See the block in compute_factor_scores.
# v3_no_fake_orderflow — Order Flow REMOVED from the formula entirely. It was never
#                    real order flow: agents/orderflow.py showed the model
#                    hourly OHLC candles (which contain neither individual trades nor
#                    bid/ask) and asked it to make up a "large player"/"divergence".
#                    On 169 signals with a counterfactual the momentum group with this term
#                    gave 36.8% hits at a 45% base rate (worse than a coin), and
#                    VWAP alone — 34.6% [23-48%] (significantly worse than a coin). The momentum
#                    group is now = MACD + VWAP (see compute_factor_scores), the rule
#                    "all agree -> ±1, otherwise 0" is unchanged. The ±3 threshold is kept:
#                    a simulation on 303 accumulated v2 observations gave |total|>=3 for
#                    20.1% (it was 23.8% under the previous formula at ±3) — the same
#                    order of frequency, no reason to change the threshold.
#                    Real trade flow is now computed by utils/real_flow.py
#                    (Alpaca/IEX, real trades, not LLM text) — but for now ONLY
#                    as diagnostics in signal_history, not in the formula: counterfactuals
#                    must accumulate first to measure its value before
#                    trusting it (the same discipline already applied to RSI/BB).
# v4_stumpy_flat2  — STUMPY compressed from ±2/±1 (by win_rate/avg_return strength) to
#                    a single ±2 step. On 256 observations with a counterfactual the sign
#                    of STUMPY matched the direction 52.0% of the time (base 42.4%) —
#                    a real effect. But at |score|=1 and |score|=2 the hit rate
#                    is the same (52.1% and 51.8%) — the "confident" step is no more accurate than
#                    the "moderate" one, i.e. splitting into two steps carries no
#                    information and only down-weights half the cases for no reason.
#                    See compute_factor_scores. Technical (RSI+BB), unlike
#                    STUMPY, DID show a difference by strength (51.7% at |1| vs.
#                    61.3% at |2|, albeit on a small n=31) — so the step split is
#                    kept there.
#                    The threshold was NOT raised, although the formula produces more
#                    signals: a simulation on 781 accumulated observations (all
#                    versions, momentum and technical per the v3 formula) — at
#                    the ±3 threshold the old STUMPY gives 17.3%, the new one 25.1%. This
#                    is a goal now, not a side effect: collecting statistics
#                    faster means more passing signals, as long as it does not
#                    require down-weighting factors without grounds — and here it is
#                    exactly the opposite case, the down-weighting was unjustified.
SCORING_VERSION = "v4_stumpy_flat2"

# Minimum number of historical analogues below which STUMPY is ignored.
# Above this threshold the contribution is still scaled by n (confidence_weight).
STUMPY_MIN_SAMPLES = 5

# How many days before earnings entry is forbidden. The holding window is the signal session
# plus the next one (that is how utils/signal_history.py computes the outcome); in the worst case
# this covers 3 calendar days: a signal on Friday -> the Friday and
# Monday sessions. Hence 3, not 2. The slight overshoot is deliberate: the asymmetry here
# is sharp — a missed trade costs zero, while a gap through an announcement goes right
# past the stop and costs a full, unbounded loss.
EARNINGS_BLOCK_DAYS = 3

# RSI thresholds. They were 75/25 — so extreme that the factor almost never fired:
# on RKLB at RSI 73.5 (the Junior agent called it "overbought", Bollinger showed
# 91% of the band height) the technical factor said "neutral". Back to the classic 70/30.
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30


def compute_factor_scores(
    indicators: Dict,
    intraday_levels: Dict,
    news_analysis: Dict,
    stumpy_data: Dict,
    relative_context: Dict
) -> Tuple[Dict, Dict]:
    """
    Computes scores for each factor.

    Returns:
        scores: dict with the score of each factor
        reasons: dict with explanations for each factor
    """
    # SUMMED factors: momentum, technical, news, stumpy.
    # macd/vwap stay in the dict as DIAGNOSTICS — they enter total not
    # individually but via momentum (see the "MOMENTUM GROUP" block below). We keep
    # them visible because without them a record does not show WHAT made up
    # the momentum score, and that is the first thing you need when investigating.
    #
    # Order Flow is NO LONGER here (see SCORING_VERSION v3_no_fake_orderflow) —
    # it was not order flow but an LLM making up a "large player" from
    # hourly candles without a single real trade inside.
    scores = {
        "momentum": 0,
        "technical": 0,
        "news": 0,
        "stumpy": 0,
        "total": 0,
        # diagnostics, NOT summands:
        "macd": 0,
        "vwap": 0,
    }

    reasons = {
        "momentum": "",
        "technical": "",
        "news": "",
        "stumpy": "",
        "vwap": "",
        # Details for Technical
        "rsi": "",
        "macd": "",
        "ema": "",
        "bb": "",
        # Details for News
        "sentiment": "",
        "importance": ""
    }

    # ═══════════════════════════════════════════
    # 2. TECHNICAL (max ±2)
    # ═══════════════════════════════════════════
    rsi = indicators.get('rsi', 50)
    macd = indicators.get('macd', 0)
    macd_signal = indicators.get('macd_signal', 0)
    price = indicators.get('price', 0)
    bb_lower = indicators.get('bb_lower', price * 0.95)
    bb_upper = indicators.get('bb_upper', price * 1.05)
    
    reasons_tech = []
    tech_score = 0
    
    # RSI (weak signal, ±1)
    if rsi > RSI_OVERBOUGHT:
        tech_score += -1
        reasons['rsi'] = f"RSI {rsi:.1f} (overbought)"
        reasons_tech.append(f"RSI {rsi:.1f} (overbought)")
    elif rsi < RSI_OVERSOLD:
        tech_score += 1
        reasons['rsi'] = f"RSI {rsi:.1f} (oversold)"
        reasons_tech.append(f"RSI {rsi:.1f} (oversold)")
    else:
        reasons['rsi'] = f"RSI {rsi:.1f} (neutral)"
        reasons_tech.append(f"RSI {rsi:.1f} (neutral)")
    
    # MACD (±1) — NOT part of technical, goes into the momentum group below.
    # The reason is a measurement on 477 observations: the MACD histogram and the deviation from
    # VWAP correlate with rho=+0.49 (on continuous values), i.e. they are
    # largely the same recent price move. Adding them as
    # independent confirmations inflates the score.
    if macd > macd_signal and macd > 0:
        macd_signal_score = 1
        reasons['macd'] = "MACD bullish"
    elif macd < macd_signal and macd < 0:
        macd_signal_score = -1
        reasons['macd'] = "MACD bearish"
    else:
        macd_signal_score = 0
        reasons['macd'] = "MACD neutral"
    scores['macd'] = macd_signal_score
    
    # Bollinger (weak signal, ±1)
    if bb_upper > bb_lower:
        bb_position = (price - bb_lower) / (bb_upper - bb_lower) * 100
        if bb_position > 80:
            tech_score += -1
            reasons['bb'] = f"BB near_upper ({bb_position:.0f}%)"
            reasons_tech.append(f"BB near the upper band ({bb_position:.0f}%)")
        elif bb_position < 20:
            tech_score += 1
            reasons['bb'] = f"BB near_lower ({bb_position:.0f}%)"
            reasons_tech.append(f"BB near the lower band ({bb_position:.0f}%)")
        else:
            reasons['bb'] = f"BB middle ({bb_position:.0f}%)"
            reasons_tech.append(f"BB in the middle ({bb_position:.0f}%)")
    
    # technical is now only RSI + Bollinger — both are CONTRARIAN (they signal at
    # price extremes, not in the direction of the move). MACD is moved to the momentum
    # group. Discrete RSI and BB agree in 52 cases out of 52 when both
    # are non-zero (rho=+0.86 on continuous values) — i.e. this too is almost one signal,
    # but we do NOT compress them: on accumulated counterfactuals they are the only ones
    # showing hits above the base rate (60% and 59%), and cutting
    # the only thing that works on a sample of n=169 from a biased population is
    # a decision the data does not justify yet.
    scores['technical'] = tech_score
    reasons['technical'] = ", ".join(reasons_tech)
    
    # ═══════════════════════════════════════════
    # 3. NEWS (max ±NEWS_MAX_SCORE, weighted by importance)
    # ═══════════════════════════════════════════
    if news_analysis:
        sentiment = news_analysis.get('overall_sentiment', 'neutral')
        catalysts = news_analysis.get('catalysts', [])
        risk_flags = news_analysis.get('risk_flags', [])
        summary = news_analysis.get('summary', '')
        
        # News importance estimate.
        # This used to be a flat "if catalysts: +2" — one mentioned catalyst
        # weighed as much as five confirmed ones, and risks gave only -1, i.e.
        # the scale was asymmetric. Now we scale by count: this
        # grades the signal within the cap and makes it symmetric.
        importance = 0
        importance += min(2, len(catalysts))
        importance -= min(2, len(risk_flags))

        # Keywords are searched in the SUMMARY, i.e. in the model's retelling, not in the
        # news itself — a weak heuristic on top of an already noisy input, so
        # the weight is reduced from ±2 to ±1.
        summary_lower = summary.lower()
        if any(w in summary_lower for w in ['approv', 'won ', 'contract', 'awarded']):
            importance += 1
        if any(w in summary_lower for w in ['layoff', 'lawsuit', 'penalt', 'sued']):
            importance -= 1
        
        importance = max(-NEWS_MAX_SCORE, min(NEWS_MAX_SCORE, importance))
        reasons['importance'] = f"{importance:+d}"
        
        # Sentiment sets the DIRECTION, importance sets the magnitude.
        # It used to be max(1, importance) for positive: sentiment acted as a floor,
        # and "positive, but 3 risk flags and no catalysts" (importance -2)
        # still gave +1 bullish points. Now a strong contradiction between
        # tone and content ZEROES the factor but does not flip it —
        # flipping would mean we consider ourselves smarter than the model in its own assessment.
        if sentiment == 'positive':
            raw = 1 + importance
            scores['news'] = max(0, min(NEWS_MAX_SCORE, raw))
            reasons['sentiment'] = "positive"
        elif sentiment == 'negative':
            raw = -1 + importance
            scores['news'] = min(0, max(-NEWS_MAX_SCORE, raw))
            reasons['sentiment'] = "negative"
        else:
            # "mixed" — the model could NOT determine the tone. Such a signal must weigh
            # less than an explicit positive/negative, otherwise uncertainty gives as many
            # points as confidence: on TEM, mixed with importance +2 gave the same +2
            # as a confident positive. We halve it.
            capped = max(-NEWS_MAX_SCORE, min(NEWS_MAX_SCORE, importance))
            scores['news'] = int(capped / 2)
            reasons['sentiment'] = "mixed"
        reasons['news'] = f"{reasons['sentiment']} (importance: {importance:+d})"
    else:
        scores['news'] = 0
        reasons['sentiment'] = "no news"
        reasons['news'] = "no news"
    
    # ═══════════════════════════════════════════
    # 4. STUMPY (max ±4, limited by n)
    # ═══════════════════════════════════════════
    if stumpy_data and stumpy_data.get('n_samples', 0) >= 3:
        win_rate = stumpy_data.get('win_rate', 50)
        avg_return = stumpy_data.get('avg_return', 0)
        n_samples = stumpy_data.get('n_samples', 0)
        
        # ⚠️ Limitation: at small n the signal is weakened or ignored.
        # The threshold used to be 8, but stumpy returns ~5 matches on real tickers (a hard
        # distance cut-off), so the factor was ALWAYS zeroed. The gate was lowered to
        # STUMPY_MIN_SAMPLES: below it there really is no data, and above it the contribution is
        # scaled by confidence_weight = n/15 anyway, i.e. at n=5 you get ±1, not ±2.
        if n_samples < STUMPY_MIN_SAMPLES:
            scores['stumpy'] = 0
            reasons['stumpy'] = (f"not enough data (n={n_samples}<{STUMPY_MIN_SAMPLES}), "
                                 f"signal ignored")
        else:
            # The larger the sample, the higher the trust
            confidence_weight = min(1.0, n_samples / 15)

            # THE TWO STEPS (±2 at an extreme, ±1 at a moderate signal) WERE
            # REMOVED IN v4_stumpy_flat2. Measured on 256 observations with a counterfactual
            # (utils/signals.py — the sign of scores['stumpy'] vs. the actual
            # price direction): a 52.0% hit rate at a 42.4% base rate —
            # a real effect, the group as a whole is above base. But when split by
            # signal strength (|score|=1 vs. |score|=2) the hit rate is THE SAME:
            # 52.1% and 51.8% — the moderate signal (previously ±1) is NO worse than the "confident" one
            # (previously ±2). So the two steps by win_rate/avg_return carry no
            # additional information about accuracy, just a lower weight for
            # half of the cases for no reason. Hence a single ±2 step instead of
            # ±2/±1: not "raising it by eye" but aligning the weight with what was
            # measured (the weak step is pulled up to the strong one, not the other way round
            # — the strong one is not confirmed as more accurate, so there is no reason to cut
            # the rest down to it). confidence_weight (by n_samples) is left alone —
            # almost all observations have n>=15, the weight there is always ~1.0, and
            # there is nothing to test it on separately yet.
            if win_rate >= 55 and avg_return > 0:
                raw_score = 2
            elif win_rate <= 45 and avg_return < 0:
                raw_score = -2
            else:
                raw_score = 0

            scores['stumpy'] = round(raw_score * confidence_weight)
            reasons['stumpy'] = f"win_rate {win_rate:.0f}%, avg {avg_return:.1f}% (n={n_samples}, weight={confidence_weight:.1f})"
    else:
        scores['stumpy'] = 0
        reasons['stumpy'] = "not enough data (n<3)"
    
    # ═══════════════════════════════════════════
    # 5. VWAP (max ±1)
    # ═══════════════════════════════════════════
    price = indicators.get('price', 0)
    vwap = intraday_levels.get('vwap', price)
    
    if vwap and price:
        vwap_diff = (price - vwap) / vwap * 100
        if vwap_diff > 1.0:
            scores['vwap'] = 1
            reasons['vwap'] = f"price above VWAP (+{safe_format_float(vwap_diff)}%)"
        elif vwap_diff < -1.0:
            scores['vwap'] = -1
            reasons['vwap'] = f"price below VWAP ({safe_format_float(vwap_diff)}%)"
        else:
            scores['vwap'] = 0
            reasons['vwap'] = f"price at VWAP ({safe_format_float(vwap_diff)}%)"
    else:
        scores['vwap'] = 0
        reasons['vwap'] = "no VWAP"
    
    # ═══════════════════════════════════════════════════════════════════════
    # MOMENTUM GROUP: MACD + VWAP -> maximum ±1
    #
    # WHY THE GROUP EXISTS AT ALL. Both are derivatives of the same recent
    # price move, and before v2 they (together with Order Flow) were added as
    # independent confirmations. Measured on 477 accumulated observations
    # (continuous values): MACD histogram ~ deviation from VWAP rho=+0.49.
    #
    # ORDER FLOW IS NO LONGER THE THIRD MEMBER OF THE GROUP (see SCORING_VERSION
    # v3_no_fake_orderflow) — it was not real order flow: hourly
    # OHLC candles contain neither individual trades nor bid/ask, and the LLM on top
    # of them simply made up a "large player". On 169 observations with a counterfactual
    # triple agreement (including orderflow) gave 36.8% hits at a 45% base
    # rate — worse than a coin, and VWAP alone — 34.6% [23.2, 48.2],
    # significantly worse than a coin. Real trade flow (Alpaca/IEX) is computed
    # separately in utils/real_flow.py, but for now only as diagnostics — it
    # will enter the formula only after the counterfactual shows value in it.
    #
    # WHY A HARD ±1 CAP AND NOT A ±2 SUM. With two strongly correlated
    # inputs agreement is expected and carries little information (the same rho=+0.49), while
    # disagreement means momentum cannot be trusted at all — then 0, not
    # the average.
    # ═══════════════════════════════════════════════════════════════════════
    momentum_parts = [scores['macd'], scores['vwap']]
    nonzero = [v for v in momentum_parts if v]
    if not nonzero:
        scores['momentum'] = 0
        reasons['momentum'] = "no clear momentum"
    elif all(v > 0 for v in nonzero):
        scores['momentum'] = 1
        reasons['momentum'] = _momentum_reason(reasons, nonzero, "up")
    elif all(v < 0 for v in nonzero):
        scores['momentum'] = -1
        reasons['momentum'] = _momentum_reason(reasons, nonzero, "down")
    else:
        scores['momentum'] = 0
        reasons['momentum'] = (
            f"momentum signals diverge ({reasons.get('macd', '?')} / "
            f"{reasons.get('vwap', '?')}) — momentum not trusted"
        )

    # ═══════════════════════════════════════════
    # TOTAL
    # ═══════════════════════════════════════════
    scores['total'] = sum([
        scores['momentum'],
        scores['technical'],
        scores['news'],
        scores['stumpy'],
    ])

    return scores, reasons


def _momentum_reason(reasons: Dict, nonzero: list, direction_word: str) -> str:
    """Readable explanation of the momentum group: how many of its members spoke up
    and what exactly. Without this the report shows only the final ±1, which cannot
    tell an agreeing pair from a single fired signal."""
    parts = [reasons.get(k) for k in ("macd", "vwap")
             if reasons.get(k) and "neutral" not in reasons.get(k, "").lower()
             and "at VWAP" not in reasons.get(k, "")]
    detail = "; ".join(p for p in parts if p) or "—"
    return f"{len(nonzero)} of 2 {direction_word} ({detail})"


def calculate_entry_levels(direction, price, intraday_levels, volume_profile, liquidity):
    """Python computes entry, stop and target levels."""
    levels = {
        "entry_price": price,
        "stop_loss": None,
        "target_1": None,
        "target_2": None,
        "risk_reward_1": 0,
        "risk_reward_2": 0
    }
    
    vwap = intraday_levels.get('vwap', price)
    pivot = intraday_levels.get('pivot', price)
    # Pivot/R1/R2/S1/S2 are computed from the H/L/C of the BASE (previous) session. If
    # there has already been a gap today relative to this Pivot (see PIVOT_GAP_THRESHOLD_PCT
    # in data/indicators.py), these levels extrapolate a range that
    # the market does not recognise today: there was a case with targets 20%+ away from the price.
    # In that case we drop them from stop/target candidates, keeping VWAP,
    # POC and prev_high/prev_low — those are computed from current data.
    pivot_reliable = intraday_levels.get('pivot_reliable', True)
    s1 = intraday_levels.get('s1', price * 0.98) if pivot_reliable else None
    s2 = intraday_levels.get('s2', price * 0.96) if pivot_reliable else None
    r1 = intraday_levels.get('r1', price * 1.02) if pivot_reliable else None
    r2 = intraday_levels.get('r2', price * 1.04) if pivot_reliable else None
    poc = volume_profile.get('poc', price)
    # atr_hourly can be exactly None (compute_intraday_levels writes None if
    # there is little data) — .get with a default does not cover that case, `or` is needed
    atr = intraday_levels.get('atr_hourly') or price * 0.015
    prev_low = liquidity.get('prev_low', price * 0.98)
    prev_high = liquidity.get('prev_high', price * 1.02)

    if direction in ["LONG", "STRONG_LONG"]:
        # Stop: below S1 or POC or prev_low (pick the one closest to the price)
        stop_candidates = [s1, poc * 0.99, prev_low]
        stop_candidates = [x for x in stop_candidates if x and x < price]
        stop_loss = max(stop_candidates) if stop_candidates else price - atr

        # The nearest level may turn out right next to the price — then the stop lands
        # inside normal market noise and gets hit immediately, while R:R inflates
        # to meaningless values (1:16 and 1:53 with a 0.25-0.31% stop were seen
        # in production). Push the stop at least MIN_STOP_ATR away from the entry.
        if price - stop_loss < atr * MIN_STOP_ATR:
            stop_loss = price - atr * MIN_STOP_ATR

        levels["stop_loss"] = round(stop_loss, 2)
        
        # Targets
        targets = [vwap, r1, r2]
        targets = [x for x in targets if x and x > price]
        
        if len(targets) >= 2:
            levels["target_1"] = round(targets[0], 2)
            levels["target_2"] = round(targets[1], 2)
        elif len(targets) == 1:
            levels["target_1"] = round(targets[0], 2)
            levels["target_2"] = round(targets[0] * 1.02, 2)
        else:
            levels["target_1"] = round(price + atr * 2, 2)
            levels["target_2"] = round(price + atr * 4, 2)
    
    elif direction in ["SHORT", "STRONG_SHORT"]:
        # Stop: above R1 or POC or prev_high
        stop_candidates = [r1, poc * 1.01, prev_high]
        stop_candidates = [x for x in stop_candidates if x and x > price]
        stop_loss = min(stop_candidates) if stop_candidates else price + atr

        # Symmetric to LONG: do not let the stop end up inside market noise
        if stop_loss - price < atr * MIN_STOP_ATR:
            stop_loss = price + atr * MIN_STOP_ATR

        levels["stop_loss"] = round(stop_loss, 2)
        
        # Targets
        targets = [vwap, s1, s2]
        targets = [x for x in targets if x and x < price]
        
        if len(targets) >= 2:
            levels["target_1"] = round(targets[0], 2)
            levels["target_2"] = round(targets[1], 2)
        elif len(targets) == 1:
            levels["target_1"] = round(targets[0], 2)
            levels["target_2"] = round(targets[0] * 0.98, 2)
        else:
            levels["target_1"] = round(price - atr * 2, 2)
            levels["target_2"] = round(price - atr * 4, 2)
    
    # R:R calculation
    if levels["stop_loss"] and levels["target_1"]:
        risk = abs(price - levels["stop_loss"])
        if direction in ["LONG", "STRONG_LONG"]:
            reward = abs(levels["target_1"] - price)
        else:
            reward = abs(price - levels["target_1"])
        levels["risk_reward_1"] = round(reward / risk, 1) if risk > 0 else 0
        
        if levels["target_2"]:
            if direction in ["LONG", "STRONG_LONG"]:
                reward2 = abs(levels["target_2"] - price)
            else:
                reward2 = abs(price - levels["target_2"])
            levels["risk_reward_2"] = round(reward2 / risk, 1) if risk > 0 else 0
    
    return levels


def determine_direction(scores: dict) -> tuple:
    total = scores['total']
    
    if total >= LONG_THRESHOLD:
        return "STRONG_LONG" if total >= LONG_THRESHOLD + 4 else "LONG", total
    elif total <= SHORT_THRESHOLD:
        return "STRONG_SHORT" if total <= SHORT_THRESHOLD - 4 else "SHORT", total
    else:
        return "NO_TRADE", total


def check_entry_filters(
    direction: str,
    intraday_levels: Dict,
    trader_plan: Dict,
    indicators: Dict,
    relative_context: Dict = None,
    earnings_info: Dict = None
) -> Tuple[bool, str]:
    """
    Step 2: check the entry filters.

    earnings_info (from data.fetcher.fetch_earnings_info) is optional: at
    the sweep stage it is absent, because it is a separate network request FOR EVERY
    ticker (see the comment in agents/screener.py on why earnings
    proximity is not computed for 1500 names). When it is not passed, the check
    is silently skipped — sweep behaviour does not change.

    Returns:
        (ok, reason): (True/False, "reason")
    """
    if direction in ["NO_TRADE"]:
        return False, "No signal"

    # Earnings inside the holding window is a hard cut-off, not a warning.
    # This is not about finding an edge but about risk: the plan is built "for the session until
    # the close", outcome resolution looks at the signal session + the next one, i.e.
    # a binary announcement inside this window carries the price past the stop with a gap —
    # a stop does not fill at its price on a gap. That this is not theory is visible in
    # the accumulated history: 7 of 18 resolved outcomes turned out to be gap outcomes
    # (gap_no_edge / gap_invalidated), i.e. trades that effectively did not happen.
    days = (earnings_info or {}).get("days_until")
    if days is not None and 0 <= days <= EARNINGS_BLOCK_DAYS:
        return False, f"Earnings in {days} d. — a binary event inside the holding window"

    # R:R filter
    rr1 = trader_plan.get('risk_reward_1', 0)
    if rr1 < MIN_ENTRY_RR:
        return False, f"R:R too low ({safe_format_float(rr1)})"

    # An abnormally high R:R for intraday is a sign that the target is unrealistically
    # far (e.g. R2 13% away from the price), not a profitable trade. Such "trades"
    # clutter the history and inflate the statistics.
    if rr1 > MAX_REASONABLE_RR:
        return False, f"R:R implausibly high ({safe_format_float(rr1)}) — target too far"

    # Volume (the real volume context, not intraday_levels — there is no such key there)
    volume_ratio = (relative_context or {}).get('volume_ratio', 1)
    if volume_ratio < 0.4:
        return False, f"Volume too low ({safe_format_float(volume_ratio)}x)"
    
    # ATR (volatility)
    atr = intraday_levels.get('atr_hourly', 0)
    price = indicators.get('price', 0)
    if atr and price:
        atr_pct = atr / price * 100
        if atr_pct < 0.2:
            return False, f"Volatility too low ({safe_format_float(atr_pct)}%)"
        if atr_pct > 8:
            return False, f"Volatility too high ({safe_format_float(atr_pct)}%)"
    
    return True, "OK"


def calculate_position_size(direction: str, total_score: int, entry_filters_ok: bool) -> int:
    """
    Step 3: determine position size (% of the account).
    """
    if not entry_filters_ok:
        return 0
    
    # The ladder is recalibrated to the ACTUALLY reachable range ±MAX_TOTAL_SCORE.
    # The old thresholds (10 -> 5%, 8 -> 3%) were written for a hypothetical maximum of ±12,
    # which was unreachable: STUMPY always gave 0, the ceiling was ±8, and STRONG
    # required exactly 8 = perfect on all factors. In practice position size
    # always came out as 1%, i.e. the field carried no information.
    strength = abs(total_score)

    if direction in ["STRONG_LONG", "STRONG_SHORT"]:
        if strength >= MAX_TOTAL_SCORE - 1:   # 9-10 of 10: an exceptional match
            return 4
        return 3
    elif direction in ["LONG", "SHORT"]:
        if strength >= 6:
            return 2
        return 1

    return 0


def compute_confidence(direction: str, total_score: int, entry_filters_ok: bool) -> float:
    """
    Computes confidence (0..1) — how strongly the factors agreed.

    Computed continuously from the share of the score relative to the maximum, not in steps:
    the old thresholds (>=10, >=7, >=4) with a real range of ±8 almost always
    fell into the same step, and confidence came out as exactly 65% in every
    report — i.e. the field was a constant, not information.
    """
    if not entry_filters_ok:
        return 0.0

    strength = min(abs(total_score), MAX_TOTAL_SCORE) / MAX_TOTAL_SCORE

    # Entry at the threshold -> ~55%, a perfect match of all factors -> ~90%
    confidence = 0.5 + 0.4 * strength
    return round(max(0.1, min(0.9, confidence)), 2)


def format_signals(scores: Dict, reasons: Dict) -> str:
    """
    Formats signals for console and Telegram output.
    """
    lines = []
    lines.append("📊 *SIGNALS (points):*")
    
    # Order and composition reflect what ACTUALLY adds up into total.
    # MACD and VWAP are no longer separate rows — they are collapsed into "Momentum"
    # (see the MOMENTUM GROUP block in compute_factor_scores), otherwise the report
    # would show two pluses where only one goes into the score.
    factor_order = ["momentum", "technical", "news", "stumpy"]
    factor_names = {
        "momentum": "Momentum (MACD+VWAP)",
        "technical": "Technical (RSI+BB)",
        "news": "News",
        "stumpy": "STUMPY",
    }
    
    for key in factor_order:
        score = scores.get(key, 0)
        reason = reasons.get(key, "")
        
        if score > 0:
            emoji = "🟢"
            sign = f"+{score}"
        elif score < 0:
            emoji = "🔴"
            sign = f"{score}"
        else:
            emoji = "⚪"
            sign = "0"
        
        lines.append(f"   {emoji} *{factor_names[key]}*: {sign} — {reason}")
    
    lines.append("")
    lines.append(f"   🎯 *TOTAL*: **{scores['total']:+d}**")
    
    return "\n".join(lines)


def get_dominant_factor(scores: dict, reasons: dict) -> tuple:
    """
    Returns the factor that best explains NO_TRADE.

    Previously the maximum by absolute value among ALL factors was taken — so
    the "main reason" for refusal could turn out to be the strongest factor FOR the trade
    (e.g. MACD +1 with total leaning LONG), although the reason for
    refusal should explain what is missing, not what is simply the largest.
    Now we take the factor pulling AGAINST the direction
    total leans to: at total>=0 (leaning LONG) — the most negative
    (or least positive) factor, at total<0 (leaning SHORT) — the most
    positive (or least negative).
    """
    factor_names = {
        "technical": "Technical",
        "news": "News",
        "stumpy": "STUMPY",
        "vwap": "VWAP"
    }

    candidates = {k: v for k, v in scores.items() if k != "total"}
    if not candidates:
        return None, 0, ""

    total = scores.get("total", 0)
    worst_key = min(candidates, key=candidates.get) if total >= 0 else max(candidates, key=candidates.get)

    return factor_names.get(worst_key, worst_key), candidates[worst_key], reasons.get(worst_key, "")


def format_factor_bar(score: int, max_abs: int = None) -> str:
    """
    Builds a bar proportional to signal strength.
    Length: 1 to 5 characters.
    """
    if max_abs is None:
        max_abs = 5
    if max_abs == 0:
        max_abs = 1
    ratio = min(1.0, abs(score) / max_abs)
    filled = max(1, int(ratio * 5))
    bar = "█" * filled + "░" * (5 - filled)
    return bar


def format_factor_details(scores: dict, reasons: dict, factor_key: str) -> str:
    """
    Returns a detailed explanation for a composite factor.
    """
    if factor_key == "technical":
        rsi = reasons.get('rsi', '')
        macd = reasons.get('macd', '')
        ema = reasons.get('ema', '')
        bb = reasons.get('bb', '')
        details = []
        if rsi:
            details.append(f"RSI: {rsi}")
        if macd:
            details.append(f"MACD: {macd}")
        if ema:
            details.append(f"EMA: {ema}")
        if bb:
            details.append(f"BB: {bb}")
        return " | ".join(details) if details else ""
    elif factor_key == "news":
        sentiment = reasons.get('sentiment', '')
        importance = reasons.get('importance', '')
        if sentiment and importance:
            return f"Sentiment: {sentiment} (importance: {importance})"
        return sentiment
    return ""


# Maximum absolute value each factor can produce (see compute_factor_scores).
# Needed so as not to suggest as a "path to a trade" something physically unreachable
# for a single factor (e.g. VWAP cannot give +2 — the cap is ±1).
FACTOR_MAX_ABS = {
    "technical": 3,
    "news": NEWS_MAX_SCORE,
    "stumpy": 2,
    "vwap": 1,
}


def generate_entry_conditions_detailed(scores: dict, reasons: dict, direction: str, threshold: int) -> str:
    """
    Generates entry conditions for NO_TRADE: what ANY ONE factor would need
    (with the others unchanged) to close the gap to the threshold on its own —
    these are different ALTERNATIVE paths, not a checklist "all needed at once". Previously
    the factors were listed one under another without this note, so
    "needs +1" and "needs +2" on neighbouring lines read as a simultaneous
    requirement, although each one alone already closes the whole sum (total
    is just the sum of factors, so growth of any ONE by the required amount
    is enough).

    It also fixes two bugs in the old formula:
    - `short_needed = -threshold - total` on NO_TRADE (where total is always
      strictly greater than -threshold) was ALWAYS negative — the "For SHORT" block
      could never be shown, at any total.
    - Because of that "For LONG" was always shown, even when total clearly
      leaned SHORT (e.g. -3). Now one side is shown —
      the one whose threshold total is closer to.
    """
    factor_names = {
        "technical": "Technical",
        "news": "News",
        "stumpy": "STUMPY",
        "vwap": "VWAP"
    }

    if direction != "NO_TRADE":
        return ""

    total = scores.get("total", 0)
    lean_long = total >= 0
    needed = (threshold - total) if lean_long else (total - (-threshold))
    if needed <= 0:
        return "All factors need re-evaluation"

    options = []
    for key, name in factor_names.items():
        score = scores.get(key, 0)
        cap = FACTOR_MAX_ABS.get(key, 2)
        target = score + needed if lean_long else score - needed
        feasible = target <= cap if lean_long else target >= -cap
        if feasible:
            options.append(f"   • {name}: needs {target:+d} (now {score:+d})")

    label = "LONG" if lean_long else "SHORT"
    lines = [f"🔹 *Any one of the options below closes the gap of {needed} points to {label}:*"]
    if options:
        lines.extend(options)
    else:
        lines.append(f"   • no single factor covers {needed} points alone — a combination of several is needed")
    return "\n".join(lines)


def compute_market_quality_detailed(indicators: dict, intraday_levels: dict, scores: dict, relative_context: dict) -> dict:
    """
    Deterministic market quality assessment with an explanation.
    """
    quality = {}
    
    # 1. Trend (SMA20, SMA50)
    price = indicators.get('price', 0)
    sma20 = intraday_levels.get('sma20', price)
    sma50 = intraday_levels.get('sma50', price)
    
    trend_score = 0
    if price > sma20:
        trend_score += 1
    if price > sma50:
        trend_score += 1
    quality['trend'] = min(5, 2 + trend_score)
    quality['trend_explain'] = f"Price {'above' if price > sma20 else 'below'} SMA20"
    
    # 2. Momentum (RSI, MACD)
    rsi = indicators.get('rsi', 50)
    macd = indicators.get('macd', 0)
    macd_signal = indicators.get('macd_signal', 0)
    
    momentum_score = 0
    if rsi < 30:
        momentum_score += 1
    elif rsi > 70:
        momentum_score -= 1
    if macd > macd_signal:
        momentum_score += 1
    else:
        momentum_score -= 1
    quality['momentum'] = min(5, max(1, 3 + momentum_score))
    quality['momentum_explain'] = f"RSI={safe_format_float(rsi)}, MACD={'bullish' if macd > macd_signal else 'bearish'}"
    
    # 3. Liquidity (volume) — uses relative_context
    volume_ratio = relative_context.get('volume_ratio', 1)
    if volume_ratio > 1.5:
        quality['liquidity'] = 5
        quality['liquidity_explain'] = f"Volume {safe_format_float(volume_ratio)}x above normal"
    elif volume_ratio > 1.0:
        quality['liquidity'] = 4
        quality['liquidity_explain'] = f"Volume {safe_format_float(volume_ratio)}x (normal)"
    elif volume_ratio > 0.7:
        quality['liquidity'] = 3
        quality['liquidity_explain'] = f"Volume {safe_format_float(volume_ratio)}x (below normal)"
    elif volume_ratio > 0.4:
        quality['liquidity'] = 2
        quality['liquidity_explain'] = f"Volume {safe_format_float(volume_ratio)}x (low)"
    else:
        quality['liquidity'] = 1
        quality['liquidity_explain'] = f"Volume {safe_format_float(volume_ratio)}x (very low)"
    
    # 4. Volatility (ATR)
    atr = intraday_levels.get('atr_hourly', 0)
    if price > 0:
        atr_pct = (atr / price) * 100
        if atr_pct > 3:
            quality['volatility'] = 5
            quality['volatility_explain'] = f"Volatility {safe_format_float(atr_pct)}% (high)"
        elif atr_pct > 1.5:
            quality['volatility'] = 4
            quality['volatility_explain'] = f"Volatility {safe_format_float(atr_pct)}% (medium)"
        elif atr_pct > 0.7:
            quality['volatility'] = 3
            quality['volatility_explain'] = f"Volatility {safe_format_float(atr_pct)}% (low)"
        elif atr_pct > 0.3:
            quality['volatility'] = 2
            quality['volatility_explain'] = f"Volatility {safe_format_float(atr_pct)}% (very low)"
        else:
            quality['volatility'] = 1
            quality['volatility_explain'] = f"Volatility {safe_format_float(atr_pct)}% (minimal)"
    else:
        quality['volatility'] = 3
        quality['volatility_explain'] = "No data"
    
    # 5. Signal Quality (based on the total score)
    total = scores.get('total', 0)
    if abs(total) >= 8:
        quality['signal'] = 5
        quality['signal_explain'] = f"Strong signal (|{total}| ≥ 8)"
    elif abs(total) >= 5:
        quality['signal'] = 4
        quality['signal_explain'] = f"Good signal (|{total}| ≥ 5)"
    elif abs(total) >= 3:
        quality['signal'] = 3
        quality['signal_explain'] = f"Average signal (|{total}| ≥ 3)"
    elif abs(total) >= 1:
        quality['signal'] = 2
        quality['signal_explain'] = f"Weak signal (|{total}| ≥ 1)"
    else:
        quality['signal'] = 1
        quality['signal_explain'] = "No signal (0)"
    
    return quality


def format_market_quality_detailed(quality: dict) -> str:
    """
    Formats market quality with explanations.
    """
    names = {
        'trend': 'Trend',
        'momentum': 'Momentum',
        'liquidity': 'Liquidity',
        'volatility': 'Volatility',
        'signal': 'Signal quality'
    }
    explain_keys = {
        'trend': 'trend_explain',
        'momentum': 'momentum_explain',
        'liquidity': 'liquidity_explain',
        'volatility': 'volatility_explain',
        'signal': 'signal_explain'
    }
    
    lines = []
    for key, name in names.items():
        value = quality.get(key, 0)
        stars = "⭐" * value + "☆" * (5 - value)
        explain = quality.get(explain_keys.get(key, ''), '')
        lines.append(f"   {name:12} {stars}  ({escape_markdown(explain)})")
    
    return "\n".join(lines)


def format_no_trade_report(scores: dict, reasons: dict, total_score: int,
                           indicators: dict, intraday_levels: dict,
                           relative_context: dict,
                           threshold: int = 4,
                           filter_reason: str = "") -> str:
    """
    Builds the full NO_TRADE report with a detailed explanation.
    """
    lines = []
    
    # 1. Header. This used to print "Confidence: 62%" — but there is no trade,
    # and confidence in a non-existent trade means nothing. We show signal strength
    # relative to the threshold: you can see right away how far the entry is.
    lines.append(f"❌ *NO TRADE*")
    if abs(total_score) >= threshold:
        # The score PASSED the threshold — the trade was rejected by the entry filter (R:R, volume, volatility).
        # Previously this printed "missing -1", because the formula silently
        # assumed NO_TRADE only happens from a lack of points.
        lines.append(
            f"Score: {total_score:+d} of ±{MAX_TOTAL_SCORE} — the ±{threshold} threshold is met, "
            f"but the trade was rejected by the entry filter"
        )
        if filter_reason:
            lines.append(f"⛔ Reason: {escape_markdown(filter_reason)}")
        lines.append("")
    else:
        lines.append(
            f"Score: {total_score:+d} of ±{MAX_TOTAL_SCORE} "
            f"(entry threshold ±{threshold}) — {threshold - abs(total_score)} short\n"
        )
    
    # 2. Dominant factor
    factor_name, factor_score, factor_reason = get_dominant_factor(scores, reasons)
    if factor_name:
        emoji = "🔴" if factor_score < 0 else "🟢" if factor_score > 0 else "⚪"
        lines.append(f"🎯 *Main reason:*")
        lines.append(f"   {emoji} {factor_name} ({factor_score:+d}) — {escape_markdown(factor_reason)}\n")
    
    # 3. Sort factors by strength
    sorted_items = sorted(
        [(k, v) for k, v in scores.items() if k != "total"],
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    factor_names = {
        "technical": "Technical",
        "news": "News",
        "stumpy": "STUMPY",
        "vwap": "VWAP"
    }

    # Split into For/Against/Neutral
    for_factors = []
    against_factors = []
    neutral_factors = []
    
    # Determine the maximum absolute value for the scale
    max_abs = max([abs(v) for k, v in scores.items() if k != "total"] + [1])
    
    for key, value in sorted_items:
        name = factor_names.get(key, key)
        reason = reasons.get(key, "")
        bar = format_factor_bar(value, max_abs)
        
        # Detail for composite factors
        details = ""
        if key == "technical":
            details = format_factor_details(scores, reasons, key)
        elif key == "news":
            details = format_factor_details(scores, reasons, key)
        
        if details:
            reason = f"{reason} ({details})" if reason else details
        
        # Escape reason before inserting
        reason_escaped = escape_markdown(reason)
        
        if value > 0:
            for_factors.append(f"   🟢 {name:12} {bar}  (+{value}) — {reason_escaped}")
        elif value < 0:
            against_factors.append(f"   🔴 {name:12} {bar}  ({value}) — {reason_escaped}")
        else:
            neutral_factors.append(f"   ⚪ {name:12} {bar}  (0) — {reason_escaped}")
    
    if for_factors:
        lines.append("🟢 *For:*")
        lines.extend(for_factors)
        lines.append("")
    
    if against_factors:
        lines.append("🔴 *Against:*")
        lines.extend(against_factors)
        lines.append("")
    
    if neutral_factors:
        lines.append("⚪ *Neutral:*")
        lines.extend(neutral_factors)
        lines.append("")
    
    # 4. What needs to change (detailed)
    lines.append("💡 *What needs to change:*")
    conditions = generate_entry_conditions_detailed(scores, reasons, "NO_TRADE", threshold)
    lines.append(conditions)
    lines.append("")
    
    # 5. Market quality (deterministic)
    quality = compute_market_quality_detailed(indicators, intraday_levels, scores, relative_context)
    lines.append("📈 *Market Quality:*")
    lines.append(format_market_quality_detailed(quality))
    
    return "\n".join(lines)