from config import TRADER_MODEL, MAX_LINE_LOG, LONG_THRESHOLD, SHORT_THRESHOLD
from utils.schemas import TRADER_SCHEMA
from utils.ollama import call_ollama_json
from agents.prompts import build_trader_prompt
from utils.signals import (
    compute_factor_scores,
    determine_direction,
    check_entry_filters,
    calculate_position_size,
    compute_confidence,
    format_signals,
    calculate_entry_levels,
    format_no_trade_report,
    get_dominant_factor,
    format_factor_bar,
    format_factor_details,
    generate_entry_conditions_detailed,
    compute_market_quality_detailed,
    format_market_quality_detailed
)
from utils.helpers import safe_format_float


def run_trader(ticker, indicators, intraday_levels, volume_profile, liquidity, analysis, news, extracted_rules=None, stumpy_text="", value_text="", news_analysis=None, relative_context=None, stumpy_result=None, earnings_info=None, real_flow=None, short_vol=None):
    """
    Runs the trader.

    Gemma → description only (historical analogy, risks, commentary)
    Python → final decision (direction, levels, position size)
    """
    print(f"\n📈 [7/7] DAY TRADER")

    # ═══════════════════════════════════════════
    # 1. EXTRACT STUMPY
    # ═══════════════════════════════════════════
    # These numbers used to be pulled by regex out of stumpy_text (a ready-made
    # multi-horizon report: "in 1 day / 3 days / 5 days" blocks one after
    # another). re.search takes the FIRST match of "Win rate:"/"Average
    # result:" — and the first one in the text is the "in 1 day" horizon, not 3 —
    # although the 3-day horizon is treated as the main one everywhere (see the final
    # "Conclusion" in data/stumpy_finder.py::format_stumpy_result, it is built on
    # win_rate_3d). Because of this, 1-day window figures leaked into scoring
    # (e.g. win_rate 55% instead of the real 65% on the 3-day). Now we take
    # the structured STUMPY result directly — no text parsing.
    stumpy_data = {}
    if stumpy_result:
        if stumpy_result.get('win_rate_3d') is not None:
            stumpy_data['win_rate'] = stumpy_result['win_rate_3d']
        if stumpy_result.get('avg_ret_3d') is not None:
            stumpy_data['avg_return'] = stumpy_result['avg_ret_3d']
        if stumpy_result.get('n_3d') is not None:
            stumpy_data['n_samples'] = stumpy_result['n_3d']
    
    # ═══════════════════════════════════════════
    # 2. PYTHON — COMPUTE FACTORS AND SCORES
    # ═══════════════════════════════════════════
    relative_context = relative_context or {}

    scores, reasons = compute_factor_scores(
        indicators=indicators,
        intraday_levels=intraday_levels,
        news_analysis=news_analysis,
        stumpy_data=stumpy_data,
        relative_context=relative_context
    )
    
    direction, total_score = determine_direction(scores)

    print(f"\n   {format_signals(scores, reasons)}")
    print(f"   🎯 Direction: {direction} (score: {total_score:+d})")

    # Snapshot of the RAW inputs the scores are built from. Scores say "news +2",
    # but not at what sentiment and how many risk flags that came out —
    # and that is exactly what is needed later to check whether a factor deserves its weight.
    # Without a snapshot, factor attribution in hindsight is impossible: historical
    # inputs cannot be recomputed, yfinance does not return past news context.
    # A local copy rather than reassigning the argument: news_analysis below
    # goes into build_trader_prompt, and replacing None with {} would change what
    # the model sees.
    _news = news_analysis or {}
    _rf = real_flow or {}
    # real_flow (utils/real_flow.py, Alpaca/IEX) is NOT part of compute_factor_scores,
    # only a snapshot for a future counterfactual measurement (see the file header).
    # We take the middle window (15m): the shortest (5m) is too noisy on
    # illiquid names, the longest (30m) blurs what has just
    # happened.
    _rf15 = _rf.get("flow_15m") or {}
    # short_vol (utils/short_volume.py, FINRA Reg SHO) — same logic: only
    # diagnostics in features, does not take part in compute_factor_scores.
    _sv = short_vol or {}
    features = {
        "price": indicators.get("price"),
        "rsi": indicators.get("rsi"),
        "macd": indicators.get("macd"),
        "macd_signal": indicators.get("macd_signal"),
        "bb_upper": indicators.get("bb_upper"),
        "bb_lower": indicators.get("bb_lower"),
        "volume_ratio": relative_context.get("volume_ratio"),
        "position_in_range_pct": relative_context.get("position_in_range_pct"),
        "vwap_position": intraday_levels.get("vwap_position"),
        "distance_to_vwap": intraday_levels.get("distance_to_vwap"),
        "trend_5m": intraday_levels.get("trend_5m"),
        "trend_15m": intraday_levels.get("trend_15m"),
        "atr_hourly": intraday_levels.get("atr_hourly"),
        "news_sentiment": _news.get("overall_sentiment"),
        "news_catalysts_n": len(_news.get("catalysts", [])),
        "news_risk_flags_n": len(_news.get("risk_flags", [])),
        "stumpy_win_rate": stumpy_data.get("win_rate"),
        "stumpy_avg_return": stumpy_data.get("avg_return"),
        "stumpy_n": stumpy_data.get("n_samples"),
        "real_flow_source": _rf.get("source"),
        "real_flow_n_trades_15m": _rf15.get("n_trades"),
        "real_flow_delta_pct_15m": _rf15.get("delta_pct"),
        "real_flow_delta_quote_15m": _rf15.get("delta_quote"),
        "short_ratio": _sv.get("short_ratio"),
        "short_volume_date": _sv.get("date"),
    }
    
    # ═══════════════════════════════════════════
    # 3. GEMMA — DESCRIPTION ONLY
    # ═══════════════════════════════════════════
    trader_prompt = build_trader_prompt(
        ticker, indicators, intraday_levels, volume_profile, liquidity,
        analysis, news, extracted_rules, stumpy_text, value_text,
        news_analysis=news_analysis,
        direction_hint=direction,
        total_score=total_score
    )
    
    print(f"   📝 PROMPT: {len(trader_prompt)} chars")
    for line in trader_prompt.split('\n'):
        print(f"   │ {line[:MAX_LINE_LOG]}")
    print("   🚀 SENDING REQUEST TO OLLAMA...")
    
    # Gemma returns the description only
    gemma_output = call_ollama_json(trader_prompt, TRADER_MODEL, json_schema=TRADER_SCHEMA, temperature=0)
    
    print(f"\n   📖 *HISTORICAL ANALOGY:*")
    print(f"   │ {gemma_output.get('historical_analogy', '')[:MAX_LINE_LOG]}")
    print(f"\n   ⚠️ *RISKS:*")
    for risk in gemma_output.get('risks', []):
        print(f"   │ • {risk[:MAX_LINE_LOG]}")
    print(f"\n   💬 *COMMENTARY:*")
    print(f"   │ {gemma_output.get('commentary', '')[:MAX_LINE_LOG]}")
    
    # ═══════════════════════════════════════════
    # 4. PYTHON — LEVEL CALCULATION (FIRST)
    # ═══════════════════════════════════════════
    levels = calculate_entry_levels(
        direction=direction,
        price=indicators['price'],
        intraday_levels=intraday_levels,
        volume_profile=volume_profile,
        liquidity=liquidity
    )
    
    # ═══════════════════════════════════════════
    # 5. ENTRY FILTER CHECK (with real levels)
    # ═══════════════════════════════════════════
    entry_ok, entry_reason = check_entry_filters(
        direction=direction,
        intraday_levels=intraday_levels,
        trader_plan=levels,
        indicators=indicators,
        relative_context=relative_context,
        earnings_info=earnings_info
    )
    
    if not entry_ok:
        print(f"\n   ⚠️ FILTER NOT PASSED: {entry_reason}")
        # If the filter is not passed, turn it into NO_TRADE
        report = format_no_trade_report(
            scores=scores,
            reasons=reasons,
            total_score=total_score,
            indicators=indicators,
            intraday_levels=intraday_levels,
            relative_context=relative_context,
            threshold=LONG_THRESHOLD,
            filter_reason=entry_reason
        )
        print(f"\n{report}")
        return {
            "direction": "NO_TRADE",
            "direction_score": total_score,
            "scores": scores,
            "reasons": reasons,
            "features": features,
            "gemma": gemma_output,
            "entry_price": None,
            "stop_loss": None,
            "target_1": None,
            "target_2": None,
            "position_size_pct": 0,
            "confidence": 0.0,
            "entry_filter_reason": entry_reason,
            "reasoning": f"Entry filter not passed: {entry_reason}",
            "no_trade_report": report
        }
    
    # ═══════════════════════════════════════════
    # 6. POSITION SIZE AND CONFIDENCE
    # ═══════════════════════════════════════════
    position_size = calculate_position_size(direction, total_score, entry_ok)
    confidence = compute_confidence(direction, total_score, entry_ok)
    
    # ═══════════════════════════════════════════
    # 7. OUTPUT
    # ═══════════════════════════════════════════
    print(f"\n   ✅ DIRECTION: {direction} (score: {total_score:+d})")
    print(f"   ✅ ENTRY: ${safe_format_float(levels.get('entry_price'))} | STOP: ${safe_format_float(levels.get('stop_loss'))}")
    print(f"   ✅ TARGETS: ${safe_format_float(levels.get('target_1'))} / ${safe_format_float(levels.get('target_2'))}")
    print(f"   ✅ R:R 1:{levels.get('risk_reward_1', '?')} / 1:{levels.get('risk_reward_2', '?')}")
    print(f"   ✅ POSITION: {position_size}% | CONFIDENCE: {confidence:.0%}")
    
    return {
        "direction": direction,
        "direction_score": total_score,
        "scores": scores,
        "reasons": reasons,
        "features": features,
        "gemma": gemma_output,
        "entry_price": levels.get('entry_price'),
        "stop_loss": levels.get('stop_loss'),
        "target_1": levels.get('target_1'),
        "target_2": levels.get('target_2'),
        "risk_reward_1": levels.get('risk_reward_1', 0),
        "risk_reward_2": levels.get('risk_reward_2', 0),
        "position_size_pct": position_size,
        "confidence": confidence,
        "reasoning": f"{direction}. Total score: {total_score:+d}."
    }