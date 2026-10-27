from config import NEWS_TEXT_MAX_CHARS, MAX_LINE_LOG
from data.fetcher import format_relative_context, bars_with_volume
from data.news_filter import filter_relevant_news


def build_junior_prompt(ticker, indicators, news, relative_context,
                        intraday_levels=None, volume_profile=None, 
                        stumpy_text="", value_text="",
                        news_analysis=None):
    rsi = indicators['rsi']
    position = relative_context.get('position_in_range_pct', 50)
    volume_ratio = relative_context.get('volume_ratio', 1)
    
    hints = []
    if position is not None and position > 90: hints.append(f"Price is at {position:.0f}% of the yearly range — an all-time high zone.")
    elif position is not None and position < 10: hints.append(f"Price is at {position:.0f}% of the yearly range — near the bottom.")
    if rsi > 70: hints.append(f"RSI={rsi} — overbought zone.")
    elif rsi < 30: hints.append(f"RSI={rsi} — oversold zone.")
    if volume_ratio is not None and volume_ratio < 0.5: hints.append(f"Volume {volume_ratio:.1f}x (full-day projection) — below normal.")
    elif volume_ratio is not None and volume_ratio > 1.5: hints.append(f"Volume {volume_ratio:.1f}x (full-day projection) — elevated.")
    
    hints_text = "\n\n".join(f"💡 CONTEXT: {h}" for h in hints) if hints else ""
    
    # COMPUTE THE POSITION INSIDE THE BOLLINGER BANDS
    bb_lower = indicators.get('bb_lower', 0)
    bb_upper = indicators.get('bb_upper', 0)
    price = indicators.get('price', 0)
    
    if bb_upper != bb_lower:
        bb_position = (price - bb_lower) / (bb_upper - bb_lower) * 100
    else:
        bb_position = 50.0
    
    if bb_position > 80:
        bb_status = f"near_upper ({bb_position:.0f}% of height)"
    elif bb_position < 20:
        bb_status = f"near_lower ({bb_position:.0f}% of height)"
    else:
        bb_status = f"middle ({bb_position:.0f}% of height)"
    
    # 🔥 NEWS BLOCK
    if news_analysis:
        sentiment = news_analysis.get("overall_sentiment", "neutral")
        summary = news_analysis.get("summary", "")
        themes = ", ".join(news_analysis.get("key_themes", []))
        catalysts = ", ".join(news_analysis.get("catalysts", []))
        risks = ", ".join(news_analysis.get("risk_flags", []))
        
        sentiment_emoji = {
            "positive": "🟢", "neutral": "⚪", "negative": "🔴", "mixed": "🟡"
        }.get(sentiment, "⚪")
        
        news_block = f"""
NEWS BACKGROUND (agent analysis) {sentiment_emoji}:
• Sentiment: {sentiment}
• Summary: {summary}
• Key themes: {themes}
• Catalysts: {catalysts}
• Risks: {risks}
"""
    else:
        from data.news_filter import filter_relevant_news
        filtered_news = filter_relevant_news(ticker, news, max_news=5)
        news_block = "\n\n".join([f"- {n['title']}\n  {n['text'][:600]}" for n in filtered_news]) or "none"
    
    trader_block = ""
    trend_divergence = False
    if intraday_levels:
        vwap_pos = intraday_levels.get('vwap_position', '?')
        dist_vwap = intraday_levels.get('distance_to_vwap', 0)
        dist_r1 = intraday_levels.get('distance_to_r1', 0)
        dist_s1 = intraday_levels.get('distance_to_s1', 0)
        trend_5m = intraday_levels.get('trend_5m', '?')
        trend_15m = intraday_levels.get('trend_15m', '?')
        trend_divergence = intraday_levels.get('trend_divergence', False)
        trend_warning = " ⚠️ DIVERGENCE — mention this in the summary" if trend_divergence else ""

        trader_block = f"""
TRADER LEVELS:
• VWAP: ${intraday_levels.get('vwap', '?')} (price {vwap_pos}, {dist_vwap:+.2f}%)
• Pivot: ${intraday_levels.get('pivot', '?')}
• R1: ${intraday_levels.get('r1', '?')} (distance: {dist_r1:+.2f}%)
• S1: ${intraday_levels.get('s1', '?')} (distance: {dist_s1:+.2f}%)
• ATR: ${intraday_levels.get('atr_hourly', '?')}
• Trend 5m: {trend_5m} | 15m: {trend_15m}{trend_warning}
"""
    if volume_profile:
        trader_block += f"• POC (max volume): ${volume_profile.get('poc', '?')}\n"
    
    stumpy_block = ""
    if stumpy_text:
        stumpy_block = f"\n{stumpy_text}\n"
    
    value_block = ""
    if value_text:
        value_block = f"\n{value_text}\n"

    return f"""You are a financial researcher. Interpret the data on {ticker} for day trading.

PRICE: {indicators['price']} ({indicators['price_change']:+.2f}%) | RSI: {indicators['rsi']}
MACD: {indicators['macd']} / signal: {indicators['macd_signal']}
BB: {indicators['bb_lower']} — {indicators['bb_upper']} (price at {bb_status}) | Volume: {indicators['volume']}

{format_relative_context(relative_context)}
{trader_block}
{stumpy_block}
{value_block}
{hints_text}

{news_block}

Give a SHORT summary (4-5 sentences), list risks (2-3) and opportunities (2-3) for day trading.
{"⚠️ The 5m and 15m trends diverge (see above) — be sure to mention this in the summary or risks." if trend_divergence else ""}
ANSWER WITH JSON ONLY, in English."""


def build_extractor_prompt(indicators, intraday_levels, relative_context, tags, orderflow, earnings_info, raw_rag_chunks):
    """The extractor pulls rules from books and does NOT use market data."""
    print(f"   🔍 DEBUG EXTRACTOR:")
    print(f"      raw_rag_chunks type: {type(raw_rag_chunks)}")
    print(f"      raw_rag_chunks is None: {raw_rag_chunks is None}")
    print(f"      raw_rag_chunks is empty string: {raw_rag_chunks == ''}")
    if raw_rag_chunks:
        print(f"      raw_rag_chunks length: {len(raw_rag_chunks)}")
        print(f"      raw_rag_chunks first 100 chars: {str(raw_rag_chunks)[:MAX_LINE_LOG]}")
    
    if not raw_rag_chunks:
        print(f"   ⚠️ raw_rag_chunks is None or empty, extractor not run")
        return None
    
    if isinstance(raw_rag_chunks, str) and len(raw_rag_chunks.strip()) == 0:
        print(f"   ⚠️ raw_rag_chunks is empty string, extractor not run")
        return None
    
    if isinstance(raw_rag_chunks, str) and len(raw_rag_chunks.strip()) < 10:
        print(f"   ⚠️ raw_rag_chunks too short ({len(raw_rag_chunks)} chars), extractor not run")
        return None
    
    print(f"   ✅ raw_rag_chunks passed all checks, building prompt...")

    prompt = f"""You are a book-rule analyser. Your task is to extract trading rules, patterns and warnings from the given book excerpts.

⚠️ CRITICAL RULES:
1. Do NOT use market data (VWAP, RSI, MACD, POC, support/resistance levels, current price, volume, etc.)
2. Do NOT invent your own levels, numbers or patterns
3. Do NOT refer to specific indicators
4. If an excerpt contains no rule — skip it
5. Extract ONLY general principles and ideas from the books

RULES FROM BOOKS (raw excerpts):
{raw_rag_chunks[:4000]}

YOUR TASK — extract the essence:

1. **pattern** — which model/pattern do the books describe? (1-2 words, e.g. "false breakout", "trailing stop", "position management")

2. **rules** — concrete rules that can be applied to any trading situation:
   - E.g.: "Do not place the stop too close to a critical level"
   - E.g.: "After the target is reached, trail the protective stop"
   - E.g.: "Do not add more than 2 units to a position at a time"

3. **warnings** — warnings from the books:
   - E.g.: "A false breakout of the level is possible before a reversal"
   - E.g.: "Do not wait for a reversal after the stop is violated"

RESPONSE FORMAT (JSON ONLY):
{{
  "pattern": "pattern name",
  "rules": ["rule 1", "rule 2"],
  "warnings": ["warning 1"]
}}

IMPORTANT:
- Do NOT INVENT rules that are not in the text
- Do NOT use market data
- If the text has no rules — return empty arrays

ANSWER WITH JSON ONLY.
"""
    
    print(f"   ✅ Prompt built: {len(prompt)} chars")
    return prompt


def build_trader_prompt(ticker, indicators, intraday_levels, volume_profile, liquidity, junior_analysis, news, extracted_rules=None, stumpy_text="", value_text="", news_analysis=None, direction_hint=None, total_score=None):
    """
    Builds the prompt for the Trader.
    
    Args:
        direction_hint: hint from the signal system (STRONG_LONG/LONG/NO_TRADE/SHORT/STRONG_SHORT)
        total_score: total score from the signal system
    """
    levels = intraday_levels
    vp = volume_profile
    liq = liquidity
    price = indicators['price']
    
    # NEWS BLOCK
    if news_analysis:
        sentiment = news_analysis.get("overall_sentiment", "neutral")
        summary = news_analysis.get("summary", "")
        themes = ", ".join(news_analysis.get("key_themes", []))
        catalysts = ", ".join(news_analysis.get("catalysts", []))
        risks = ", ".join(news_analysis.get("risk_flags", []))
        
        sentiment_emoji = {
            "positive": "🟢", "neutral": "⚪", "negative": "🔴", "mixed": "🟡"
        }.get(sentiment, "⚪")
        
        news_block = f"""
NEWS BACKGROUND (agent analysis) {sentiment_emoji}:
• Sentiment: {sentiment}
• Summary: {summary}
• Key themes: {themes}
• Catalysts: {catalysts}
• Risks: {risks}
"""
    else:
        filtered_news = filter_relevant_news(ticker, news, max_news=5)
        news_items = [f"* {n['title']}" for n in filtered_news]
        if news_items:
            news_block = "NEWS:\n" + "\n".join(news_items) + "\n"
        else:
            news_block = ""
    
    # ═══════════════════════════════════════════
    # RULES FROM BOOKS (NEW STRUCTURE)
    # ═══════════════════════════════════════════
    rules_block = ""
    if extracted_rules and extracted_rules.get('rules'):
        pattern = extracted_rules.get('pattern', '')
        rules_list = extracted_rules.get('rules', [])
        warnings_list = extracted_rules.get('warnings', [])
        
        rules_block = f"""
📝 *RULES EXTRACTED FROM BOOKS:*
• Pattern: {pattern}
• Rules:
"""
        for rule in rules_list:
            rules_block += f"  - {rule}\n"
        
        if warnings_list:
            rules_block += f"• Warnings:\n"
            for warning in warnings_list:
                rules_block += f"  - {warning}\n"
    
    stumpy_block = ""
    if stumpy_text:
        stumpy_block = f"\n{stumpy_text}\n"
    
    value_block = ""
    if value_text:
        value_block = f"\n{value_text}\n"
    
    # ═══════════════════════════════════════════
    # HINT FROM THE SIGNAL SYSTEM
    # ═══════════════════════════════════════════
    hint_block = ""
    if direction_hint and direction_hint != "NO_TRADE":
        hint_block = f"""
💡 *SIGNAL SYSTEM (Python):*
• Direction: {direction_hint}
• Total score: {total_score:+d}
• Take this into account when choosing a direction, but do not copy it blindly.
• If your analysis contradicts the system — explain why.

"""
    elif direction_hint == "NO_TRADE":
        hint_block = f"""
💡 *SIGNAL SYSTEM (Python):*
• Direction: NO_TRADE (score: {total_score:+d})
• Signals are contradictory — do not open a trade.

"""
    
    # ═══════════════════════════════════════════
    # DATA PREPARATION FOR THE RULES
    # ═══════════════════════════════════════════
    pivot_val = levels.get('pivot', '?')
    s1_val = levels.get('s1', '?')
    poc_val = vp.get('poc', '?')
    vwap_val = levels.get('vwap', '?')
    r1_val = levels.get('r1', '?')
    r2_val = levels.get('r2', '?')
    prev_high_val = liq.get('prev_high', '?')
    prev_low_val = liq.get('prev_low', '?')
    trend_5m = levels.get('trend_5m', '?')
    trend_15m = levels.get('trend_15m', '?')
    trend_divergence = levels.get('trend_divergence', False)

    return f"""You are a professional day trader. The decision on {ticker} HAS ALREADY BEEN MADE by the system
(direction, entry/stop/target levels and position size are computed deterministically
in Python). Your task is NOT to make a decision and NOT to propose your own plan, but to give
expert commentary on the decision already made.

{news_block}{rules_block}{stumpy_block}{value_block}
MARKET DATA (all numbers are already computed, use them):
* Current price: ${price:.2f}
* VWAP: ${vwap_val:.2f} (price {'ABOVE' if levels.get('price_above_vwap') else 'BELOW'} VWAP)
* Pivot: ${pivot_val:.2f}
* R1: ${r1_val:.2f} | R2: ${r2_val:.2f}
* S1: ${s1_val:.2f} | S2: ${levels.get('s2', '?'):.2f}
* ATR (hourly): ${levels.get('atr_hourly', '?'):.2f}
* POC (max volume): ${poc_val:.2f}
* Trend 5m: {trend_5m} | 15m: {trend_15m}{' ⚠️ DIVERGENCE (5m and 15m point in different directions)' if trend_divergence else ''}

LIQUIDITY:
* Stops above: {liq.get('stops_above', []) if liq.get('stops_above') else 'none'}
* Stops below: {liq.get('stops_below', []) if liq.get('stops_below') else 'none'}
* Prev. day High: ${prev_high_val} | Low: ${prev_low_val}

JUNIOR CONTEXT:
{junior_analysis.get('executive_summary', '')[:5000]}

{hint_block}
FILL IN EXACTLY THREE FIELDS:

1. historical_analogy — a similar situation from market history (1-2 sentences).
   If no analogy comes to mind — a short "no clear analogy", do not make one up.

2. risks — 2-4 SPECIFIC risks for exactly this situation, not generic words.
   Rely on the data above: contradictions between factors, proximity of levels,
   earnings, volume, company valuation.{" BE SURE to include the 5m/15m trend divergence (see ⚠️ above) as one of the risks." if trend_divergence else ""}

3. commentary — 3-5 sentences: what is happening with the stock, which factors
   support the system's decision and which contradict it, and what to watch
   next (what confirmation would make the picture clearer).

⚠️ IMPORTANT — WHAT NOT TO DO:
- Do NOT propose your own direction, entry, stop, targets or position size — they are already
  computed, your plan will not be used anyway.
- Do NOT argue with the system's decision and do not write "I must give a plan" — you don't have to.
  If you think the decision is debatable, just explain in commentary what exactly
  contradicts it and what confirmation is missing.
- Do NOT invent numbers that are not in the data above.

ANSWER WITH JSON ONLY, IN ENGLISH.
"""

def build_librarian_prompt(ticker, analysis, raw_rag_chunks):
    junior_summary = analysis.get("executive_summary", "")
    print(f"\n   📥 LIBRARIAN: received {len(raw_rag_chunks)} raw excerpts:")
    numbered_previews = []
    for i, chunk in enumerate(raw_rag_chunks[:20]):
        preview = chunk[:MAX_LINE_LOG].replace('\n', ' ')
        numbered_previews.append(f"[{i+1}] {preview}...")
        print(f"      [{i+1}] {preview[:MAX_LINE_LOG]}...")
    chunks_text = "\n\n".join(numbered_previews)
    
    prompt = f"""You are a librarian. Pick 3-5 excerpt numbers about technical analysis ONLY: candles, levels, trends.

REMOVE excerpts about:
- statistics, gamma functions, distributions, formulas, Excel
- generic talk about investing, pensions, bonds
- mathematics, probability theory

KEEP excerpts about:
- RSI, MACD, stochastic, oscillators
- candlestick patterns (doji, hammer, engulfing, star)
- support/resistance levels
- trends, reversals
- VWAP, volume analysis

CONTEXT: {junior_summary[:300]}

EXCERPTS:
{chunks_text}

Return JSON ONLY: {{"selected": [1, 4, 7]}}"""
    
    return prompt


def build_news_prompt(ticker, news):
    """Builds the prompt for the news analyst."""
    news_block = "\n\n".join(
        f"- {n['title']}\n  {n['text'][:5000]}" for n in news
    ) or "no news found"

    prompt = f"""You are a financial analyst. Your task is to write a DETAILED review of the news background
for {ticker} for a trader deciding whether to enter a position.

NEWS:
{news_block}

INSTRUCTIONS:
1. overall_sentiment – overall tone (positive/neutral/negative/mixed).
2. summary – a SUMMARY of 5-8 sentences: what is really happening with the company? Highlight the main events,
   their possible impact on the price, the market reaction, investor mood. Write coherently, like an analytical article.
3. key_themes – 2-4 main themes (e.g. "regulatory pressure", "weak earnings").
4. detailed_analysis – FOR EACH key theme write 2-4 sentences: what it is about,
   which news items relate to it, what risks or opportunities it creates, how it may
   affect the price in the coming days. Connect different news items with each other.
5. catalysts – specific catalyst events (earnings release, court ruling, deal, product launch)
   that can move the price. Give the event date if known.
6. risk_flags – specific warning signs (falling revenue, customer churn, sanctions, lawsuits).
   If there is nothing obvious – leave an empty list.
7. Facts (contract, revenue guidance, earnings date, regulatory decision) — only if confirmed by sources.

Do NOT INVENT events that are not in the texts.
ANSWER STRICTLY WITH JSON in English.
"""

    # Debug output (the prompt is defined now)
    print(f"\n   📝 FULL NEWS ANALYST PROMPT:")
    print(f"   {'='*60}")
    print(f"   Ticker: {ticker}")
    print(f"   Number of news items: {len(news)}")
    print(f"   Prompt length: {len(prompt)} chars")
    print(f"   {'='*60}")
    print(f"   {prompt[:8000]}...")
    print(f"   {'='*60}")

    return prompt

def build_market_prompt(snapshot, news):
    """
    Market overview prompt.

    NUMBERS ARE GIVEN READY-MADE and the model must only USE them. This is the
    same principle as in the trader prompt: give the model
    a place where it can compute something itself, and it will, and the report
    ends up with two different numbers for the same thing. That is why MARKET_SCHEMA has
    no numeric fields at all, and there is an explicit prohibition here.
    """
    idx_block = "\n".join(
        f"- {i['name']}: {i['price']:,.2f}"
        + (f" ({i['change_pct']:+.2f}% for the session)" if i.get("change_pct") is not None else "")
        for i in snapshot.get("indices", [])
    ) or "no data"

    vix = snapshot.get("vix", {})
    vix_line = (f"- VIX (fear index): {vix['value']:.2f} — level \"{vix['level']}\""
                if vix.get("value") is not None else "- VIX: no data")

    y = snapshot.get("yield_10y", {})
    yield_line = (f"- US 10-year Treasury yield: {y['value']:.2f}%"
                  + (f" ({y['change_pct']:+.1f}% for the session)" if y.get("change_pct") is not None else "")
                  if y.get("value") is not None else "- 10-year yield: no data")

    sectors = snapshot.get("sectors", [])
    breadth = snapshot.get("breadth", {})
    sec_block = ", ".join(f"{s['name']} {s['change_pct']:+.1f}%" for s in sectors) \
        or "no data"
    breadth_line = (f"- Breadth: {breadth.get('up', 0)} sectors up out of "
                    f"{breadth.get('total', 0)}")

    news_block = "\n".join(
        f"- {n['title']}" + (f"  [{n['source']}]" if n.get("source") else "")
        for n in news
    ) or "no headlines"

    status_line = ("The trading session is in progress."
                   if snapshot.get("market_status") == "open"
                   else "The market is closed, figures are for the last completed session.")

    prompt = f"""You are a market analyst. Task: explain to a trader what is happening
in the market right now — as if you had carefully read the front page
of a financial portal and pulled it all together.

{status_line}

FIGURES (computed in advance, take them as they are):
{idx_block}
{vix_line}
{yield_line}
- Sectors by change over the session: {sec_block}
{breadth_line}

NEWS HEADLINES FROM THE LAST FEW DAYS:
{news_block}

INSTRUCTIONS:
1. overall_tone — one of: risk_on (risk appetite, money flowing into stocks),
   risk_off (flight to safety), mixed (diverging), neutral (no clear tone).
   Rely on the WHOLE picture: index direction, VIX level, sector breadth,
   which sectors lead (cyclical vs. defensive), bond yields.
2. summary — COHERENT text of 6-10 sentences: what drives the market today, why
   it moves the way it does, what is behind the move. Write it as an analytical note,
   not a list. Be sure to connect the news with the figures given.
3. key_events — 3-6 specific events from the headlines that actually move the market
   (Fed decision, large-company earnings, macro data, geopolitics).
4. themes — 2-4 cross-cutting themes (e.g. "rates", "AI capex", "oil").
5. detailed_analysis — for each theme 2-4 sentences: the essence, which headlines
   relate to it, who is affected, how it ties in with index and sector moves.
6. risks — 2-5 specific risks visible from the news and figures.
7. what_to_watch — 2-5 items: what happens next and what to watch (meeting
   dates, data releases, expected earnings) — only if it follows from the headlines.

FORBIDDEN:
- Inventing or recomputing percentages, index levels, VIX values and yields.
  All numbers are already given above — use EXACTLY THOSE and do not add your own.
- Inventing events that are not in the headlines.
- Giving trading recommendations ("buy", "sell"). Your job is to describe
  the situation; a human makes the decision.

ANSWER STRICTLY WITH JSON in English.
"""

    print(f"\n   📝 MARKET OVERVIEW PROMPT: {len(prompt)} chars, "
          f"{len(news)} headlines, {len(sectors)} sectors")
    return prompt


def format_news_report(ticker, analysis, news):
    sentiment_emoji = {
        "positive": "🟢", "neutral": "⚪", "negative": "🔴", "mixed": "🟡"
    }.get(analysis.get("overall_sentiment"), "⚪")

    lines = [f"📰 *NEWS REVIEW — {ticker}* {sentiment_emoji}\n"]
    lines.append(analysis.get("summary", ""))

    themes = analysis.get("key_themes", [])
    if themes:
        lines.append(f"\n*Key themes:* {', '.join(themes)}")

    detailed = analysis.get("detailed_analysis", "")
    if detailed:
        lines.append(f"\n📝 *Detailed analysis:*\n{detailed}")

    catalysts = analysis.get("catalysts", [])
    if catalysts:
        lines.append(f"\n⚡ *Catalysts:*")
        for c in catalysts:
            lines.append(f"  • {c}")

    risks = analysis.get("risk_flags", [])
    if risks:
        lines.append(f"\n⚠️ *Risk flags:*")
        for r in risks:
            lines.append(f"  • {r}")

    lines.append(f"\n📋 *Sources ({len(news)}):*")
    for n in news:
        lines.append(f"  • {n['title']}")

    return "\n".join(lines)