import re
from config import RAG_N_RESULTS, RAG_CHUNK_SIZE, MAX_LINE_LOG
from knowledge.embeddings import collection, RAG_ENABLED


def query_knowledge(query: str, n_results: int = RAG_N_RESULTS) -> str:
    if not RAG_ENABLED or collection is None:
        return ""
    try:
        results = collection.query(query_texts=[query], n_results=n_results * 2)
        docs = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
        if metadatas:
            filtered = [(doc, m) for doc, m in zip(docs, metadatas) if m.get('category', '') != 'garbage']
            categories = [m.get('category', '?') for _, m in filtered]
            print(f"   📚 Found: {len(docs)} in total, after the garbage filter: {len(filtered)} chunks, categories: {categories}")
            filtered = filtered[:n_results]
            return "\n\n".join([f"📖 [{m.get('category', '?')}] {doc}" for doc, m in filtered]) if filtered else ""
        return ""
    except Exception as e:
        print(f"   ⚠️ RAG query error: {e}")
        return ""


def query_knowledge_by_tags(tags, n_results=5):
    """Searches chunks by market-state tags."""
    if not RAG_ENABLED or collection is None or not tags:
        return ""
    
    all_results = []
    seen = set()
    
    for tag in tags:
        results = collection.query(query_texts=[tag], n_results=2)
        docs = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
        
        for doc, m in zip(docs, metadatas):
            if doc[:100] not in seen and m.get('category', '') != 'garbage':
                all_results.append((doc, m))
                seen.add(doc[:100])
    
    all_results = all_results[:n_results]
    
    if all_results:
        return "\n\n".join([f"📖 [{m.get('category', '?')}] {doc}" 
                           for doc, m in all_results])
    return ""


def score_chunk_for_tags(chunk_text, tags, chunk_category="", chunk_keywords=None):
    """Python scoring of a chunk by tags — without an LLM."""
    score = 0
    chunk_lower = chunk_text.lower()
    
    TAG_WEIGHTS = {
        "delta_divergence": 5,
        "bullish_divergence_complex": 5,
        "large_player_active": 4,
        "orderflow_bullish": 4,
        "orderflow_bearish": 4,
        "below_vwap": 3,
        "above_vwap": 3,
        "at_poc": 3,
        "above_poc": 3,
        "below_poc": 3,
        "downtrend": 3,
        "uptrend": 3,
        "macd_bearish": 2,
        "macd_bullish": 2,
        "macd_crossing": 3,
        "earnings_soon": 2,
        "earnings_imminent": 3,
        "low_volume": 2,
        "ultra_low_volume": 3,
        "high_volume": 2,
        "overbought": 2,
        "oversold": 2,
        "neutral_rsi": 1,
        "elevated_rsi": 1,
        "depressed_rsi": 2,
        "near_52w_high": 2,
        "near_52w_low": 2,
        "all_time_high": 3,
        "near_bb_upper": 2,
        "near_bb_lower": 2,
        "high_volatility": 2,
        "stops_above_cluster": 2,
        "stops_below_cluster": 2,
        "gap_up": 1,
        "gap_down": 1,
        "absorption": 4,
        "capitulation_possible": 3,
        "exhaustion_possible": 3,
    }
    
    for tag in tags:
        weight = TAG_WEIGHTS.get(tag, 1)
        if tag in chunk_lower:
            score += weight
        for word in tag.split('_'):
            if len(word) > 3 and word in chunk_lower:
                score += weight * 0.5
    
    if chunk_keywords:
        for tag in tags:
            tag_base = tag.replace("bullish_", "").replace("bearish_", "").replace("complex_", "").replace("orderflow_", "")
            for kw in chunk_keywords:
                if kw in tag_base or tag_base in kw:
                    score += 5
                    break
    
    # Real metadata["category"] values set by the distiller
    # (knowledge/distiller.py DISTILLER_SCHEMA) — entry/exit/risk/confirmation
    # are directly programmable, "context" is background information, not a signal.
    CATEGORY_BONUS = {
        "entry": 4,
        "exit": 4,
        "risk": 3,
        "confirmation": 2,
        "context": -2,
    }
    score += CATEGORY_BONUS.get(chunk_category, 0)
    
    if "contents" in chunk_lower or "table of contents" in chunk_lower:
        score -= 10
    if "chapter" in chunk_lower and len(chunk_text) < 300:
        score -= 5
    if "figure" in chunk_lower or "fig." in chunk_lower:
        score -= 1
    if "see fig" in chunk_lower or "see figure" in chunk_lower:
        score -= 1
    
    if chunk_text.startswith("[RULE]"):
        score += 3
    elif chunk_text.startswith("[WARNING]"):
        score += 2
    elif chunk_text.startswith("[IDEA]"):
        score += 1
    
    if len(chunk_text) > 100:
        score += 1
    if len(chunk_text) > 200:
        score += 1
    
    return score


def query_knowledge_by_tags_scored(tags, n_results=5):
    """Searches chunks by tags with Python scoring (without an LLM)."""
    if not RAG_ENABLED or collection is None or not tags:
        return ""
    
    all_results = []
    seen = set()
    
    for tag in tags:
        results = collection.query(query_texts=[tag], n_results=3)
        docs = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
        
        for doc, m in zip(docs, metadatas):
            doc_key = doc
            if doc_key not in seen and m.get('category', '') != 'garbage':
                keywords = m.get('keywords', [])
                score = score_chunk_for_tags(doc, tags, m.get('category', ''), keywords)
                all_results.append((doc, m, score))
                seen.add(doc_key)
    
    all_results.sort(key=lambda x: x[2], reverse=True)
    top_results = all_results[:n_results]
    
    if top_results:
        print(f"\n   📊 Python scoring: selected {len(top_results)} excerpts:")
        print(f"   {'─'*50}")
        for i, (doc, m, score) in enumerate(top_results):
            keywords = m.get('keywords', [])
            preview = doc[:MAX_LINE_LOG].replace('\n', ' ')
            print(f"   [{i+1}] score={score} | {m.get('category', '?')} | kw: {keywords[:5]}")
            print(f"   {preview[:MAX_LINE_LOG]}...")
            print(f"   {'─'*50}")
        
        return "\n\n".join([f"📖 [{m.get('category', '?')}] {doc}" 
                           for doc, m, _ in top_results])
    return ""


def compute_all_tags(indicators, intraday_levels, relative_context, earnings_info, orderflow=None, volume_profile=None, liquidity=None):
    """
    Collects ALL tags of the market state and of the specific stock.
    """
    tags = []
    rsi = indicators['rsi']
    price = indicators['price']
    macd_val = indicators['macd']
    macd_sig = indicators['macd_signal']
    bb_upper = indicators['bb_upper']
    bb_lower = indicators['bb_lower']
    volume_ratio = relative_context.get('volume_ratio', 1) if relative_context else 1
    position = relative_context.get('position_in_range_pct', 50) if relative_context else 50
    days = earnings_info.get('days_until') if earnings_info else None
    
    if rsi > 70: tags.append("overbought")
    elif rsi > 60: tags.append("elevated_rsi")
    elif rsi < 30: tags.append("oversold")
    elif rsi < 40: tags.append("depressed_rsi")
    else: tags.append("neutral_rsi")
    
    vwap_pos = intraday_levels.get('vwap_position', '') if intraday_levels else ''
    if vwap_pos == 'above': tags.append("above_vwap")
    elif vwap_pos == 'below': tags.append("below_vwap")
    else: tags.append("at_vwap")
    
    if macd_val > macd_sig: tags.append("macd_bullish")
    else: tags.append("macd_bearish")
    
    if abs(macd_val - macd_sig) < 0.05: tags.append("macd_crossing")
    
    if volume_ratio is not None:
        if volume_ratio < 0.3: tags.append("ultra_low_volume")
        elif volume_ratio < 0.5: tags.append("low_volume")
        elif volume_ratio > 2.0: tags.append("ultra_high_volume")
        elif volume_ratio > 1.5: tags.append("high_volume")
        else: tags.append("normal_volume")
    else:
        tags.append("normal_volume")
    
    trend_5m = intraday_levels.get('trend_5m', '') if intraday_levels else ''
    trend_15m = intraday_levels.get('trend_15m', '') if intraday_levels else ''
    if trend_5m == 'bearish' and trend_15m == 'bearish': tags.append("downtrend")
    elif trend_5m == 'bullish' and trend_15m == 'bullish': tags.append("uptrend")
    elif trend_5m != trend_15m: tags.append("trend_divergence")
    else: tags.append("sideways")
    
    if price >= bb_upper * 0.99: tags.append("near_bb_upper")
    elif price <= bb_lower * 1.01: tags.append("near_bb_lower")
    else: tags.append("middle_bb")
    
    if position is not None:
        if position > 95: tags.append("all_time_high")
        elif position > 90: tags.append("near_52w_high")
        elif position < 10: tags.append("near_52w_low")
        elif position < 5: tags.append("all_time_low")
    
    atr = intraday_levels.get('atr_hourly', 0) if intraday_levels else 0
    if atr and price > 0:
        atr_pct = atr / price * 100
        if atr_pct > 2: tags.append("high_volatility")
        elif atr_pct < 0.5: tags.append("low_volatility")
    
    if orderflow:
        if orderflow.get('orderflow_bias') == 'bullish': tags.append("orderflow_bullish")
        elif orderflow.get('orderflow_bias') == 'bearish': tags.append("orderflow_bearish")
        if orderflow.get('large_player_active'): tags.append("large_player_active")
        if orderflow.get('delta_divergence'): tags.append("delta_divergence")
        if orderflow.get('absorption_detected'): tags.append("absorption")
    
    if volume_profile:
        poc = volume_profile.get('poc', 0)
        if poc and price > 0:
            if price > poc * 1.01: tags.append("above_poc")
            elif price < poc * 0.99: tags.append("below_poc")
            else: tags.append("at_poc")
    
    if liquidity:
        stops_above = liquidity.get('stops_above', [])
        stops_below = liquidity.get('stops_below', [])
        if len(stops_above) >= 3: tags.append("stops_above_cluster")
        if len(stops_below) >= 3: tags.append("stops_below_cluster")
    
    if days is not None:
        if days <= 3: tags.append("earnings_imminent")
        elif days <= 7: tags.append("earnings_this_week")
        elif days <= 14: tags.append("earnings_soon")
        elif days <= 30: tags.append("earnings_this_month")
    
    price_change = indicators.get('price_change', 0)
    if price_change > 2: tags.append("strong_gap_up")
    elif price_change > 0.5: tags.append("gap_up")
    elif price_change < -2: tags.append("strong_gap_down")
    elif price_change < -0.5: tags.append("gap_down")
    
    if trend_5m == 'bearish' and trend_15m == 'bearish' and rsi < 40:
        tags.append("capitulation_possible")
    if trend_5m == 'bullish' and trend_15m == 'bullish' and rsi > 60:
        tags.append("exhaustion_possible")
    
    if orderflow and orderflow.get('delta_divergence') and macd_val < macd_sig:
        tags.append("bullish_divergence_complex")
    
    return list(set(tags))


def route_knowledge_search(tags):
    """
    Python decides WHAT to look for in RAG based on the tags.

    The "keywords" of each route are NOT Chroma categories (when loading books
    a completely different taxonomy is written into metadata "category" — entry/exit/risk/
    confirmation/context, see knowledge/distiller.py), but topics from TERM_MAP
    (knowledge/distiller.py: extract_keywords_from_chunk), which are actually
    set on each chunk in metadata["keywords"]. They are used as
    a soft bonus in scoring in query_knowledge_routed, not as a hard
    filter — otherwise a non-matching route would quietly return nothing.
    """
    routes = []

    # ❌ THE UNCONDITIONAL "specific trading rule" ROUTE WAS REMOVED
    # Now it is added ONLY if there are no other routes (at the end of the function)

    if "delta_divergence" in tags or "bullish_divergence_complex" in tags:
        routes.append({
            "query": "divergence delta absorption institutional buying selling pressure",
            "keywords": ["orderflow", "delta", "divergence", "absorption", "institutional"],
            "priority": 10
        })

    if "large_player_active" in tags:
        routes.append({
            "query": "large player institutional volume accumulation building a position",
            "keywords": ["orderflow", "institutional", "volume"],
            "priority": 9
        })

    if "below_vwap" in tags:
        routes.append({
            "query": "trading below VWAP intraday trend reversal support",
            "keywords": ["vwap", "support_resistance", "trend"],
            "priority": 8
        })

    if "at_poc" in tags or "above_poc" in tags or "below_poc" in tags:
        routes.append({
            "query": "POC volume profile value area acceptance rejection level",
            "keywords": ["poc", "volume_profile", "pattern"],
            "priority": 7
        })

    if "macd_bearish" in tags and "downtrend" in tags:
        routes.append({
            "query": "MACD bearish downtrend momentum weakness continuation",
            "keywords": ["macd", "trend", "pattern"],
            "priority": 7
        })

    if "neutral_rsi" in tags and "downtrend" in tags:
        routes.append({
            "query": "RSI neutral downtrend hidden divergence",
            "keywords": ["rsi", "divergence", "trend"],
            "priority": 6
        })

    if "macd_bullish" in tags and "uptrend" in tags:
        routes.append({
            "query": "MACD bullish uptrend momentum strength continuation",
            "keywords": ["macd", "trend", "pattern"],
            "priority": 7
        })

    if "low_volume" in tags or "ultra_low_volume" in tags:
        routes.append({
            "query": "low volume trading consolidation breakout false breakout liquidity",
            "keywords": ["volume", "breakout", "false_signal", "risk"],
            "priority": 6
        })

    if "high_volume" in tags or "ultra_high_volume" in tags:
        routes.append({
            "query": "high volume spike climax exhaustion blow-off",
            "keywords": ["volume", "climax", "spike"],
            "priority": 6
        })

    if "earnings_soon" in tags or "earnings_imminent" in tags:
        routes.append({
            "query": "trading before earnings volatility position size risk management",
            "keywords": ["risk", "psychology", "volatility"],
            "priority": 5
        })

    if "near_52w_high" in tags or "all_time_high" in tags:
        routes.append({
            "query": "trading at highs breakout continuation resistance",
            "keywords": ["breakout", "extremum", "psychology"],
            "priority": 5
        })

    if "near_52w_low" in tags or "all_time_low" in tags:
        routes.append({
            "query": "trading at lows breakdown support decline",
            "keywords": ["breakout", "extremum", "psychology"],
            "priority": 5
        })

    if "near_bb_upper" in tags:
        routes.append({
            "query": "upper bollinger band overbought resistance pullback",
            "keywords": ["bollinger", "extremum", "pattern"],
            "priority": 4
        })

    if "near_bb_lower" in tags:
        routes.append({
            "query": "lower bollinger band oversold support bounce",
            "keywords": ["bollinger", "extremum", "pattern"],
            "priority": 4
        })

    if "capitulation_possible" in tags:
        routes.append({
            "query": "capitulation panic reversal bottom hammer hanging man",
            "keywords": ["candles", "psychology", "climax"],
            "priority": 6
        })

    if "exhaustion_possible" in tags:
        routes.append({
            "query": "exhaustion euphoria reversal top shooting star doji",
            "keywords": ["candles", "psychology", "climax"],
            "priority": 6
        })

    # ✅ ONLY IF THERE ARE NO SPECIFIC ROUTES — THEN ADD THE GENERAL ONE
    if not routes:
        routes.append({
            "query": "specific trading rule entry exit stop target signal",
            "keywords": ["entry_exit", "buy_sell"],
            "priority": 15
        })
        # A universal technical analysis query can also be added as a second option
        routes.append({
            "query": "technical analysis trend support resistance candles",
            "keywords": ["candles", "pattern", "trend", "support_resistance"],
            "priority": 1
        })

    routes.sort(key=lambda x: x["priority"], reverse=True)

    return routes


def query_knowledge_routed(tags, n_results=5):
    """
    Knowledge Router: Python decides WHAT to look for (the query + the route's thematic
    keywords), then searches semantically across the WHOLE base and only then
    ranks the result in Python — with a bonus for matching the route's keywords
    and for the route's priority.

    There used to be a hard Chroma filter where={"category": {"$in": categories}},
    but the routes' categories came from a different taxonomy than the real metadata["category"]
    (entry/exit/risk/confirmation/context) — the filter almost always gave 0 matches
    (not an exception but a valid empty result, so the except fallback did not fire),
    and only chunks with an accidentally matching "risk" category leaked out —
    hence "the same 2-3 rules from the whole book". Now there is no hard filter:
    semantic search always returns candidates across the whole base, and the route's
    keywords work as a soft bonus in scoring, not as a zeroing filter.
    """
    if not RAG_ENABLED or collection is None or not tags:
        return ""

    routes = route_knowledge_search(tags)

    all_results = []
    seen = set()

    print(f"\n   🧭 KNOWLEDGE ROUTER: {len(routes)} routes:")

    for route in routes[:5]:
        query = route["query"]
        route_keywords = set(route.get("keywords", []))
        priority = route["priority"]

        print(f"      🔍 [{priority}] '{query[:MAX_LINE_LOG]}...' → keywords: {route_keywords or 'no filter'}")

        results = collection.query(query_texts=[query], n_results=8)

        docs = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []

        for doc, m in zip(docs, metadatas):
            doc_key = doc[:100]
            if doc_key not in seen:
                keywords = m.get('keywords', [])
                score = score_chunk_for_tags(doc, tags, m.get('category', ''), keywords)
                score += priority * 2
                if route_keywords and (set(keywords) & route_keywords):
                    score += 8  # bonus for hitting the route's topic
                all_results.append((doc, m, score))
                seen.add(doc_key)

    all_results.sort(key=lambda x: x[2], reverse=True)
    top_results = all_results[:n_results]
    
    if top_results:
        print(f"\n   📊 Routed search: selected {len(top_results)} excerpts:")
        for i, (doc, m, score) in enumerate(top_results):
            keywords = m.get('keywords', [])
            print(f"   {'='*60}")
            print(f"   [{i+1}] score={score} | {m.get('category', '?')} | kw: {keywords}")
            print(f"   {doc}")
            print(f"   {'='*60}")
        
        clean_chunks = [doc for doc, _, _ in top_results]
        return "\n\n".join(clean_chunks)
    return ""