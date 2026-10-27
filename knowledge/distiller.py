# knowledge/distiller.py
import re
from config import MAX_LINE_LOG, DISTILLER_MODEL
from utils.ollama import call_ollama_json

DISTILLER_SCHEMA = {
    "type": "object",
    "properties": {
        "has_knowledge": {"type": "boolean"},
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string"},
                    "action": {"enum": ["buy", "sell", "hold", "confirm", "exit", ""]},
                    "market_phase": {
                        "type": "array",
                        "items": {"enum": ["downtrend", "uptrend", "sideways", "any", "oversold", "overbought"]}
                    },
                    "confidence": {"enum": ["explicit", "derived"]},
                    # "context" is deliberately excluded from the enum: it was a dumping-ground category —
                    # 45% of the whole extracted base fell into it (sentence fragments, retellings
                    # of definitions without a condition and an action). If a fact does not qualify for any of
                    # the 4 actionable categories, it should not be extracted at all (has_knowledge=false).
                    "category": {"enum": ["entry", "exit", "risk", "confirmation"]},
                    "original_text": {"type": "string"},
                    "visual_markers": {"type": "array", "items": {"type": "string"}},
                    "programmable_markers": {"type": "array", "items": {"type": "string"}},
                    "requires_visual": {"type": "boolean"},
                    "confirmation_needed": {"type": "boolean"}
                },
                "required": ["concept", "original_text", "action", "market_phase", "programmable_markers", "visual_markers", "category"]
            }
        }
    },
    "required": ["has_knowledge"]
}


TERM_MAP = {
    "divergence": ["divergence", "diverging", "non-confirmation"],
    "vwap": ["vwap", "volume-weighted", "volume weighted average"],
    "delta": ["delta", "cumulative delta"],
    "volume": ["volume", "open interest", "turnover"],
    "support_resistance": ["support", "resistance", "level", "levels",
                          "lows", "highs", "cheap", "expensive", "bottom",
                          "top", "peak", "pullback", "bounce"],
    "trend": ["trend", "uptrend", "downtrend", "rising", "falling", "rally", "decline", "price rise", "price decline"],
    "rsi": ["rsi", "relative strength index", "relative strength"],
    "macd": ["macd", "convergence divergence", "moving averages"],
    "candles": ["candle", "doji", "hammer", "engulfing", "star", "harami", "dark cloud", "hanging man",
                "morning star", "evening star", "shooting star", "abandoned baby",
                "belt hold", "tweezer", "counterattack", "gap", "window"],
    "oscillator": ["oscillator", "stochastic"],
    "breakout": ["breakout", "break out", "breaks above", "exceeds", "breaks below"],
    "bollinger": ["bollinger", "band"],
    "pivot": ["pivot", "pivot point"],
    "fibonacci": ["fibonacci", "retracement", "fibo"],
    "elliott": ["elliott", "wave"],
    "dow": ["dow theory"],
    "volume_profile": ["volume profile", "market profile"],
    "poc": ["poc", "point of control"],
    "orderflow": ["order flow", "orderflow", "tape"],
    "absorption": ["absorption", "absorbing", "absorbed"],
    "institutional": ["institutional", "large player", "market maker"],
    "stops": ["stop", "stop-loss", "stop loss", "stop order", "protective"],
    "risk": ["risk", "risk management", "money management", "loss", "profit"],
    "entry_exit": ["entry", "exit", "entry point", "exit point", "open a", "close a",
                   "position", "trade", "order", "on the open", "on the close"],
    "volatility": ["volatility", "fluctuation", "range", "swing"],
    "moving_average": ["moving average", "moving averages", "sma", "ema",
                       "average"],
    "buy_sell": ["buy", "sell", "buying", "selling", "purchase", "sale", "go long",
                 "go short", "buy signal", "sell signal",
                 "open a long", "open a short", "close the position"],
    "spike": ["spike", "spikes", "spike up", "spike down"],
    "extremum": ["high", "low", "maximum", "minimum", "peak", "trough",
                 "relative high", "relative low", "major top",
                 "major bottom", "local high", "local low", "highs", "lows"],
    "tick": ["tick", "ticks", "tick chart", "10-tick"],
    "turn": ["turn", "turns", "reversal", "turning point", "reversed", "back"],
    "trending_market": ["strong trend", "trending market", "choppy market",
                        "trading range", "directional move"],
    "false_signal": ["false signal", "false signals", "false breakout", "false breakouts"],
    "climax": ["climax", "selling pressure", "buying pressure",
               "temporary climax", "exhaustion"],
    "closing_price": ["close", "closed", "open", "opened", "closing price", "opening price",
                       "upper boundary", "lower boundary", "closing", "opening"],
    "direction": ["decline", "advance", "after a decline", "after an advance",
                  "up", "down", "rise", "fall"],
    "pattern": ["pattern", "forms", "formation", "figure",
                "head and shoulders", "double top", "double bottom", "flag", "pennant",
                "triangle", "wedge", "consolidation"],
    "trading_system": ["trading system", "system", "computerized", "mechanical"],
    "psychology": ["common sense", "intuition", "crowd", "majority", "sentiment"],
    "backtest": ["test", "testing", "historical data", "past", "profitable"],
}

# A flat set of all TERM_MAP terms — used as a filter "does the paragraph
# relate to trading at all" when loading a PDF (knowledge/loader.py), instead of
# a separate short list that did not match the book's
# vocabulary well.
ALL_TRADE_TERMS = sorted({term for terms in TERM_MAP.values() for term in terms})


def extract_keywords_from_chunk(text):
    """Extracts keywords (unchanged)."""
    keywords = []
    text_lower = text.lower()
    for keyword, terms in TERM_MAP.items():
        for term in terms:
            if term in text_lower:
                keywords.append(keyword)
                break
    if not keywords:
        if re.search(r'\d+', text):
            keywords.append("numeric")
        if '$' in text or 'usd' in text_lower:
            keywords.append("price")
        action_words = ["buy", "sell", "open", "close",
                        "stop", "target", "entry", "exit"]
        for word in action_words:
            if word in text_lower:
                keywords.append("action")
                break
    return list(set(keywords)) if keywords else ["general"]


FIND_RULES_SCHEMA = {
    "type": "object",
    "properties": {
        "rules": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["rules"]
}


def _normalize_for_match(text):
    """
    Collapses hyphenated line breaks like "effec- tive" -> "effective" and extra
    whitespace — otherwise the model's verbatim quote won't match the PDF original
    line by line, although in meaning it is the same word.
    """
    text = re.sub(r'-\s+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()


def _find_candidate_rules(para_text):
    """
    Stage 1: free search for candidate rules with a SIMPLE schema without enum fields.

    Tested experimentally: with the full DISTILLER_SCHEMA (many enum fields
    that have to be filled at the same time as deciding "is there a fact here at all")
    the model massively returned has_knowledge=false even where the rule was
    right there in the text almost verbatim — apparently the combination "find + immediately strictly
    classify" was too risky for a 7B model, and it fell back
    to a safe empty answer. With this simple schema (just find the rule text,
    no classification) the model finds them reliably.
    """
    prompt = f"""Find in the text below concrete trading rules or principles — sentences
that have a condition and an action (if X — then Y), OR a general trading principle/lesson (even if
it is phrased as a conclusion from analysing a specific trade in a "trader's diary" rather than as
a direct instruction).

Do NOT include:
- plain definitions of terms without a condition and an action
- sentences referring to figures/charts ("fig. X.Y", "the chart above") without restating
  their substance in words
- bare statements of "what happened" in one trade WITHOUT a general conclusion/lesson
  (e.g. "the position was liquidated on the stop" — just a fact, skip it)
- rhetorical questions to the reader ("do you agree with this analysis?")

Return each rule as a VERBATIM quote from the text — do not paraphrase, do not change the word
order, take the whole sentence as it is. If there are no rules, return an empty list,
which is a normal result for many paragraphs.

TEXT:
{para_text[:4000]}
"""
    try:
        result = call_ollama_json(prompt, DISTILLER_MODEL, json_schema=FIND_RULES_SCHEMA, temperature=0)
        return result.get("rules", []) or []
    except Exception as e:
        print(f"   ⚠️ Rule search error: {str(e)[:200]}")
        return []


def _structure_rule(rule_text):
    """
    Stage 2: structures a rule ALREADY found (by stage 1) using the full schema.
    The model does this reliably when it does not need to SIMULTANEOUSLY decide
    whether there is a fact here at all — that question is already settled, only classification remains.
    """
    prompt = f"""Here is a trading rule already found in a book. Structure it.

RULE: "{rule_text}"

- concept: a short name (e.g. "spike_up", "exhaustion_gap")
- action: "buy", "sell", "hold", "confirm" or "" (an empty string)
- market_phase: a list of "downtrend", "uptrend", "sideways", "any" — NOT empty
- category: "entry" (entry condition), "exit" (exit/stop condition), "risk" (risk/position
  management) or "confirmation" (a confirming signal)
- original_text: repeat the rule VERBATIM as given above, changing nothing
- programmable_markers: concrete checkable conditions (close > open, volume > avg_volume*1.5).
  If the rule is not about a concrete price signal (e.g. risk management/discipline) — ["none"]
- visual_markers: visual features of the candle/pattern, or ["none"]
- confidence: "explicit" (stated directly as a rule) or "derived" (inferred from
  a described example/case rather than stated directly as a general rule)

has_knowledge: true — the rule is already found, your task is only to structure it.
Do NOT invent RSI, MACD — only close, open, high, low, volume, unless named explicitly.
"""
    try:
        result = call_ollama_json(prompt, DISTILLER_MODEL, json_schema=DISTILLER_SCHEMA, temperature=0)
        facts = result.get("facts", [])
        return facts[0] if facts else None
    except Exception as e:
        print(f"   ⚠️ Rule structuring error: {str(e)[:200]}")
        return None


def distill_paragraph(para_text, chapter_title):
    """
    Extracts atomic facts (concepts) from a paragraph via an LLM — in 2 stages
    (see _find_candidate_rules and _structure_rule): first a free search for
    rules with a simple schema, then structuring each found rule
    with the full schema. Separately the model does these two steps reliably; together in one
    call it almost always fell back to an empty answer (tested experimentally).
    """
    if len(para_text) < 200:
        return None

    print(f"\n   📝 PARAGRAPH DISTILLATION:")
    print(f"   {'─'*50}")
    for line in para_text[:MAX_LINE_LOG].split('\n'):
        print(f"   │ {line[:MAX_LINE_LOG]}")
    print(f"   {'─'*50}")

    candidate_rules = _find_candidate_rules(para_text)
    if not candidate_rules:
        print(f"   ℹ️ No rules found")
        return None

    print(f"   🔍 Candidates found: {len(candidate_rules)}")

    para_normalized = _normalize_for_match(para_text)
    facts = []
    for rule_text in candidate_rules[:3]:  # no more than 3 facts per paragraph
        if not rule_text or not rule_text.strip():
            continue

        # Anti-hallucination: the candidate must actually occur in the source text
        rule_key = _normalize_for_match(rule_text)[:60]
        if len(rule_key) > 20 and rule_key not in para_normalized:
            print(f"      ⚠️ Skipped a hallucinated candidate: {rule_text[:80]}")
            continue

        fact = _structure_rule(rule_text)
        if not fact:
            continue
        facts.append(fact)
        print(f"      🧠 {fact.get('concept')} | {fact.get('action')} | {fact.get('market_phase')}")

    if facts:
        return {"has_knowledge": True, "facts": facts}
    print(f"   ℹ️ No candidate passed structuring")
    return None