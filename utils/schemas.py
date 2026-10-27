# ============================================================
# JSON SCHEMAS
# ============================================================

JUNIOR_SCHEMA = {
    "type": "object",
    "properties": {
        "rsi_state": {"enum": ["overbought", "oversold", "neutral", "elevated_but_normal", "depressed_but_normal"]},
        "rsi_reason": {"type": "string"},
        "macd_state": {"enum": ["bullish", "bearish", "neutral"]},
        "macd_reason": {"type": "string"},
        "trend_state": {"enum": ["strong_uptrend", "uptrend", "sideways", "downtrend", "strong_downtrend"]},
        "trend_reason": {"type": "string"},
        "volume_state": {"enum": ["above_average", "average", "below_average"]},
        "volume_reason": {"type": "string"},
        "bollinger_state": {"enum": ["near_upper", "near_lower", "middle"]},
        "bollinger_reason": {"type": "string"},
        "executive_summary": {"type": "string"},
        "risks": {"type": "array", "items": {"type": "string"}},
        "opportunities": {"type": "array", "items": {"type": "string"}},
        "key_news": {"type": "array", "items": {"type": "string"}},
        "support_levels": {"type": "array", "items": {"type": "number"}},
        "resistance_levels": {"type": "array", "items": {"type": "number"}},
        "intraday_bias": {"enum": ["long", "short", "neutral"]},
        "key_pattern": {"type": "string"}
    },
    "required": ["rsi_state", "macd_state", "trend_state", "volume_state", 
                 "bollinger_state", "executive_summary", "risks", "opportunities"]
}

LIBRARIAN_SCHEMA = {
    "type": "object",
    "properties": {
        "selected": {"type": "array", "items": {"type": "integer"}}
    },
    "required": ["selected"]
}

TRADER_SCHEMA = {
    "type": "object",
    "properties": {
        "direction": {"enum": ["LONG", "SHORT", "OUT"]},
        "entry_price": {"type": "number"},
        "entry_type": {"enum": ["LIMIT", "MARKET", "STOP"]},
        "entry_condition": {"type": "string"},
        "stop_loss": {"type": "number"},
        "stop_pct": {"type": "number"},
        "target_1": {"type": "number"},
        "target_1_pct": {"type": "number"},
        "target_2": {"type": "number"},
        "target_2_pct": {"type": "number"},
        "risk_reward_1": {"type": "number"},
        "risk_reward_2": {"type": "number"},
        "position_size_pct": {"type": "number"},
        "max_holding_time": {"type": "string"},
        "invalidation_condition": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reasoning": {"type": "string"}
    },
    "required": ["direction", "entry_price", "stop_loss", "target_1", "target_2", 
                 "risk_reward_1", "risk_reward_2", "position_size_pct", "reasoning"]
}

EXTRACTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": "Name of the pattern or model from the book (e.g. false breakout, trailing the stop)"
        },
        "rules": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Concrete rules extracted from the text (not tied to levels, prices or indicators)"
        },
        "warnings": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Warnings from the book (not tied to market data)"
        }
    },
    "required": ["pattern", "rules", "warnings"]
}

DISTILLER_SCHEMA = {
    "type": "object",
    "properties": {
        "has_useful_knowledge": {"type": "boolean"},
        "rules": {"type": "array", "items": {"type": "string"}, "description": "Concrete trading rules"},
        "key_ideas": {"type": "array", "items": {"type": "string"}, "description": "Key ideas"},
        "warnings": {"type": "array", "items": {"type": "string"}, "description": "Warnings"},
    },
    "required": ["has_useful_knowledge"]
}

NEWS_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_sentiment": {"enum": ["positive", "neutral", "negative", "mixed"]},
        "summary": {"type": "string"},
        "key_themes": {"type": "array", "items": {"type": "string"}},
        "detailed_analysis": {"type": "string"},
        "catalysts": {"type": "array", "items": {"type": "string"}},
        "risk_flags": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["overall_sentiment", "summary", "key_themes", "detailed_analysis", "catalysts", "risk_flags"]
}

# Whole-market overview (agents/market_overview.py). A separate schema rather than
# NEWS_SCHEMA: that one is built around ONE company ("catalysts" = what will move
# its price, "risk_flags" = issuer problems). Here the subject is different — the market —
# and the fields mean different things: tone in risk-on/risk-off terms, events instead
# of issuer catalysts, "what to watch" instead of company flags.
# THE SCHEMA HAS NO NUMBERS ON PURPOSE: indices, VIX, yields and breadth are computed
# in Python and go into the report directly. Give the model a field for a percentage and it
# will fill it, and the report ends up with two different numbers for the same thing.
MARKET_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_tone": {"enum": ["risk_on", "risk_off", "mixed", "neutral"]},
        "summary": {"type": "string"},
        "key_events": {"type": "array", "items": {"type": "string"}},
        "themes": {"type": "array", "items": {"type": "string"}},
        "detailed_analysis": {"type": "string"},
        "risks": {"type": "array", "items": {"type": "string"}},
        "what_to_watch": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["overall_tone", "summary", "key_events", "themes",
                 "detailed_analysis", "risks", "what_to_watch"]
}

TRADER_SCHEMA = {
    "type": "object",
    "properties": {
        "historical_analogy": {"type": "string", "description": "A similar situation from history"},
        "risks": {"type": "array", "items": {"type": "string"}, "description": "Key risks (3-5)"},
        "commentary": {"type": "string", "description": "General commentary, context"}
    },
    "required": ["historical_analogy", "risks", "commentary"]
}