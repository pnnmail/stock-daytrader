import time
import json
import re
import requests
from config import OLLAMA_URL, MAX_LINE_LOG, THINKING_MODEL_PREFIXES


def _payload_for(model, prompt, json_schema, temperature):
    """
    Shared request-body builder for call_ollama/call_ollama_json — the single
    place that decides whether think=False is needed.

    WHY BY NAME PREFIX AND NOT BY THE Ollama "thinking" CAPABILITY: formally it is
    reported both by models that reason by default (Qwen3) and by those
    that do not (gemma4:e4b) — see THINKING_MODEL_PREFIXES in config.py.
    What differs is the default behaviour of a given model family, not the
    ability to think as such.
    """
    payload = {"model": model, "prompt": prompt, "stream": False}
    if json_schema is not None:
        payload["format"] = json_schema
    if temperature is not None:
        payload["options"] = {"temperature": temperature, "top_p": 0.85}
    if model.startswith(THINKING_MODEL_PREFIXES):
        payload["think"] = False
    return payload


def call_ollama(prompt, model, temperature=None):
    start = time.time()
    payload = _payload_for(model, prompt, None, temperature)
    resp = requests.post(OLLAMA_URL, json=payload, timeout=1820)
    resp.raise_for_status()
    elapsed = time.time() - start
    result = resp.json()["response"].strip()
    ts = f"{elapsed:.1f} s" if elapsed < 60 else f"{elapsed/60:.1f} min"
    print(f"⏱ {model} answered in {ts}")
    return result


def call_ollama_json(prompt, model, json_schema=None, temperature=None):
    start = time.time()
    payload = _payload_for(model, prompt, json_schema, temperature)
    resp = requests.post(OLLAMA_URL, json=payload, timeout=1820)
    resp.raise_for_status()
    elapsed = time.time() - start
    raw = resp.json()["response"].strip()
    ts = f"{elapsed:.1f} s" if elapsed < 60 else f"{elapsed/60:.1f} min"
    print(f"⏱ {model} answered in {ts}")
    print(f"   🔍 RAW RESPONSE ({len(raw)} chars):")
    print(f"   {'─'*50}")
     # Print the WHOLE response without truncation
    for line in raw.split('\n'):
        print(f"   │ {line}")
    print(f"   {'─'*50}")
    try:
        result = json.loads(raw)
        print(f"   ✅ JSON parsed successfully (structured output)")
        if json_schema:
            result = validate_json_response(result, json_schema)
        return result
    except json.JSONDecodeError:
        pass
    result = extract_from_raw(raw)
    if json_schema:
        result = validate_json_response(result, json_schema)
    return result


def validate_json_response(data, schema):
    if not isinstance(data, dict):
        return data
    if data.get("error"):
        print(f"   ⚠️ The model returned an error: {data.get('message', '')}")
        return {"recommendation": "hold", "changed": False, "reason": "The model could not check the recommendation"}
    
    RUS_TO_ENG = {
        "rec": "recommendation", "dir": "direction",
        "entry": "entry_price", "stop": "stop_loss", "target1": "target_1", "target2": "target_2",
        "rationale": "reasoning",
    }
    new_data = {}
    for key, value in data.items():
        eng_key = RUS_TO_ENG.get(key, key)
        new_data[eng_key] = value
    data = new_data
    
    ENUM_MAP = {
        "macd_state": {"slightly_bearish": "bearish", "slightly_bullish": "bullish"},
        "volume_state": {"low": "below_average", "high": "above_average"},
    }
    required_fields = schema.get("required", [])
    properties = schema.get("properties", {})
    
    for field in required_fields:
        if field not in data:
            prop = properties.get(field, {})
            if "enum" in prop:
                data[field] = prop["enum"][0]
            elif prop.get("type") == "string":
                data[field] = ""
            elif prop.get("type") == "number":
                data[field] = 0
            elif prop.get("type") == "boolean":
                data[field] = False
            elif prop.get("type") == "array":
                data[field] = []
    
    for field, prop in properties.items():
        if "enum" in prop and field in data:
            current_val = data[field]
            if current_val in prop["enum"]:
                continue
            if field in ENUM_MAP and current_val in ENUM_MAP[field]:
                data[field] = ENUM_MAP[field][current_val]
                continue
            matched = None
            for enum_val in prop["enum"]:
                if enum_val in str(current_val).lower():
                    matched = enum_val
                    break
            if matched:
                data[field] = matched
            else:
                data[field] = prop["enum"][0]
    return data


def extract_from_raw(raw):
    raw = re.sub(r'```(?:json)?\s*\n?', '', raw)
    raw = re.sub(r'```\s*$', '', raw)
    raw = raw.strip()
    result = {
        "rsi_state": "neutral", "macd_state": "neutral", "trend_state": "sideways",
        "volume_state": "average", "bollinger_state": "middle",
        "executive_summary": "", "risks": [], "opportunities": [],
        "direction": "OUT", "entry_price": 0, "stop_loss": 0,
        "target_1": 0, "target_2": 0, "risk_reward_1": 0, "risk_reward_2": 0,
        "position_size_pct": 0, "reasoning": "",
        "orderflow_bias": None, "absorption_detected": False,
        "large_player_active": False, "delta_divergence": False,
        "similar_pattern": "", "recommended_actions": "", "key_warnings": ""
    }
    patterns = {
        "executive_summary": r'"executive_summary":\s*"([^"]+)"',
        "reasoning": r'"reasoning":\s*"([^"]+)"',
        "direction": r'"direction":\s*"(\w+)"',
        "entry_price": r'"entry_price":\s*([\d.]+)',
        "stop_loss": r'"stop_loss":\s*([\d.]+)',
        "target_1": r'"target_1":\s*([\d.]+)',
        "target_2": r'"target_2":\s*([\d.]+)',
        "risk_reward_1": r'"risk_reward_1":\s*([\d.]+)',
        "risk_reward_2": r'"risk_reward_2":\s*([\d.]+)',
        "position_size_pct": r'"position_size_pct":\s*([\d.]+)',
        "orderflow_bias": r'"orderflow_bias":\s*"(\w+)"',
        "similar_pattern": r'"similar_pattern":\s*"([^"]+)"',
        "recommended_actions": r'"recommended_actions":\s*"([^"]+)"',
        "key_warnings": r'"key_warnings":\s*"([^"]+)"',
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, raw)
        if match:
            val = match.group(1)
            if key in ["entry_price", "stop_loss", "target_1", "target_2", "risk_reward_1", "risk_reward_2", "position_size_pct"]:
                try:
                    result[key] = float(val)
                except:
                    pass
            elif key == "orderflow_bias":
                result[key] = val
            else:
                result[key] = val
    for bool_field in ["absorption_detected", "large_player_active", "volume_confirms_trend", "delta_divergence"]:
        match = re.search(rf'"{bool_field}":\s*(true|false)', raw)
        if match:
            result[bool_field] = match.group(1).lower() == 'true'
    return result