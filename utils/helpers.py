import re
import json
from config import STATE_FILE


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except:
        return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def age_str(iso):
    """
    Age of an ISO timestamp in words: '40 min ago', '3 h ago', '2 d ago'.
    Used wherever data age is shown: nobody converts '2026-08-07T07:12:44'
    in their head, while "3 h ago" immediately tells you whether to refresh.
    """
    from datetime import datetime
    try:
        seconds = (datetime.now() - datetime.fromisoformat(iso)).total_seconds()
    except (TypeError, ValueError):
        return "age unknown"
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    if seconds < 48 * 3600:
        return f"{seconds / 3600:.0f} h ago"
    return f"{seconds / 86400:.0f} d ago"


def flatten_to_nested(flat):
    return {
        "technical": {
            "rsi": {"state": flat.get("rsi_state", "neutral"), "reason": flat.get("rsi_reason", "")},
            "macd": {"state": flat.get("macd_state", "neutral"), "reason": flat.get("macd_reason", "")},
            "trend": {"state": flat.get("trend_state", "sideways"), "reason": flat.get("trend_reason", "")},
            "volume": {"state": flat.get("volume_state", "average"), "reason": flat.get("volume_reason", "")},
            "bollinger": {"state": flat.get("bollinger_state", "middle"), "reason": flat.get("bollinger_reason", "")}
        },
        "executive_summary": flat.get("executive_summary", ""),
        "risks": flat.get("risks", []),
        "opportunities": flat.get("opportunities", []),
        "key_news": flat.get("key_news", []),
    }

def safe_format_float(value, decimals=2, default="?"):
    """
    Safely formats a floating-point number.
    If value is None or not a number, returns default.
    """
    if value is None:
        return default
    try:
        return f"{float(value):.{decimals}f}"
    except (ValueError, TypeError):
        return default

def is_chapter_title(line):
    stripped = line.strip()
    if not stripped or len(stripped) > 100:
        return False
    if re.match(r'^(Chapter|CHAPTER|Ch\.|Section|Part)\s+\d+', stripped, re.IGNORECASE):
        return True
    if re.match(r'^[IVX]+\.\s', stripped):
        return True
    if re.match(r'^\d+\.\s+[A-Z]', stripped):
        return True
    if len(stripped) < 80 and stripped.isupper() and len(stripped.split()) > 2:
        return True
    return False


def is_outline_or_junk(text):
    text_clean = text.strip()
    if not text_clean:
        return True

    dialog_markers = ["PROFESSOR COIN", "MISS TREND", "HOST"]
    if any(marker in text_clean for marker in dialog_markers):
        return True

    words = text_clean.split()
    if len(words) < 5:
        return len(text_clean) < 80

    page_numbers = 0
    for w in words:
        w_clean = w.strip('.,;:')
        if re.match(r'^\d{2,4}$', w_clean):
            if not (len(w_clean) == 4 and w_clean.startswith(('19', '20'))):
                page_numbers += 1

    ratio_page_numbers = page_numbers / len(words)

    word_lengths = [len(w.strip('.,;:')) for w in words if not re.match(r'^\d+$', w.strip('.,;:'))]
    if word_lengths:
        avg_word_len = sum(word_lengths) / len(word_lengths)
    else:
        avg_word_len = 0

    if ratio_page_numbers > 0.15:
        return True
    if avg_word_len < 4.5 and page_numbers >= 2:
        return True

    leading_digit_words = sum(1 for w in words if re.match(r'^\d{1,3}\.\s', w) or re.match(r'^\d{2,4}\s', w))
    if leading_digit_words >= 3:
        return True

    alpha_chars = sum(1 for c in text_clean if c.isalpha())
    if alpha_chars > 0:
        upper_ratio = sum(1 for c in text_clean if c.isupper()) / alpha_chars
        if upper_ratio > 0.7 and len(text_clean) < 200:
            return True

    if len(text_clean) < 60:
        return True

    return False


def compute_quality_score(text):
    score = 0
    length = len(text)
    if 50 <= length <= 100:
        score += 2
    elif 100 < length <= 200:
        score += 3
    elif length > 200:
        score += 2
    else:
        score += 1

    if re.search(r'\d+', text):
        score += 2

    condition_words = ["if", "when", "on", "after", "above", "below",
                       "exceeds", "breakout", "bounce", "close"]
    matches = sum(1 for w in condition_words if w in text.lower())
    if matches >= 2:
        score += 2
    elif matches >= 1:
        score += 1

    action_words = ["buy", "sell", "open", "close",
                    "enter", "exit", "stop", "target"]
    if any(w in text.lower() for w in action_words):
        score += 2

    if re.search(r'[\d]+\s*%', text) or re.search(r'\$\d+', text):
        score += 1

    vague_words = ["maybe", "possibly", "sometimes", "as a rule"]
    if any(w in text.lower() for w in vague_words):
        score -= 1

    return min(10, max(0, score))


def escape_markdown(text: str) -> str:
    """
    Escapes Telegram parse_mode="Markdown" (legacy) special characters: _ * ` [.
    Unlike MarkdownV2, legacy mode does not need (or understand) escaping of
    dots, brackets, dashes etc. — an extra '\' before them is printed literally.
    """
    if not text:
        return ""
    escape_chars = r'_*`['
    return ''.join(f'\\{c}' if c in escape_chars else c for c in text)
