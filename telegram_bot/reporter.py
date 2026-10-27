import time
import requests
from config import TELEGRAM_BOT_TOKEN
from utils.helpers import safe_format_float, escape_markdown  # escape_markdown added


# Telegram limit is 4096 characters per message. Keep a margin: emoji and
# non-ASCII text are counted differently in different parts of the API, no point cutting it close.
TELEGRAM_CHUNK = 3800


def split_for_telegram(text, limit=TELEGRAM_CHUNK):
    """
    Splits a long report at PARAGRAPH BOUNDARIES rather than blindly by characters.

    Blind slicing (text[i:i+3800]) breaks a line in the middle and, worse,
    splits a '*...*' pair — Telegram answers 400 on the markup, and the message
    goes out as plain text, losing all formatting. Here the break always
    falls between paragraphs, so each part stays self-contained
    in Markdown terms.

    A paragraph longer than the limit (possible with the model's detailed_analysis)
    is split by lines, and a single very long line — by characters,
    as a last resort.
    """
    if len(text) <= limit:
        return [text]

    parts, current = [], ""
    for block in text.split("\n\n"):
        candidate = f"{current}\n\n{block}" if current else block
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            parts.append(current)
            current = ""
        if len(block) <= limit:
            current = block
            continue
        # The paragraph alone does not fit — split by lines
        for line in block.split("\n"):
            candidate = f"{current}\n{line}" if current else line
            if len(candidate) <= limit:
                current = candidate
                continue
            if current:
                parts.append(current)
                current = ""
            while len(line) > limit:
                parts.append(line[:limit])
                line = line[limit:]
            current = line
    if current:
        parts.append(current)
    return parts


def send_telegram_sync(chat_id, text, max_retries=3):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    for part in split_for_telegram(text):
        for attempt in range(max_retries):
            try:
                resp = requests.post(url, json={"chat_id": chat_id, "text": part, "parse_mode": "Markdown"}, timeout=60)
                if resp.status_code == 400 and "parse" in resp.text.lower():
                    # Markup did not match (e.g. '*'/'_' was broken by the split) —
                    # send as plain text so the message is not lost.
                    print(f"   ⚠️ Markdown failed to parse, sending as plain text: {resp.text[:200]}")
                    resp = requests.post(url, json={"chat_id": chat_id, "text": part}, timeout=60)
                resp.raise_for_status()
                break  # success, leave the retry loop
            except requests.exceptions.Timeout:
                print(f"   ⚠️ Send timeout, attempt {attempt+1}/{max_retries}...")
                time.sleep(2 ** attempt)  # exponential backoff
            except requests.exceptions.RequestException as e:
                print(f"   ⚠️ Send error: {e}, attempt {attempt+1}/{max_retries}...")
                time.sleep(2 ** attempt)
        else:
            print(f"   ❌ Could not send the message after {max_retries} attempts")
        if len(text) > 3800:
            time.sleep(1.5)


def send_trader_report(chat_id, ticker, junior_analysis, orderflow, trader_plan):
    summary = junior_analysis.get("executive_summary", "")
    if isinstance(summary, list):
        summary = " ".join(summary)
    risks = junior_analysis.get("risks", [])
    
    # Junior
    msg1 = f"🔍 *JUNIOR — {escape_markdown(ticker)}*\n\n📝 {escape_markdown(summary)}"
    if risks:
        msg1 += f"\n\n⚠️ Risks: " + ", ".join(escape_markdown(r) for r in risks[:3])
    send_telegram_sync(chat_id, msg1)
    time.sleep(0.5)
    
    # Order Flow
    if orderflow:
        msg2 = f"📊 *ORDER FLOW — {escape_markdown(ticker)}*\n\n"
        msg2 += f"  • Flow: *{escape_markdown(orderflow.get('orderflow_bias', '?'))}*\n"
        msg2 += f"  • Large player: *{'YES' if orderflow.get('large_player_active') else 'NO'}*\n"
        msg2 += f"  • Divergence: *{'YES' if orderflow.get('delta_divergence') else 'NO'}*\n"
        msg2 += f"  • {escape_markdown(orderflow.get('recommendation', ''))}\n"
        msg2 += f"\n📝 {escape_markdown(orderflow.get('reasoning', '')[:2000])}"
        send_telegram_sync(chat_id, msg2)
        time.sleep(0.5)
    
    # Trade Plan
    if trader_plan and trader_plan.get("direction") != "OUT":
        msg3 = f"📈 *TRADE PLAN — {escape_markdown(ticker)}*\n\n"
        msg3 += f"  • Direction: *{escape_markdown(trader_plan.get('direction', '?'))}*\n"
        msg3 += f"  • Entry: ${safe_format_float(trader_plan.get('entry_price'))}\n"
        msg3 += f"  • Stop: ${safe_format_float(trader_plan.get('stop_loss'))} (-{safe_format_float(trader_plan.get('stop_pct', 0))}%)\n"
        msg3 += f"  • Target 1: ${safe_format_float(trader_plan.get('target_1'))} R:R 1:{trader_plan.get('risk_reward_1', 0)}\n"
        msg3 += f"  • Target 2: ${safe_format_float(trader_plan.get('target_2'))} R:R 1:{trader_plan.get('risk_reward_2', 0)}\n"
        msg3 += f"  • Position: {trader_plan.get('position_size_pct', 0)}%\n"
        msg3 += f"\n📝 {escape_markdown(trader_plan.get('reasoning', '')[:900])}"
        send_telegram_sync(chat_id, msg3)


def format_trade_plan(trader_plan, indicators, intraday_levels):
    """
    Formats the trading plan for output.

    The entry_type / max_holding_time / invalidation_condition fields are no longer shown:
    they are left over from the old scheme where the model wrote the whole plan. Now levels and
    position size are computed by Python (utils/signals.py), nobody fills these fields,
    and empty "?" ended up in the report. Instead we show what the model actually
    produces now — risks and commentary.
    """
    lines = []
    lines.append(f"📈 *DAY TRADER — TRADE*")
    lines.append(f"💰 Price: ${indicators['price']} | VWAP: ${intraday_levels.get('vwap', '?')}")
    lines.append("")
    lines.append(f"🎯 Direction: *{escape_markdown(trader_plan.get('direction', '?'))}*")
    lines.append(f"📍 Entry: ${safe_format_float(trader_plan.get('entry_price'))}")
    lines.append(f"🔻 Stop: ${safe_format_float(trader_plan.get('stop_loss'))}")
    lines.append(f"🎯 Target 1: ${safe_format_float(trader_plan.get('target_1'))} (R:R 1:{safe_format_float(trader_plan.get('risk_reward_1', 0), 1)})")
    lines.append(f"🎯 Target 2: ${safe_format_float(trader_plan.get('target_2'))} (R:R 1:{safe_format_float(trader_plan.get('risk_reward_2', 0), 1)})")
    lines.append(f"📊 Position: {safe_format_float(trader_plan.get('position_size_pct', 0), 1)}% of the account")
    lines.append(f"✅ Confidence: {safe_format_float(trader_plan.get('confidence', 0) * 100, 0)}%")

    if intraday_levels.get('pivot_reliable') is False:
        gap = safe_format_float(intraday_levels.get('pivot_gap_pct', 0), 1)
        lines.append(f"⚠️ Pivot unreliable (gap {gap}% from the previous session) — stop/targets not taken from Pivot/R1/R2")

    if intraday_levels.get('trend_divergence'):
        lines.append(
            f"⚠️ Trends diverge: 5m {intraday_levels.get('trend_5m', '?')} "
            f"vs 15m {intraday_levels.get('trend_15m', '?')}"
        )

    gemma = trader_plan.get("gemma") or {}
    risks = gemma.get("risks") or []
    if risks:
        lines.append("")
        lines.append("⚠️ *Risks:*")
        for r in risks[:4]:
            lines.append(f"  • {escape_markdown(str(r))}")

    commentary = gemma.get("commentary") or ""
    if commentary:
        lines.append("")
        lines.append(f"💬 {escape_markdown(commentary[:1200])}")

    return "\n".join(lines)