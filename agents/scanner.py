"""
Scanner agent
Runs the full pipeline for several tickers and builds a summary report.
"""

import time
import asyncio
from datetime import datetime
from typing import List, Dict, Any
from config import MAX_NEWS
from utils.helpers import escape_markdown, load_state, save_state, safe_format_float
from utils.signal_history import append_signal, format_history_report
from utils.signals import format_signals
from utils.market_hours import market_status, format_market_status

from data.fetcher import (
    fetch_price_data, fetch_news, fetch_earnings_info, 
    fetch_relative_context, fetch_extended_history, fetch_previous_day_levels
)
from data.indicators import (compute_indicators, compute_intraday_levels, compute_volume_profile,
                             detect_liquidity_levels, compute_moving_averages, check_data_quality)
from agents.junior import run_junior_analyst
from agents.trader import run_trader
from agents.news_agent import analyze_news
from utils.real_flow import build_real_flow_features
from utils.short_volume import get_short_volume
from utils.deepseek_verdict import request_verdict_background


def scan_single_ticker(ticker: str, status_callback=None, current=0, total=0) -> Dict[str, Any]:
    """
    Full pipeline for one ticker (like build_trade_report, but without sending to Telegram).
    Returns structured data for the summary report.
    """
    print(f"\n{'='*60}\n🔍 SCANNING {ticker} ({current}/{total})\n{'='*60}")
    
    if status_callback:
        status_callback(f"📊 Collecting data for {ticker}...")
    
    # 1. DATA COLLECTION
    print(f"\n📊 [1/7] DATA COLLECTION FOR {ticker}")
    df_hourly, df_daily = fetch_price_data(ticker)
    if df_hourly.empty:
        raise ValueError(f"Ticker {ticker} not found")
    data_ok, data_warning = check_data_quality(df_hourly)
    if not data_ok:
        raise ValueError(f"Unreliable data: {data_warning}")
    indicators = compute_indicators(df_hourly, df_daily)
    print(f"   ✅ Price: ${safe_format_float(indicators['price'])} | RSI: {indicators['rsi']:.1f}")
    
    # 2. NEWS
    print(f"\n📰 [2/7] NEWS FOR {ticker}")
    if status_callback:
        status_callback(f"📰 Loading news for {ticker}...")
    
    news = fetch_news(ticker, limit=MAX_NEWS)
    news_analysis = analyze_news(ticker, news, use_cache=True)
    print(f"   ✅ Sentiment: {news_analysis.get('overall_sentiment', 'neutral')}")
    
    # 3. CONTEXT AND LEVELS
    print(f"\n📐 [3/7] CONTEXT AND LEVELS FOR {ticker}")
    if status_callback:
        status_callback(f"📐 Computing levels for {ticker}...")
    
    relative_context = fetch_relative_context(ticker)
    earnings_info = fetch_earnings_info(ticker)
    history = fetch_extended_history(ticker)
    hourly = history["hourly"]
    intraday_levels = compute_intraday_levels(hourly) if not hourly.empty else {}
    # SMA on one year of daily data — see the comment in main.py
    if not history["daily"].empty:
        intraday_levels.update(compute_moving_averages(history["daily"]))
    volume_profile = compute_volume_profile(hourly) if not hourly.empty else {}

    # Clusters of other traders' stops from hourly data + the previous day's real High/Low
    # — as in main.py (see the comment there about the /now vs /scan discrepancy).
    liquidity = detect_liquidity_levels(hourly) if not hourly.empty else {}
    prev_day = fetch_previous_day_levels(ticker)
    liquidity["prev_high"] = prev_day.get("prev_high")
    liquidity["prev_low"] = prev_day.get("prev_low")
    
    # 4. JUNIOR ANALYST
    print(f"\n🧠 [4/7] JUNIOR ANALYST FOR {ticker}")
    if status_callback:
        status_callback(f"🧠 Junior Analyst is analysing {ticker}...")
    
    # Stumpy and Value (optional)
    stumpy_text = ""
    value_text = ""
    stumpy_result = None
    try:
        from data.stumpy_finder import find_similar_patterns, format_stumpy_result
        stumpy_result = find_similar_patterns(ticker, df_hourly)
        stumpy_text = format_stumpy_result(stumpy_result) if stumpy_result else ""
    except:
        pass
    
    try:
        from data.value_investing import compute_value_investing_score, format_value_investing
        value_score = compute_value_investing_score(ticker, indicators)
        value_text = format_value_investing(value_score) if value_score else ""
    except:
        pass
    
    analysis = run_junior_analyst(
        ticker, indicators, news, relative_context, 
        intraday_levels, volume_profile, stumpy_text, value_text,
        news_analysis=news_analysis
    )
    
    # 5. REAL TRADE FLOW (Alpaca/IEX) — diagnostics, NOT part of the score.
    # This used to be the LLM agent agents/orderflow.py (hourly candles without a single
    # real trade inside, measured worse than a coin) — removed entirely, see
    # SCORING_VERSION v3_no_fake_orderflow in utils/signals.py.
    print(f"\n📊 [5/7] REAL FLOW FOR {ticker}")
    if status_callback:
        status_callback(f"📊 Computing real trade flow for {ticker}...")

    real_flow = build_real_flow_features(ticker)
    short_vol = get_short_volume(ticker)

    # 6. RAG + EXTRACTOR — removed from the pipeline (see main.py: it was not in the score,
    # and there was no way to measure the value of book text). The knowledge library on disk
    # is left untouched — a decision not to call it, not a decision to rip it out.
    extracted_rules = {}

    # 7. TRADER
    print(f"\n📈 [7/7] TRADER FOR {ticker}")
    if status_callback:
        status_callback(f"📈 Trader is deciding on {ticker}...")
    
    trader_plan = run_trader(
        ticker, indicators, intraday_levels, volume_profile, liquidity,
        analysis, news, extracted_rules, stumpy_text, value_text,
        news_analysis=news_analysis, relative_context=relative_context,
        stumpy_result=stumpy_result, earnings_info=earnings_info, real_flow=real_flow,
        short_vol=short_vol
    )

    # History of past signals for the ticker (before recording the current one)
    history_text = format_history_report(ticker)
    append_signal(ticker, trader_plan, market_info=market_status(df_hourly))
    request_verdict_background(ticker, trader_plan)

    # Save state
    state = load_state()
    state[ticker] = {
        "price": indicators["price"],
        "rsi": indicators["rsi"],
        "macd": indicators["macd"],
        "volume": indicators["volume"],
        "timestamp": datetime.now().isoformat()
    }
    save_state(state)
    
    print(f"   ✅ {ticker}: {trader_plan.get('direction', '?')} (score: {trader_plan.get('confidence', 0):.0%})")
    
    return {
        "ticker": ticker,
        "price": indicators["price"],
        "rsi": indicators["rsi"],
        "volume": indicators["volume"],
        "direction": trader_plan.get("direction", "?"),
        "entry_price": trader_plan.get("entry_price"),
        "stop_loss": trader_plan.get("stop_loss"),
        "target_1": trader_plan.get("target_1"),
        "target_2": trader_plan.get("target_2"),
        "confidence": trader_plan.get("confidence", 0),
        "risk_reward_1": trader_plan.get("risk_reward_1", 0),
        "risk_reward_2": trader_plan.get("risk_reward_2", 0),
        "position_size_pct": trader_plan.get("position_size_pct", 0),
        "reasoning": trader_plan.get("reasoning", ""),
        "analysis": analysis,
        "real_flow": real_flow,
        "trader_plan": trader_plan,
        "intraday_levels": intraday_levels,
        "relative_context": relative_context,
        "news_analysis": news_analysis,
        "earnings_info": earnings_info,
        "history_text": history_text
    }


def format_single_result(r: Dict, index: int = 1) -> str:
    """
    Detailed block for ONE scan_single_ticker result — ticker, factors,
    levels, position, news, rationale. This used to be hardcoded
    inside build_scan_report only for the top-2 summary; moved into a separate
    function so /autopilot can show a found signal right away in the same
    format, without a second copy that sooner or later diverges from
    the original (the same lesson as with the duplicated /scan earlier).
    """
    ticker_escaped = escape_markdown(r['ticker'])
    direction_escaped = escape_markdown(r.get('direction', '?'))
    confidence = r.get('confidence', 0) * 100

    lines = [f"*{index}. {ticker_escaped}*",
             f"   📈 Direction: {direction_escaped} (confidence: {safe_format_float(confidence, 0)}%)"]

    tp = r.get('trader_plan', {})
    scores, reasons = tp.get('scores'), tp.get('reasons')
    if scores and reasons:
        signals_text = format_signals(scores, reasons)
        lines.append("   " + signals_text.replace("\n", "\n   "))

    entry = r.get('entry_price')
    stop = r.get('stop_loss')
    target1 = r.get('target_1')
    target2 = r.get('target_2')

    if entry is not None and stop is not None:
        lines.append(f"   📍 Entry: ${safe_format_float(entry)} | Stop: ${safe_format_float(stop)}")
        if target1 is not None:
            risk = abs(entry - stop)
            reward = abs(target1 - entry)
            rr = f"1:{reward/risk:.1f}" if risk > 0 else "?"
            lines.append(f"   🎯 Target 1: ${safe_format_float(target1)} (R:R {rr})")
        if target2 is not None:
            risk = abs(entry - stop)
            reward = abs(target2 - entry)
            rr = f"1:{reward/risk:.1f}" if risk > 0 else "?"
            lines.append(f"   🎯 Target 2: ${safe_format_float(target2)} (R:R {rr})")
    else:
        lines.append(f"   📍 Levels: not computed")

    position = r.get('position_size_pct', 0)
    pos_str = f"{position:.0f}%" if position > 0 else "0%"
    lines.append(f"   📊 Position: {pos_str} of the portfolio")

    news_sent = r.get('news_analysis', {}).get('overall_sentiment', 'neutral')
    news_emoji = "😊" if news_sent == "positive" else "😐" if news_sent == "neutral" else "😟"
    lines.append(f"   {news_emoji} News: {news_sent}")

    reasoning_raw = r.get('reasoning', '')
    reasoning_escaped = escape_markdown(reasoning_raw)
    reasoning_short = reasoning_escaped[:150]
    if reasoning_short:
        lines.append(f"   📝 {reasoning_short}...")

    levels = r.get('intraday_levels', {})
    vwap_str = f"${safe_format_float(levels.get('vwap'))}"
    pivot_str = f"${safe_format_float(levels.get('pivot'))}"
    lines.append(f"   📐 VWAP: {vwap_str} | Pivot: {pivot_str}")

    history_text = r.get('history_text', '')
    if history_text:
        lines.append("")
        lines.append(f"   {history_text}".replace("\n", "\n   "))

    return "\n".join(lines)


def build_scan_report(results: List[Dict]) -> str:
    """
    Builds the summary report for all tickers for Telegram.
    """
    lines = []
    
    # Header
    lines.append("📊 *SCAN RESULTS*\n")
    lines.append(f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    lines.append(f"📊 Processed: {len(results)} tickers")

    # Closed-market note: otherwise a day later in the chat an evening scan cannot be told
    # apart from a live one — prices and levels look the same
    market_note = format_market_status(market_status())
    if market_note:
        lines.append("")
        lines.append(market_note)

    lines.append("")
    lines.append("═" * 40)
    lines.append("")
    
    # Sort by confidence
    sorted_results = sorted(results, key=lambda x: x.get("confidence", 0), reverse=True)
    
    # TOP 2
    top2 = [r for r in sorted_results if r.get("confidence", 0) > 0][:2]
    if top2:
        lines.append("🏆 *TOP 2 RECOMMENDATIONS:*\n")
        for i, r in enumerate(top2, 1):
            medal = "🥇" if i == 1 else "🥈"
            direction_escaped = escape_markdown(r.get('direction', '?'))
            confidence = r.get('confidence', 0) * 100
            ticker_escaped = escape_markdown(r['ticker'])
            emoji = "🟢" if r.get('direction') in ["LONG", "BUY"] else "🟡" if r.get('direction') in ["HOLD", "WAIT"] else "🔴"
            lines.append(
                f"{medal} *{ticker_escaped}* — {emoji} *{direction_escaped}* "
                f"(confidence: {safe_format_float(confidence, 0)}%)"
            )
        lines.append("")
    
    # Full table
    lines.append("📋 *FULL TABLE:*\n")
    lines.append("| # | Ticker | Direction | Score | Confidence | Price | R:R1 | Size |")
    lines.append("|---|-------|-------------|------|-------------|------|------|--------|")
    
    for i, r in enumerate(sorted_results, 1):
        ticker_escaped = escape_markdown(r['ticker'])
        direction_escaped = escape_markdown(r.get('direction', '?'))
        confidence = r.get('confidence', 0) * 100
        price = r.get('price', 0)
        
        if r.get('direction') in ["LONG", "BUY"]:
            emoji = "🟢"
        elif r.get('direction') in ["NO_TRADE", "OUT"]:
            emoji = "⚪"
        else:
            emoji = "🔴"
        
        rr1 = r.get('risk_reward_1')
        if rr1 is None or rr1 == 0:
            rr1_str = "—"
        else:
            rr1_str = f"1:{rr1:.1f}"
        
        position = r.get('position_size_pct', 0)
        if position == 0:
            pos_str = "0%"
        else:
            pos_str = f"{position:.0f}%"

        score = r.get('trader_plan', {}).get('direction_score')
        score_str = f"{score:+d}" if isinstance(score, int) else "—"

        lines.append(
            f"| {i} | {ticker_escaped} | {emoji}{direction_escaped} | {score_str} | {safe_format_float(confidence, 0)}% | ${safe_format_float(price)} | {rr1_str} | {pos_str} |"
        )
    
    lines.append("")
    
    # Details for the TOP 2 — in the same format /autopilot uses for
    # each found signal (see format_single_result above).
    if top2:
        lines.append("📊 *DETAILS FOR TOP 2 TICKERS:*\n")
        for i, r in enumerate(top2, 1):
            lines.append(format_single_result(r, i))
            lines.append("")
    else:
        lines.append("📊 No active signals (all tickers NO_TRADE).")

    return "\n".join(lines)


def print_scan_results(results):
    """Prints nice scan results to the console (as in Telegram)."""
    print("\n" + "="*80)
    print("📊  SCAN RESULTS".center(80))
    print("="*80)
    print(f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    print(f"📊 Processed: {len(results)} tickers\n")
    
    # Sort by confidence
    sorted_results = sorted(results, key=lambda x: x.get("confidence", 0), reverse=True)
    
    # TOP 2
    top2 = [r for r in sorted_results if r.get("confidence", 0) > 0][:2]
    if top2:
        print("🏆 TOP 2 RECOMMENDATIONS:")
        for i, r in enumerate(top2, 1):
            medal = "🥇" if i == 1 else "🥈"
            direction = r.get('direction', '?')
            confidence = r.get('confidence', 0) * 100
            emoji = "🟢" if direction in ["LONG", "BUY"] else "🟡" if direction in ["HOLD", "WAIT"] else "🔴"
            print(f"   {medal} {r['ticker']} — {emoji} {direction} (confidence: {safe_format_float(confidence, 0)}%)")
        print()
    
    # FULL TABLE
    print("📋 FULL TABLE:")
    print("| # | Ticker | Direction | Score | Confidence | Price | R:R1 | Size |")
    print("|---|-------|-------------|------|-------------|------|------|--------|")
    
    for i, r in enumerate(sorted_results, 1):
        direction = r.get('direction', '?')
        confidence = r.get('confidence', 0) * 100
        price = r.get('price', 0)
        
        if direction in ["LONG", "BUY"]:
            emoji = "🟢"
        elif direction in ["NO_TRADE", "OUT"]:
            emoji = "⚪"
        else:
            emoji = "🔴"
        
        rr1 = r.get('risk_reward_1')
        if rr1 is None or rr1 == 0:
            rr1_str = "—"
        else:
            rr1_str = f"1:{rr1:.1f}"
        
        position = r.get('position_size_pct', 0)
        if position == 0:
            pos_str = "0%"
        else:
            pos_str = f"{position:.0f}%"
        
        score = r.get('trader_plan', {}).get('direction_score')
        score_str = f"{score:+d}" if isinstance(score, int) else "—"

        print(f"| {i} | {r['ticker']} | {emoji}{direction} | {score_str} | {safe_format_float(confidence, 0)}% | ${safe_format_float(price)} | {rr1_str} | {pos_str} |")
    
    print()
    
    # DETAILS FOR TOP 2
    if top2:
        print("📊 DETAILS FOR TOP 2 TICKERS:")
        print()
        for i, r in enumerate(top2, 1):
            direction = r.get('direction', '?')
            confidence = r.get('confidence', 0) * 100
            price = r.get('price', 0)
            
            print(f"{i}. {r['ticker']}")
            print(f"   📈 Direction: {direction} (confidence: {safe_format_float(confidence, 0)}%)")
            
            entry = r.get('entry_price')
            stop = r.get('stop_loss')
            target1 = r.get('target_1')
            target2 = r.get('target_2')
            
            if entry is not None and stop is not None:
                print(f"   📍 Entry: ${safe_format_float(entry)} | Stop: ${safe_format_float(stop)}")
                if target1 is not None:
                    risk = abs(entry - stop)
                    reward = abs(target1 - entry)
                    rr = f"1:{reward/risk:.1f}" if risk > 0 else "?"
                    print(f"   🎯 Target 1: ${safe_format_float(target1)} (R:R {rr})")
                if target2 is not None:
                    risk = abs(entry - stop)
                    reward = abs(target2 - entry)
                    rr = f"1:{reward/risk:.1f}" if risk > 0 else "?"
                    print(f"   🎯 Target 2: ${safe_format_float(target2)} (R:R {rr})")
            else:
                print(f"   📍 Levels: not computed")
            
            position = r.get('position_size_pct', 0)
            pos_str = f"{position:.0f}%" if position > 0 else "0%"
            print(f"   📊 Position: {pos_str} of the portfolio")
            
            news_analysis = r.get('news_analysis', {})
            news_sentiment = news_analysis.get('overall_sentiment', 'mixed')
            news_emoji = "😊" if news_sentiment == "positive" else "😐" if news_sentiment == "neutral" else "😟"
            print(f"   {news_emoji} News: {news_sentiment}")
            
            reasoning = r.get('reasoning', '')
            if reasoning:
                print(f"   📝 {reasoning[:150]}...")
            
            levels = r.get('intraday_levels', {})
            vwap = levels.get('vwap')
            pivot = levels.get('pivot')
            
            vwap_str = f"${safe_format_float(vwap)}"
            pivot_str = f"${safe_format_float(pivot)}"
            
            print(f"   📐 VWAP: {vwap_str} | Pivot: {pivot_str}")
            print()
    else:
        print("📊 No active signals (all tickers NO_TRADE).")
    
    print("="*80)


async def scan_tickers(tickers: List[str], update=None, context=None) -> str:
    """
    Main scan function.
    Runs the full pipeline for each ticker sequentially.
    """
    if not tickers:
        return "❌ No tickers to scan"
    
    tickers = tickers[:10]  # 10 at most
    
    # Send the start message
    status_msg = None
    if update:
        status_msg = await update.message.reply_text(
            f"🔍 *Starting a scan of {len(tickers)} tickers...*\n"
            f"⏱️ Expect ~{len(tickers) * 2} minutes\n"
            f"📊 Tickers: {', '.join(tickers)}\n\n"
            f"🔄 Progress: 0/{len(tickers)}"
        )
    
    results = []
    total = len(tickers)
    # Grab the loop reference here: update_status is called from the worker thread,
    # where asyncio.get_event_loop() will not work
    loop = asyncio.get_running_loop()

    # Status update function
    def update_status(text):
        if status_msg:
            try:
                asyncio.run_coroutine_threadsafe(
                    status_msg.edit_text(
                        f"🔍 *Scanning {total} tickers...*\n"
                        f"⏱️ ~{(total - len(results)) * 2} minutes left\n"
                        f"📊 Tickers: {', '.join(tickers)}\n\n"
                        f"🔄 {text}"
                    ),
                    loop
                )
            except:
                pass

    for i, ticker in enumerate(tickers, 1):
        try:
            update_status(f"🔄 {i-1}/{total} — processing *{ticker}*...")
            # scan_single_ticker is synchronous and takes minutes (LLM calls). Calling it
            # directly from here blocked the event loop for the whole scan: the bot stopped
            # responding, progress did not update, and the HTTP connection to Telegram had time
            # to go stale — and the finished report failed with ReadTimeout.
            result = await asyncio.to_thread(scan_single_ticker, ticker, update_status, i, total)
            results.append(result)
        except Exception as e:
            print(f"   ❌ Error {ticker}: {e}")
            results.append({
                "ticker": ticker,
                "error": str(e),
                "direction": "ERROR",
                "confidence": 0
            })

        # Pause between tickers (await, otherwise we block the loop again)
        if i < total:
            await asyncio.sleep(2)
    
    # Final report
    if status_msg:
        try:
            await status_msg.edit_text("📊 *Building the final report...*")
        except:
            pass
    
    report = build_scan_report(results)
    
    # ✅ CONSOLE OUTPUT
    print_scan_results(results)
    
    # Update the status in Telegram
    if status_msg:
        sorted_results = sorted(
            [r for r in results if r.get("confidence", 0) > 0],
            key=lambda x: x["confidence"],
            reverse=True
        )
        try:
            if sorted_results:
                text = (
                    f"✅ *Scan complete!*\n"
                    f"📊 Processed: {len(results)} tickers\n"
                    f"🏆 Top 1: {sorted_results[0]['ticker']} ({sorted_results[0]['direction']})"
                )
                if len(sorted_results) > 1:
                    text += f"\n🥈 Top 2: {sorted_results[1]['ticker']} ({sorted_results[1]['direction']})"
                await status_msg.edit_text(text)
        except:
            pass
    
    return report