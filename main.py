import time
import asyncio
import threading
from datetime import datetime
from telegram import Update
from telegram.error import BadRequest
from telegram.ext import Application, CommandHandler, TypeHandler
from config import (
    TELEGRAM_BOT_TOKEN, MY_CHAT_ID, MAX_LINE_LOG,
    TRADER_MODEL, JUNIOR_MODEL
)
from data.fetcher import (fetch_price_data, fetch_news, fetch_earnings_info, fetch_relative_context,
                          fetch_extended_history, fetch_previous_day_levels)
from data.indicators import (compute_indicators, compute_intraday_levels, compute_volume_profile,
                             detect_liquidity_levels, compute_moving_averages,
                             check_data_quality)
from agents.junior import run_junior_analyst
from utils.real_flow import build_real_flow_features
from utils.short_volume import get_short_volume
from utils.deepseek_verdict import request_verdict_background
from agents.trader import run_trader
from agents.news_agent import analyze_news, format_news_for_telegram
from telegram_bot.reporter import send_telegram_sync, send_trader_report, format_trade_plan
from telegram_bot.handlers import (cmd_go, cmd_now, cmd_status, cmd_stop, cmd_help,
                                   scheduled_check, cmd_news, cmd_stats, cmd_screen, cmd_lab, cmd_earnings, cmd_insiders,
                                   cmd_massscreen, cmd_autoscan, cmd_queue, cmd_refilter, cmd_find,
                                   cmd_morning, morning_job, intraday_resolve_job, cmd_autopilot,
                                   hourly_autopilot_job, daily_digest_job)
from utils.helpers import load_state, save_state, safe_format_float
from utils.signal_history import append_signal, format_history_report
from utils.market_hours import market_status, format_market_status
from agents.scanner import scan_tickers


def build_trade_report(ticker):
    print(f"\n{'='*60}\n🚀 STARTING DAY-TRADE REPORT FOR {ticker}\n{'='*60}")
    
    # 1. DATA COLLECTION
    print(f"\n📊 [1/7] DATA COLLECTION")
    df_hourly, df_daily = fetch_price_data(ticker)
    if df_hourly.empty:
        raise ValueError(f"Ticker {ticker} not found")
    data_ok, data_warning = check_data_quality(df_hourly)
    if not data_ok:
        # Analysing a stock whose "current price" is not a trade price
        # is pointless: VWAP, pivots and levels would be built on nothing.
        raise ValueError(f"Unreliable data for {ticker}: {data_warning}")
    indicators = compute_indicators(df_hourly, df_daily)
    print(f"   ✅ Price: ${safe_format_float(indicators['price'])} | RSI: {indicators['rsi']:.1f} | MACD: {indicators['macd']:.3f}")
    
    # 1.5 STUMPY
    print(f"\n🔍 [1.5/7] STUMPY PATTERN FINDER")
    from data.stumpy_finder import find_similar_patterns, format_stumpy_result
    stumpy_result = find_similar_patterns(ticker, df_hourly)
    stumpy_text = format_stumpy_result(stumpy_result) if stumpy_result else ""
    if stumpy_text:
        print(stumpy_text)

    # 1.6 VALUE INVESTING
    print(f"\n📊 [1.6/7] VALUE INVESTING CHECKLIST")
    from data.value_investing import compute_value_investing_score, format_value_investing
    value_score = compute_value_investing_score(ticker, indicators)
    value_text = format_value_investing(value_score) if value_score else ""
    if value_text:
        print(value_text)

    # 2. NEWS
    print(f"\n📰 [2/7] NEWS")
    news = fetch_news(ticker)
    for i, n in enumerate(news[:5]):
        print(f"   [{i+1}] {n['title'][:MAX_LINE_LOG]}")
    
    news_analysis = analyze_news(ticker, news, use_cache=True)
    print(f"   ✅ Sentiment: {news_analysis.get('overall_sentiment', 'neutral')}")
    print(f"   ✅ Themes: {', '.join(news_analysis.get('key_themes', [])[:3])}")
    news_report = format_news_for_telegram(ticker, news_analysis, news)
    if news_report:
        threading.Thread(target=send_telegram_sync, args=(MY_CHAT_ID, news_report)).start()
    
    # 3. CONTEXT AND LEVELS
    print(f"\n📐 [3/7] CONTEXT AND LEVELS")
    relative_context = fetch_relative_context(ticker)
    earnings_info = fetch_earnings_info(ticker)
    history = fetch_extended_history(ticker)
    hourly = history["hourly"]
    intraday_levels = compute_intraday_levels(hourly) if not hourly.empty else {}
    # Moving averages are computed on one year of DAILY data (otherwise there is nothing to build
    # SMA50/200 on) and put into intraday_levels — the market-quality score reads them from there.
    # Previously compute_moving_averages was not called anywhere, and sma20 was substituted with
    # the price itself: the comparison price > price is always false, so every report showed
    # "Trend ⭐⭐☆☆☆ (Price below SMA20)" regardless of the market.
    if not history["daily"].empty:
        intraday_levels.update(compute_moving_averages(history["daily"]))
    volume_profile = compute_volume_profile(hourly) if not hourly.empty else {}
    # Clusters of other traders' stops come from hourly data, and the previous DAY's High/Low
    # from daily data (as in /scan). Previously only detect_liquidity_levels was here,
    # and its prev_high/prev_low were the high/low of the previous HOUR:
    # /now and /scan gave different stops for the same ticker.
    liquidity = detect_liquidity_levels(hourly) if not hourly.empty else {}
    prev_day = fetch_previous_day_levels(ticker)
    liquidity["prev_high"] = prev_day.get("prev_high")
    liquidity["prev_low"] = prev_day.get("prev_low")
    print(f"   ✅ VWAP: ${safe_format_float(intraday_levels.get('vwap'))} | POC: ${safe_format_float(volume_profile.get('poc'))}")
    print(f"   ✅ Pivot: ${safe_format_float(intraday_levels.get('pivot'))} | R1: ${safe_format_float(intraday_levels.get('r1'))} | S1: ${safe_format_float(intraday_levels.get('s1'))}")
    
    # === TELEGRAM 1: PYTHON METRICS ===
    python_metrics_msg = f"""📊 *PYTHON METRICS — {ticker}*

*Indicators:*
• Price: ${safe_format_float(indicators['price'])} | RSI: {indicators['rsi']}
• MACD: {indicators['macd']} / signal: {indicators['macd_signal']}
• BB: ${safe_format_float(indicators['bb_lower'])} — ${safe_format_float(indicators['bb_upper'])}
• Volume: {indicators['volume']:,}

*Levels:*
• VWAP: ${safe_format_float(intraday_levels.get('vwap'))} (price {intraday_levels.get('vwap_position', '?')}, {safe_format_float(intraday_levels.get('distance_to_vwap', 0))}%)
• Pivot: ${safe_format_float(intraday_levels.get('pivot'))}
• R1: ${safe_format_float(intraday_levels.get('r1'))} | S1: ${safe_format_float(intraday_levels.get('s1'))}
• POC: ${safe_format_float(volume_profile.get('poc'))}
• ATR: ${safe_format_float(intraday_levels.get('atr_hourly'))}

*Context:*
• Volume: {safe_format_float(relative_context.get('volume_ratio', 1))}x (full-day projection)
• 52W position: {safe_format_float(relative_context.get('position_in_range_pct', '?'))}%
• Trend 5m: {intraday_levels.get('trend_5m', '?')} | 15m: {intraday_levels.get('trend_15m', '?')}
• Earnings in: {earnings_info.get('days_until', '?')} d.
"""
    threading.Thread(target=send_telegram_sync, args=(MY_CHAT_ID, python_metrics_msg)).start()
    
    # 4. JUNIOR
    analysis = run_junior_analyst(
        ticker, indicators, news, relative_context, 
        intraday_levels, volume_profile, stumpy_text, value_text,
        news_analysis=news_analysis
    )
    
    # === TELEGRAM 2: JUNIOR ===
    summary = analysis.get("executive_summary", "")
    if isinstance(summary, list):
        summary = " ".join(summary)
    risks = analysis.get('risks', [])
    opps = analysis.get('opportunities', [])
    junior_msg = f"""🔍 *JUNIOR ANALYST — {ticker}*

📝 {summary}

⚠️ *Risks:* {', '.join(risks[:3]) if risks else 'none'}
✅ *Opportunities:* {', '.join(opps[:3]) if opps else 'none'}
"""
    threading.Thread(target=send_telegram_sync, args=(MY_CHAT_ID, junior_msg)).start()
    
    # 5. REAL TRADE FLOW (Alpaca/IEX) — diagnostics, NOT part of the score.
    # This used to be agents/orderflow.py: an LLM shown hourly
    # candles (not a single real trade in them) and asked to make up
    # a "large player"/"divergence". On 169 signals with a counterfactual this gave
    # 36.8% hits at a 45% base rate — worse than a coin, so it was removed from
    # the pipeline and the formula entirely (see SCORING_VERSION v3_no_fake_orderflow
    # in utils/signals.py). In its place — real trades/quotes, but for now
    # they are only logged: first counterfactuals must accumulate to measure whether there is
    # any predictive value before letting it into the formula (the same
    # discipline as crypto_daytrader/utils/flow_features.py).
    real_flow = build_real_flow_features(ticker)
    if real_flow and not real_flow.get("error") and real_flow.get("n_trades") != 0:
        rf_lines = [f"📡 *REAL FLOW (Alpaca/IEX, for reference, does not affect the decision) — {ticker}*", ""]
        for m in sorted(k for k in real_flow if k.startswith("flow_")):
            w = real_flow[m]
            rf_lines.append(
                f"• {w['window_min']}m: {w['n_trades']} trades, "
                f"delta ${w['delta_quote']:+,.0f} ({(w['delta_pct'] or 0):+.1%})"
            )
        threading.Thread(target=send_telegram_sync, args=(MY_CHAT_ID, "\n".join(rf_lines))).start()

    # 5c. SHORT SHARE (FINRA Reg SHO) — also diagnostics, also not part of the score.
    # An external fact about volume structure, not a reinterpretation of the same candle —
    # see the utils/short_volume.py header. Lag of at least one trading day.
    short_vol = get_short_volume(ticker)
    if short_vol and short_vol.get("short_ratio") is not None:
        threading.Thread(target=send_telegram_sync, args=(MY_CHAT_ID,
            f"📉 *Short share (FINRA, for {short_vol['date']}) — {ticker}*\n"
            f"{short_vol['short_ratio']:.1%} of the day's volume"
        )).start()

    # 6. RAG + EXTRACTOR — removed from the pipeline. The extracted "rules" never
    # took part in the score (Python computes the decision without them, see the architecture
    # principle), and there was no way to measure whether book text helps the trade description or
    # just burns tokens and time — the same reasoning as for
    # the removed Order Flow, only without a counterfactual measurement (here there
    # could not be one: it is not a number but text in a prompt). The knowledge library
    # (knowledge/*.py, agents/extractor.py) is left untouched on disk —
    # a decision not to call it, not a decision to rip it out.
    extracted_rules = {}

    # 7. DAY TRADER
    trader_plan = run_trader(
        ticker, indicators, intraday_levels, volume_profile, liquidity,
        analysis, news, extracted_rules, stumpy_text, value_text,
        news_analysis=news_analysis, relative_context=relative_context,
        stumpy_result=stumpy_result, earnings_info=earnings_info, real_flow=real_flow,
        short_vol=short_vol
    )
    
    # === MARKET STATUS (which session the plan belongs to) ===
    market_info = market_status(df_hourly)

    # === PAST SIGNAL HISTORY FOR THIS TICKER (before recording the current one) ===
    history_text = format_history_report(ticker)
    append_signal(ticker, trader_plan, market_info=market_info)

    # Parallel DeepSeek verdict — fire-and-forget, does not block and does not
    # affect trader_plan/the message below. See utils/deepseek_verdict.py.
    request_verdict_background(ticker, trader_plan)

    # === TELEGRAM 4: TRADE PLAN ===
    # On NO_TRADE there are no levels, and format_trade_plan would print "Entry: $? / Stop: $? /
    # R:R 1:0.0" — a meaningless template with empty fields. For this case the trader
    # already has a detailed breakdown (factors for/against, what is missing for an entry,
    # market quality) — use that.
    no_trade_report = trader_plan.get("no_trade_report")
    if no_trade_report:
        trade_plan_text = no_trade_report
    else:
        trade_plan_text = format_trade_plan(trader_plan, indicators, intraday_levels)
    market_note = format_market_status(market_info)
    if market_note:
        trade_plan_text = f"{market_note}\n\n{trade_plan_text}"
    if history_text:
        trade_plan_text += f"\n\n{history_text}"
    print(f"\n📐 TRADE PLAN:")
    for line in trade_plan_text.split('\n'):
        print(f"   │ {line[:MAX_LINE_LOG]}")

    # The final plan is NOT sent here — both callers (cmd_now,
    # scheduled_check) already send the returned trade_plan_text themselves, each
    # to its own chat (update.effective_chat / job.data["chat_id"]). Sending from here
    # would duplicate it to MY_CHAT_ID as a second message — exactly what
    # showed up as "the verdict is sent twice". The intermediate messages above
    # (news_report, python_metrics_msg, junior_msg, rf_lines) do not have this problem
    # — they are not returned anywhere and are sent only here.

    state = load_state()
    state[ticker] = {
        "price": indicators["price"], 
        "rsi": indicators["rsi"], 
        "macd": indicators["macd"], 
        "volume": indicators["volume"], 
        "timestamp": datetime.now().isoformat()
    }
    save_state(state)
    print(f"\n💾 State saved\n{'='*60}\n")
    
    return trade_plan_text


async def cmd_scan(update, context):
    print("\n🔍 [/scan] command received", flush=True)
    args = context.args
    if len(args) < 2:
        await update.message.reply_text(
            "❌ /scan TICKER1 TICKER2 TICKER3 ... (up to 10 tickers)\n"
            "Example: /scan AAPL TSLA NVDA MSFT GOOGL"
        )
        return
    
    tickers = [t.upper() for t in args[:10]]
    report = await scan_tickers(tickers, update, context)

    parts = [report[i:i+3800] for i in range(0, len(report), 3800)]
    for idx, part in enumerate(parts, 1):
        # Scanning 10 tickers takes minutes — losing the whole finished report
        # to a single Telegram network timeout is not acceptable, so we retry
        # each part separately and continue even on failure.
        for attempt in range(3):
            try:
                await update.message.reply_text(part, parse_mode="Markdown")
                break
            except BadRequest as e:
                # Markup did not match (e.g. '*' was split at a part boundary) —
                # send as plain text so that part of the report is not lost
                print(f"   ⚠️ Markdown rejected ({e}), sending as plain text")
                try:
                    await update.message.reply_text(part)
                    break
                except Exception as e2:
                    print(f"   ⚠️ Could not send part {idx}: {e2}")
            except Exception as e:
                print(f"   ⚠️ Error sending part {idx}/{len(parts)} "
                      f"(attempt {attempt+1}/3): {e}")
                await asyncio.sleep(2 ** attempt)
        # IMPORTANT: await, not time.sleep — a blocking sleep in an async handler
        # stops the bot's whole event loop for that time
        await asyncio.sleep(0.5)


def main():
    print(f"🤖 Day Trader Agent v2")
    print(f"Models: JUNIOR={JUNIOR_MODEL} | TRADER={TRADER_MODEL}")

    # python-telegram-bot's default timeouts (5 s) are a bit short for long
    # scanner reports — raise them so sending does not fail with ReadTimeout
    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .read_timeout(30)
        .write_timeout(30)
        .connect_timeout(15)
        .pool_timeout(15)
        .build()
    )

    async def on_error(update, context):
        """Without it, any unhandled error dumps a full traceback into the log."""
        print(f"[TELEGRAM ERROR] {context.error}")

    app.add_error_handler(on_error)

    async def log_update(update, context):
        """
        Logs EVERY incoming message before the command handlers (group=-1).
        Without this it was impossible to tell "the command never reached the bot" from "the handler
        failed silently" — the log was empty in both cases.
        """
        msg = getattr(update, "effective_message", None)
        if msg is not None and msg.text:
            user = getattr(update.effective_user, "username", None) or "?"
            print(f"\n📥 INCOMING from @{user} (chat {msg.chat_id}): {msg.text[:200]}", flush=True)

    # group=-1 -> runs before the command handlers and does not interfere with them
    app.add_handler(TypeHandler(Update, log_update), group=-1)

    # Descriptions for the Telegram menu (what is suggested when typing "/").
    # Without set_my_commands the list in the client stays empty, and new commands
    # are not visible — they work, but only if typed by hand.
    COMMAND_MENU = [
        ("screen", "Quick candidate screen from Yahoo"),
        ("mass", "Mass pass over ~100 stocks in batches"),
        ("autoscan", "Pass over the WHOLE market + auto-scan of the top 10 LONG"),
        ("queue", "Next N LONG candidates from the /autoscan list"),
        ("refilter", "Recompute the /autoscan filter without a new market pass"),
        ("find", "Collect N candidates by stepping down the relaxation ladder"),
        ("autopilot", "Search for N real LONG signals with the expensive analysis"),
        ("morning", "Auto-build the list every morning before the open: on/off/now/status"),
        ("lab", "Measure whether filter thresholds give a real edge"),
        ("earnings", "Stocks reporting within a given window of days (whole market)"),
        ("insiders", "Form 4 insider purchases over N days (whole market)"),
        ("scan", "Full analysis of several tickers"),
        ("now", "Full analysis of one ticker"),
        ("go", "Monitor a ticker every N minutes"),
        ("stats", "Signal statistics"),
        ("news", "Whole-market overview; /news TICKER — for one stock"),
        ("status", "Active monitors"),
        ("stop", "Stop monitoring"),
        ("help", "Help"),
    ]

    async def register_menu(application):
        # In a freshly started process no hunt can be running: the flag in
        # state.json is a leftover of an interrupted run (see autopilot_busy in handlers).
        try:
            from utils.helpers import load_state, save_state
            st = load_state()
            if st.get("_autopilot_running"):
                st["_autopilot_running"] = False
                save_state(st)
                print("⚠️ Cleared the hunt flag left over from the previous run")
        except Exception as e:
            print(f"⚠️ Could not check the hunt flag: {e}")
        try:
            await application.bot.set_my_commands(COMMAND_MENU)
            print(f"✅ Command menu registered ({len(COMMAND_MENU)} items)")
        except Exception as e:
            print(f"⚠️ Could not register the command menu: {e}")

    app.post_init = register_menu

    for cmd, handler in [
        ("go", cmd_go),
        ("now", cmd_now), 
        ("status", cmd_status), 
        ("stop", cmd_stop), 
        ("help", cmd_help), 
        ("start", cmd_help), 
        ("news", cmd_news),
        ("scan", cmd_scan),
        ("stats", cmd_stats),
        ("screen", cmd_screen),
        ("mass", cmd_massscreen),
        ("autoscan", cmd_autoscan),
        ("queue", cmd_queue),
        ("refilter", cmd_refilter),
        ("find", cmd_find),
        ("autopilot", cmd_autopilot),
        ("morning", cmd_morning),
        ("lab", cmd_lab),
        ("earnings", cmd_earnings),
        ("insiders", cmd_insiders)
    ]:
        app.add_handler(CommandHandler(cmd, handler))

    # Morning auto-build. The job is ALWAYS scheduled, and the decision
    # "run today or not" is made by the job itself (see morning_job): otherwise
    # enabling it via /morning would require restarting the bot for the job
    # to appear in the queue at all.
    #
    # Time with an EXPLICIT zone, not naive: a naive one would silently depend on the
    # container's TZ — the very hidden dependency that would shift outcome resolution
    # by 4-5 hours when TZ changes in docker-compose, without a single
    # error in the log. Here that dependency is closed.
    from datetime import time as dtime
    from zoneinfo import ZoneInfo

    from agents.morning_scan import load_config as load_morning_config

    mcfg = load_morning_config()
    app.job_queue.run_daily(
        morning_job,
        time=dtime(mcfg["hour"], mcfg["minute"],
                   tzinfo=ZoneInfo("America/New_York")),
        name="morning_scan",
    )
    print(f"🌅 Morning auto-build scheduled at "
          f"{mcfg['hour']:02d}:{mcfg['minute']:02d} ET — "
          + ("enabled" if mcfg["enabled"] else "disabled, /morning on"))

    # Resolution of today's signals by the daily rule. Unlike the morning
    # auto-build it has no on/off switch: it is not a source of messages of its own accord
    # but recording outcomes of signals already created. Not resolving them today
    # means losing the data for good (5-minute bars live ~60 days).
    app.job_queue.run_daily(
        intraday_resolve_job,
        time=dtime(15, 35, tzinfo=ZoneInfo("America/New_York")),
        name="intraday_resolve",
    )
    print("🔔 Signal resolution by the daily rule scheduled at 15:35 ET")

    # Hourly autopilot — once an hour looks for up to 5 signals while the trading
    # session is open (the job itself checks is_trading_day/is_within_market_hours and
    # exits quietly if it is not allowed now or another hunt is already running — see the
    # hourly_autopilot_job docstring). Not tied to a specific time
    # (run_repeating, not run_daily): the first tick 5 minutes after the bot
    # starts, then every hour, regardless of when the bot was started —
    # the session check filters out nights/weekends on every tick anyway.
    app.job_queue.run_repeating(
        hourly_autopilot_job,
        interval=3600,
        first=300,
        name="hourly_autopilot",
    )
    print("🕐 Hourly autopilot scheduled: every hour during the session, up to 5 signals per tick")

    # Daily digest — the only scheduled message to the chat (TELEGRAM_SCHEDULED_QUIET).
    from config import DAILY_DIGEST_TIME
    app.job_queue.run_daily(
        daily_digest_job,
        time=dtime(*DAILY_DIGEST_TIME, tzinfo=ZoneInfo("America/New_York")),
        name="daily_digest",
    )
    print(f"📊 Daily digest scheduled at {DAILY_DIGEST_TIME[0]:02d}:{DAILY_DIGEST_TIME[1]:02d} ET")

    def startup():
        time.sleep(5)
        try:
            pass
        except Exception as e:
            print(f"[EXAMPLE] Error: {e}")
    
    threading.Thread(target=startup, daemon=True).start()
    print("✅ Waiting for commands...")
    # bootstrap_retries=-1 — RETRY THE FIRST TELEGRAM CALL FOREVER.
    # The default is zero: one failed get_me at startup throws TimedOut/
    # Bad Gateway out and kills the process — the same class of failure already caught
    # in another bot ("eight restarts in a row"). Here
    # waiting is better than crashing: the bot cannot do anything without
    # Telegram anyway, and crashing with exit(0)/restarting fixes nothing by itself.
    app.run_polling(bootstrap_retries=-1)


if __name__ == "__main__":
    main()