import asyncio
import os
from telegram import Update
from telegram.error import BadRequest
from telegram.ext import ContextTypes
from data.fetcher import fetch_price_data, fetch_news
from telegram_bot.reporter import send_telegram_sync, split_for_telegram
from agents.prompts import build_news_prompt, format_news_report
from utils.ollama import call_ollama_json
from config import NEWS_MODEL, MAX_NEWS
from utils.schemas import NEWS_SCHEMA
from data.news_filter import filter_relevant_news
from agents.news_agent import analyze_news, format_news_for_telegram
from agents.market_overview import build_market_overview
from agents.filter_lab import report as lab_report, list_sweeps
from data.fetcher import fetch_news
from agents.scanner import scan_tickers, scan_single_ticker  # ← new import
from utils.signal_history import (compute_performance_stats, format_performance_stats,
                                  compute_score_distribution, format_score_distribution)
from utils.market_hours import is_within_market_hours



async def scheduled_check(context):
    from main import build_trade_report
    job = context.job

    # Building a plan outside the trading session is pointless: the price is yesterday's close,
    # a plan "for today until the close" is no longer relevant. Quietly skip the tick so as not
    # to burn Ollama calls all night and weekend and not spam the user.
    # Checked by the clock only (without df) — enough to cut off nights and
    # weekends; a rare holiday costs one extra run.
    if not is_within_market_hours():
        print(f"[SKIP] {job.data['ticker']}: market closed, tick skipped")
        return

    try:
        # As in /now: the analysis is synchronous and takes minutes, so we move it to
        # a thread. Otherwise every monitoring tick froze the whole bot — with
        # several active tickers it would hardly respond to commands.
        msg = await asyncio.to_thread(build_trade_report, job.data["ticker"])
        await asyncio.to_thread(send_telegram_sync, job.data["chat_id"], msg)
    except Exception as e:
        print(f"[ERROR] {job.data['ticker']}: {e}")


async def cmd_go(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) < 2:
        await update.message.reply_text("❌ /go TICKER INTERVAL_MINUTES")
        return
    ticker = args[0].upper()
    try:
        interval = int(args[1])
        if interval < 5:
            raise ValueError
    except:
        await update.message.reply_text("❌ Interval ≥ 5 min")
        return
    if context.job_queue.get_jobs_by_name(f"monitor_{ticker}"):
        await update.message.reply_text(f"⚠️ Already monitoring {ticker}")
        return
    try:
        df_hourly, _ = fetch_price_data(ticker)
        if df_hourly.empty:
            await update.message.reply_text(f"❌ Ticker {ticker} not found")
            return
    except:
        await update.message.reply_text(f"❌ Data error")
        return
    context.job_queue.run_repeating(
        scheduled_check, 
        interval=interval*60, 
        first=interval*60, 
        name=f"monitor_{ticker}", 
        data={"ticker": ticker, "chat_id": update.effective_chat.id}
    )
    msg = f"👁️ Monitoring *{ticker}* every {interval} min."
    if not is_within_market_hours():
        msg += "\n⏸ The market is closed now — checks will start when the session opens (9:30 ET)."
    await update.message.reply_text(msg, parse_mode="Markdown")


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("❌ /stop TICKER or /stop all")
        return
    arg = context.args[0].upper()
    if arg == "ALL":
        for job in context.job_queue.jobs():
            job.schedule_removal()
        await update.message.reply_text(f"🛑 Everything stopped")
        return
    jobs = context.job_queue.get_jobs_by_name(f"monitor_{arg}")
    if jobs:
        for job in jobs:
            job.schedule_removal()
        await update.message.reply_text(f"🛑 {arg} stopped")
    else:
        await update.message.reply_text(f"❌ {arg} is not being monitored")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    jobs = context.job_queue.jobs()
    if not jobs:
        await update.message.reply_text("😴 Nothing active")
        return
    msg = "📊 *Active:*\n"
    for job in jobs:
        try:
            minutes = int(job.trigger.interval.total_seconds()/60)
        except:
            minutes = "?"
        msg += f"  • {job.data['ticker']} — every {minutes} min.\n"
    await update.message.reply_text(msg)


async def cmd_now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    from main import build_trade_report
    if not context.args:
        await update.message.reply_text("❌ /now TICKER")
        return

    ticker = context.args[0].upper()
    print(f"\n📈 [/now] command received: {ticker}", flush=True)
    await update.message.reply_text(
        f"📈 Starting a full analysis of *{ticker}*...\n"
        f"That is 4-5 model calls and will take a couple of minutes — the report will arrive in parts.",
        parse_mode="Markdown"
    )

    try:
        # build_trade_report is synchronous and runs for minutes (LLM calls). Calling it
        # directly here blocked the event loop: while the analysis ran, the bot answered no
        # other command, and the Telegram connection had time to go stale.
        msg = await asyncio.to_thread(build_trade_report, ticker)
        for part in [msg[i:i + 3800] for i in range(0, len(msg), 3800)]:
            await update.message.reply_text(part, parse_mode="Markdown")
            await asyncio.sleep(0.3)
    except Exception as e:
        print(f"   ❌ [/now] {ticker}: {e}", flush=True)
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_news(update, context):
    print("\n📰 [/news] command received", flush=True)
    if not context.args:
        # Without a ticker — a whole-market overview. The synchronous build_market_overview
        # goes to a separate thread: it pulls quotes, three RSS sources and
        # makes an LLM call, i.e. it would hold the bot's event loop for dozens of seconds
        # (the same class of problem that was fixed earlier).
        await update.message.reply_text("📈 Building the market overview — indices, "
                                        "VIX, sectors and news background...")
        try:
            msg = await asyncio.to_thread(build_market_overview)
        except Exception as e:
            print(f"❌ [/news] market overview failed: {e}", flush=True)
            await update.message.reply_text(f"❌ Could not build the overview: {e}")
            return
        # The market overview consistently outgrows Telegram's 4096-character limit
        # (measured on a live run: 4346) — without splitting, the message simply does not
        # go through. We split at paragraph boundaries so as not to break the markup.
        for part in split_for_telegram(msg):
            try:
                await update.message.reply_text(part, parse_mode="Markdown")
            except Exception:
                # Same fallback as send_telegram_sync: with broken markup
                # it is better to send the text as is than to lose a finished report.
                await update.message.reply_text(part)
        return

    ticker = context.args[0].upper()
    try:
        # Load news
        all_news = fetch_news(ticker, limit=MAX_NEWS)
        if not all_news:
            await update.message.reply_text(f"❌ No news found for {ticker}")
            return

        # Analyse via the new agent
        analysis = analyze_news(ticker, all_news, use_cache=False)  # no cache needed for the command

        # Format the report (same function as before)
        msg = format_news_for_telegram(ticker, analysis, all_news)

        # Send (splitting can be added if it is long)
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")

async def cmd_lab(update, context):
    """
    /lab [days] — measures whether the /refilter thresholds give a real edge,
    on the sweep archive and real prices.

    The answer deliberately does NOT look like a recommendation: the main line is how many
    sweeps out of how many the set beat the control group, with a sign test.
    With few sweeps there will be p close to 1 and an explicit "too early
    for conclusions" — this is the correct behaviour, not a malfunction.
    """
    print("\n🔬 [/lab] command received", flush=True)
    try:
        days = int(context.args[0]) if context.args else 5
    except (ValueError, IndexError):
        await update.message.reply_text("❌ /lab [number_of_days], e.g. /lab 5")
        return

    sweeps = list_sweeps()
    if not sweeps:
        await update.message.reply_text(
            "❌ The sweep archive is empty. It fills automatically with every "
            "/autoscan — run it at least once.")
        return

    await update.message.reply_text(
        f"🔬 Computing on {len(sweeps)} sweep(s), horizon {days} d. "
        f"The first run downloads prices, this will take a minute or two.")
    try:
        text = await asyncio.to_thread(lab_report, days)
    except Exception as e:
        print(f"❌ [/lab] failed: {e}", flush=True)
        await update.message.reply_text(f"❌ Error: {e}")
        return

    # Monospace block: the report is tables aligned with spaces, in
    # plain Telegram text they would fall apart.
    for part in split_for_telegram(text, limit=3600):
        try:
            await update.message.reply_text(f"```\n{part}\n```",
                                            parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(part)


def _pop_save_flag(args):
    """
    Pulls the 'save' token out of ANY position in the arguments, before parsing positional ones.

    Exactly before: for /insiders the third positional argument is the insider's role, and
    '/insiders 3 50000 save' would otherwise silently look for purchases by a person with
    the title "save" and always return nothing.

    Returns (should_save, args_without_flag).
    """
    args = list(args or [])
    want = any(a.lower() == "save" for a in args)
    return want, [a for a in args if a.lower() != "save"]


_REFILTER_WARN_NO_FIELDS = ("/refilter will not work on this list, it has no "
                            "sweep fields.")


async def _save_as_queue_list(update, candidates, source, note="",
                              refilter_warn=_REFILTER_WARN_NO_FIELDS):
    """
    Puts the found tickers into the same list /queue reads from.

    DELIBERATELY THE SAME FILE as /autoscan and /refilter, not a separate one:
    save_autoscan_candidates overwrites it entirely and resets the cursor,
    i.e. the queue after /autoscan is lost. Chosen consciously — it only
    lives until the next /autoscan anyway, and a second list would need a second
    cursor and a source argument right in /queue. Hence also the explicit
    warning in the reply: silently overwriting someone else's queue is not acceptable, otherwise
    /queue afterwards returns quite different candidates from the ones expected.

    /queue only needs the 'ticker' key from a record (see next_autoscan_queue_batch),
    the other fields are stored for eyeballing in the file itself.

    /refilter does not and cannot work on such a list: it filters
    by screen_from_batch fields (volume_ratio, atr_pct, rr1, dollar_volume,
    pivot_gap_pct), which are not here — the selection has already been made by the source itself.
    This must not be kept silent, otherwise /refilter returns an empty list (thresholds
    are compared with .get(..., 0)) and it will look like "nothing was found".
    """
    from agents.screener import save_autoscan_candidates

    await asyncio.to_thread(save_autoscan_candidates, candidates, 0)
    msg = (f"💾 The /queue list was replaced: {len(candidates)} tickers from "
           f"/{source}.\n"
           f"⚠️ The previous queue (from /autoscan or /refilter) was overwritten, "
           f"the cursor was reset to the start.")
    if note:
        msg += "\n" + note
    msg += "\n\nNext: /queue — full analysis in batches of up to 10."
    if refilter_warn:
        msg += " " + refilter_warn
    await update.message.reply_text(msg)


async def cmd_earnings(update, context):
    """
    /earnings N        — all ~7000 stocks reporting in the next N days
    /earnings N M      — earnings in the window [N, M] days
    ... [save]         — save the results as the list for /queue

    A PURE earnings-calendar filter, WITHOUT technical scoring —
    2026-08-05, at the direct request: "a filter by days to earnings across all 7000
    stocks, to check by hand, no research needed". The list is returned as
    is, no ranking/recommendation — this is raw material for manual
    analysis, not a signal.

    Reads the ALREADY built cache (agents/earnings_calendar.py) — downloads nothing
    itself. The cache for ~7000 tickers must be built in advance:
        python -m agents.earnings_calendar fill-full
    (hours, one-off; after that the command answers instantly).
    """
    print("\n📅 [/earnings] command received", flush=True)
    want_save, args = _pop_save_flag(context.args)
    if not args:
        await update.message.reply_text(
            "❌ /earnings N — stocks reporting in the next N days\n"
            "   /earnings N M — earnings in the window [N, M] days\n"
            "   /earnings N [M] save — also save them as the list for /queue\n"
            "   E.g.: /earnings 7  or  /earnings 5 14")
        return
    try:
        if len(args) >= 2:
            min_days, max_days = int(args[0]), int(args[1])
        else:
            min_days, max_days = 0, int(args[0])
        if min_days > max_days:
            min_days, max_days = max_days, min_days
    except ValueError:
        await update.message.reply_text("❌ Days must be numbers, e.g. /earnings 5 14")
        return
    if max_days > 400:
        await update.message.reply_text("❌ Window too wide (max 400 days)")
        return

    from agents.earnings_calendar import scan_by_days

    matches, total_universe, covered = await asyncio.to_thread(scan_by_days, min_days, max_days)

    lines = [f"📅 *EARNINGS IN THE WINDOW [{min_days}, {max_days}] DAYS*"]
    coverage_pct = covered / total_universe * 100 if total_universe else 0
    lines.append(f"_Cache coverage: {covered}/{total_universe} stocks ({coverage_pct:.0f}%)"
                 + ("_" if coverage_pct >= 99 else " — part of the market is not checked, "
                    "run `python -m agents.earnings_calendar fill-full`_"))
    lines.append("")
    if not matches:
        lines.append("Nothing found in this window.")
    else:
        lines.append(f"Found: {len(matches)}")
        lines.append("")
        for ticker, days, edate in matches:
            date_str = edate or "?"
            lines.append(f"`{ticker:<8}` {days:>3} d.  ({date_str})")

    text = "\n".join(lines)
    for part in split_for_telegram(text):
        try:
            await update.message.reply_text(part, parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(part)

    if want_save and matches:
        # A warning, not a cut-off: earnings within the next EARNINGS_BLOCK_DAYS
        # days is a hard stop in check_entry_filters, i.e. such tickers
        # will reach /queue and are guaranteed to return NO_TRADE, spending
        # 4-5 model calls each. They cannot be filtered out on the user's behalf:
        # the user set the window and may have wanted exactly them (to look at the report by hand).
        from utils.signals import EARNINGS_BLOCK_DAYS

        blocked = sum(1 for _, d, _ in matches if d <= EARNINGS_BLOCK_DAYS)
        note = ""
        if blocked:
            note = (f"⚠️ Of these, {blocked} report within the next "
                    f"{EARNINGS_BLOCK_DAYS} d. — the entry filter will turn them into "
                    f"NO_TRADE, the analysis on them will be wasted. Narrow the window "
                    f"(/earnings {EARNINGS_BLOCK_DAYS + 1} {max_days} save), "
                    f"if that is not what you need.")
        await _save_as_queue_list(
            update,
            [{"ticker": t, "source": "earnings", "days_to_earnings": d,
              "earnings_date": edate} for t, d, edate in matches],
            "earnings", note)


async def cmd_insiders(update, context):
    """
    /insiders N [min_usd] [role] [save]
    Open-market insider purchases (Form 4, code=P, NOT 10b5-1) over
    the last N days across the WHOLE market — not a signal, a raw list for manual
    analysis (2026-08-05, at the direct request).

    AN HONEST WARNING ABOUT TIME: measured live — a window of 2-3
    calendar days gives 2000+ filed Form 4s across the market (not 7000
    tickers but the ACTUAL number of filings, which turned out larger than expected
    in advance) and takes several minutes on a cold cache. With a warm cache
    (the same date was already scanned) — seconds. So the default is narrow (2 days),
    and wider windows give an explicit warning.
    """
    print("\n💰 [/insiders] command received", flush=True)
    want_save, args = _pop_save_flag(context.args)
    try:
        days = int(args[0]) if args else 2
        min_usd = float(args[1]) if len(args) > 1 else 25_000
        role = args[2] if len(args) > 2 else None
    except ValueError:
        await update.message.reply_text(
            "❌ /insiders [days=2] [min.amount=25000] [role] [save]\n"
            "   E.g.: /insiders 3 50000 CEO\n"
            "   save — save the found tickers as the list for /queue")
        return
    days = max(1, min(30, days))

    from agents.insider_scan import scan_insider_buys, RAW_CACHE_DIR

    warn = ""
    if days > 3 and not os.path.isdir(RAW_CACHE_DIR):
        warn = ("\n⚠️ The first run on such a window may take several "
               "minutes (the market files 500-1000+ Form 4s a day). The disk cache "
               "will speed up repeated requests for overlapping dates.")
    status = await update.message.reply_text(
        f"💰 Looking for open-market insider purchases over {days} d. "
        f"(min. ${min_usd:,.0f}{', role: ' + role if role else ''})...{warn}")

    loop = asyncio.get_running_loop()

    def on_progress(n_filings, n_tickers):
        try:
            asyncio.run_coroutine_threadsafe(
                status.edit_text(f"💰 Filings scanned: {n_filings}, "
                                 f"tickers with purchases: {n_tickers}..."),
                loop)
        except Exception:
            pass

    try:
        result, n_filings, n_suspicious = await asyncio.to_thread(
            scan_insider_buys, days, min_usd, role, on_progress)
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")
        return

    from agents.insider_scan import _role_label

    lines = [f"💰 *INSIDER PURCHASES — last {days} d.*"]
    lines.append(f"_Filings scanned: {n_filings}"
                 + (f", excluded as price-suspicious: {n_suspicious}"
                    if n_suspicious else "") + "_")
    lines.append("")
    if not result:
        lines.append(f"Nothing found above ${min_usd:,.0f}"
                     + (f" for role \"{role}\"" if role else "") + ".")
    else:
        ranked = sorted(result.items(), key=lambda x: -x[1]["usd"])
        lines.append(f"Tickers found: {len(ranked)}")
        lines.append("")
        for ticker, v in ranked:
            lines.append(f"*{ticker}* — ${v['usd']:,.0f}")
            for tr in sorted(v["trades"], key=lambda t: -t["usd"])[:4]:
                role_str = _role_label(tr)
                lines.append(f"  • {tr['owner_name']} ({role_str}) — "
                             f"${tr['usd']:,.0f} @ ${tr['price']:.2f} "
                             f"({tr['date']})")
            if len(v["trades"]) > 4:
                lines.append(f"  ...{len(v['trades'])-4} more trades")

    text = "\n".join(lines)
    try:
        await status.delete()
    except Exception:
        pass
    for part in split_for_telegram(text):
        try:
            await update.message.reply_text(part, parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(part)

    if want_save and result:
        # The order is the same as in the report — by descending purchase amount: /queue
        # goes through the list top to bottom, and the ticker with the largest purchase
        # should be analysed first, not a random one.
        ranked = sorted(result.items(), key=lambda x: -x[1]["usd"])
        await _save_as_queue_list(
            update,
            [{"ticker": t, "source": "insiders", "insider_usd": v["usd"]}
             for t, v in ranked],
            "insiders")


async def cmd_scan(update, context):
    """Runs a scan of several tickers."""
    args = context.args
    if len(args) < 2:
        await update.message.reply_text(
            "❌ /scan TICKER1 TICKER2 TICKER3 ... (up to 10 tickers)\n"
            "Example: /scan AAPL TSLA NVDA MSFT GOOGL"
        )
        return
    
    tickers = [t.upper() for t in args[:10]]
    
    # Run the scan
    report = await scan_tickers(tickers, update, context)
    
    # Send the report in parts
    for part in [report[i:i+3800] for i in range(0, len(report), 3800)]:
        await update.message.reply_text(part, parse_mode="Markdown")
        await asyncio.sleep(0.5)

async def cmd_screen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Preliminary ticker screen from Yahoo — no LLM calls, so it is fast.
    /screen [trending|actives|gainers|losers|small_gainers] [count]
    """
    print("\n🔎 [/screen] command received, starting the screen...", flush=True)
    from agents.screener import run_screen, format_screen_report, SOURCES

    args = context.args or []
    source = args[0].lower() if args and args[0].lower() in SOURCES else "trending"
    try:
        limit = int(args[1]) if len(args) > 1 else 15
    except ValueError:
        limit = 15
    limit = max(3, min(30, limit))

    status = await update.message.reply_text(
        f"🔎 Screening tickers ({SOURCES[source]}), checking up to {limit}...\n"
        f"This is the computational part without models — it will take about a minute."
    )

    def on_progress(text):
        try:
            asyncio.run_coroutine_threadsafe(
                status.edit_text(f"🔎 {SOURCES[source]}: {text}"), loop)
        except Exception:
            pass

    loop = asyncio.get_running_loop()
    # The scan is synchronous and not fast — move it to a thread, otherwise the bot freezes
    results, _ = await asyncio.to_thread(run_screen, source, limit, on_progress)

    report = format_screen_report(results, source)
    try:
        await status.delete()
    except Exception:
        pass
    await update.message.reply_text(report, parse_mode="Markdown")


async def cmd_massscreen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    A mass pass over stocks from all Yahoo screeners, in batches.
    /mass [count] [batch_size]   e.g.: /mass 100 10
    """
    print("\n🔎 [/mass] command received", flush=True)
    from agents.screener import mass_screen, format_mass_report, format_batch_line

    args = context.args or []
    try:
        limit = int(args[0]) if args else 100
    except ValueError:
        limit = 100
    try:
        batch_size = int(args[1]) if len(args) > 1 else 10
    except ValueError:
        batch_size = 10
    limit = max(10, min(200, limit))
    batch_size = max(5, min(25, batch_size))

    await update.message.reply_text(
        f"🔎 Going through up to *{limit}* stocks in batches of *{batch_size}*.\n"
        f"Computing only the computational part (no models) — it is fast, "
        f"I will send results as they are ready.",
        parse_mode="Markdown"
    )

    loop = asyncio.get_running_loop()
    chat = update.effective_chat.id

    def on_batch(idx, total, batch_res, market_pos=None, market_total=None):
        """Send a short summary for each batch without waiting for the pass to finish."""
        text = format_batch_line(idx, total, batch_res, market_pos, market_total)
        if text:
            asyncio.run_coroutine_threadsafe(
                context.bot.send_message(chat, text, parse_mode="Markdown"), loop)

    def on_status(text):
        print(f"   [/mass] {text}", flush=True)

    results, market_range = await asyncio.to_thread(mass_screen, limit, batch_size, on_status, on_batch)

    report = format_mass_report(results, market_range)
    for part in [report[i:i + 3800] for i in range(0, len(report), 3800)]:
        await update.message.reply_text(part, parse_mode="Markdown")
        await asyncio.sleep(0.3)


async def cmd_autoscan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /autoscan — a one-off pass over the WHOLE market (~7000 NASDAQ+NYSE/AMEX stocks,
    see agents/screener.py::full_market_sweep), without the cursor and pagination of
    /mass — the whole list in one call, in batches, so that Yahoo does not cut off the request
    entirely.

    It ONLY computes and saves — the raw sweep results (for /refilter) and
    the filtered/ranked list of LONG candidates (for /queue).
    It scans NOTHING with the expensive pipeline itself — it used to immediately run
    a full analysis of the top 10, but that took the choice away from the user: first
    one needs to look at fail_counts and possibly tune thresholds via
    /refilter, and only then consciously decide to run /queue. Three
    separate steps, each with its own report, instead of one that decides
    what to do next by itself.

    A long command (~40-50 minutes for the pass itself), so we send progress along
    the way instead of staying silent until the end.
    """
    print("\n🌐 [/autoscan] command received", flush=True)
    from agents.screener import (full_market_sweep, rank_long_candidates, save_autoscan_candidates,
                                 save_raw_sweep_results, format_candidates_table)

    await update.message.reply_text(
        "🌐 Starting a one-off pass over the WHOLE market (~7000 stocks).\n"
        "It will take about 40-50 minutes — I will send progress along the way.\n"
        "When done I will send a breakdown by filter and stop — then you "
        "decide: /refilter to tune thresholds, or /queue to start the analysis."
    )

    loop = asyncio.get_running_loop()
    chat = update.effective_chat.id

    def on_progress(checked, total, long_count):
        asyncio.run_coroutine_threadsafe(
            context.bot.send_message(
                chat,
                f"🔎 Checked {checked}/{total} market stocks — LONG candidates so far {long_count}"
            ),
            loop
        )

    def on_status(text):
        print(f"   [/autoscan] {text}", flush=True)

    results = await asyncio.to_thread(full_market_sweep, 25, 20, on_status, on_progress)
    # Raw sweep data — so the filter can be recomputed later via
    # /refilter without another 40-50-minute pass over the market.
    await asyncio.to_thread(save_raw_sweep_results, results)

    all_long, fail_counts = rank_long_candidates(results)
    passed_entry = sum(1 for r in results if r["direction"] == "LONG" and r["entry_ok"])

    lines = [f"🏁 Market pass complete: {passed_entry} passed the entry filter → "
             f"{len(all_long)} after extra filters."]
    if fail_counts:
        # Screen-out statistics for the intermediate filter (see agents/screener.py::
        # apply_prefilter) — so you can see whether one criterion cuts
        # disproportionately many (then it should be softened via /refilter).
        breakdown = ", ".join(f"{k}: {v}" for k, v in sorted(fail_counts.items(), key=lambda x: -x[1]))
        lines.append(f"Screen-out reasons: {breakdown}")

    if not all_long:
        lines.append(
            "\n😐 No LONG candidates left after the filters — "
            "try /refilter with softer thresholds."
        )
        await update.message.reply_text("\n".join(lines))
        return

    await asyncio.to_thread(save_autoscan_candidates, all_long, 0)
    lines.append(f"\n🎯 *Top 10 by rating:*\n{format_candidates_table(all_long, 10)}")
    lines.append(
        "\nNothing is scanned automatically. Next: /refilter — tune "
        "thresholds/sorting without a new pass, or /queue — start the full analysis."
    )
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_refilter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /refilter [param=value ...] [sort=field] — recomputes the intermediate
    filter/ranking (agents/screener.py::rank_long_candidates →
    apply_prefilter) on the data of the LAST /autoscan, without another
    40-50-minute pass over the market — takes the raw results from
    reports/autoscan_raw_results.json (save_raw_sweep_results). Needed to
    test threshold edits in seconds instead of running the whole market again
    every time (see the abs(score) case that initially cut 96%
    of candidates).

    Parameters (all are temporary overrides FOR THIS CALL ONLY, defaults in
    the code do not change, see agents/screener.py::FILTER_ARG_MAP/SORT_FIELDS):
    volume=, rr=, price=, atr_min=, atr_max=, dv=, gap_pct=, sort=(gap|
    volume_ratio|dollar_volume|rr1|atr_pct). Example:
    "/refilter volume=1.5 rr=2 sort=dollar_volume".

    Like /autoscan — scans nothing automatically, only
    recomputes and saves the list for /queue (cursor at position 0).
    """
    print("\n🔁 [/refilter] command received", flush=True)
    from agents.screener import (load_raw_sweep_results, rank_long_candidates,
                                 save_autoscan_candidates, parse_refilter_args,
                                 format_candidates_table, annotate_earnings,
                                 apply_earnings_filter)

    overrides, sort_key, unknown, earnings = parse_refilter_args(context.args or [])
    if unknown:
        await update.message.reply_text(
            "❌ Unrecognised parameters: " + ", ".join(unknown) + "\n"
            "Available: volume=, rr=, price=, atr_min=, atr_max=, dv=, gap_pct=, "
            "earnings=N (only those reporting in the next N days), "
            "no_earnings=N (the opposite, hide them), "
            "sort=(gap|volume_ratio|dollar_volume|rr1|atr_pct)"
        )
        return

    results, generated_at = await asyncio.to_thread(load_raw_sweep_results)
    if not results:
        await update.message.reply_text(
            "❌ No saved sweep data — run /autoscan at least once first."
        )
        return

    all_long, fail_counts = await asyncio.to_thread(
        rank_long_candidates, results, overrides, sort_key
    )
    passed_entry = sum(1 for r in results if r["direction"] == "LONG" and r["entry_ok"])

    lines = [f"🔁 Filter recomputed on sweep data from {generated_at}:"]
    if overrides or sort_key != "gap":
        applied = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "—"
        lines.append(f"Overrides: {applied} | sort: {sort_key}")
    lines.append(f"{passed_entry} passed the entry filter → {len(all_long)} after extra filters.")
    if fail_counts:
        breakdown = ", ".join(f"{k}: {v}" for k, v in sorted(fail_counts.items(), key=lambda x: -x[1]))
        lines.append(f"Screen-out reasons: {breakdown}")
    await update.message.reply_text("\n".join(lines))

    if not all_long:
        return

    # Earnings-date annotation — only if requested: it costs
    # network requests for missing names, and without parameters /refilter
    # must stay instant, as before.
    if earnings:
        await update.message.reply_text(
            "📅 Annotating the top of the list with earnings dates "
            "(fetching the missing ones — this may take a minute)..."
        )
        all_long, fetched = await asyncio.to_thread(annotate_earnings, all_long)
        before = len(all_long)
        all_long = apply_earnings_filter(
            all_long, keep_within=earnings.get("keep"),
            exclude_within=earnings.get("exclude")
        )
        note = []
        if "keep" in earnings:
            note.append(f"kept only those reporting in ≤{earnings['keep']} d.")
        if "exclude" in earnings:
            note.append(f"hidden those reporting in ≤{earnings['exclude']} d.")
        await update.message.reply_text(
            f"📅 {', '.join(note)}: {before} → {len(all_long)}"
            + (f" (dates fetched: {fetched})" if fetched else "")
        )
        if not all_long:
            await update.message.reply_text(
                "😐 No candidates left after the earnings filter. "
                "Note: only the top of the list is annotated, the rest have an unknown "
                "date and do not pass in earnings= mode."
            )
            return

    await asyncio.to_thread(save_autoscan_candidates, all_long, 0)
    await update.message.reply_text(
        f"💾 The /queue list was updated ({len(all_long)} candidates).\n"
        f"🎯 *Top 10 by rating:*\n{format_candidates_table(all_long, 10)}\n\n"
        f"Nothing is scanned automatically — run /queue when you are ready.",
        parse_mode="Markdown"
    )


async def cmd_find(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /find [N=10] — collect N candidates by automatically stepping down the relaxation
    ladder (agents/screener.py::FIND_LADDER) on the data of the LAST
    /autoscan. Does not touch the network: all fields are computed by the sweep, and apply_prefilter
    is in-memory arithmetic, so the whole descent takes seconds. Not a single
    model call: the decision is fully deterministic.

    SWEEP FRESHNESS IS NOT CHECKED BUT SHOWN. The command deliberately does not
    start a new market pass: it costs 40-50 minutes, and silently going into
    an hour of work on a command that looks instant is not acceptable. The data age
    is printed in the reply — the decision "time to refresh" is made by a human.

    THE KEY THING IN THE REPLY IS NOT THE LIST BUT THE STEP NUMBER. A command obliged to return
    N names will return N even on a day when the market has nothing to offer: a search that
    always ends in success carries no information by itself. What distinguishes
    one day from another is only how deep it had to go,
    so the step-zero counter is ALWAYS printed, and the number of names added
    by relaxation gets its own line. This also turns the command's built-in
    flaw into a free measure of market stinginess — it accumulates in FIND_LOG_FILE.
    """
    print("\n🎯 [/find] command received", flush=True)
    from agents.screener import (append_find_log, find_candidates,
                                 format_find_report, load_raw_sweep_results)

    try:
        want = int(context.args[0]) if context.args else 10
    except ValueError:
        await update.message.reply_text("❌ /find [N=10] — e.g. /find 15")
        return
    want = max(1, min(50, want))

    results, generated_at = await asyncio.to_thread(load_raw_sweep_results)
    if not results:
        await update.message.reply_text(
            "❌ No saved sweep data — run /autoscan at least once first.")
        return

    candidates, rung, counts, _ = await asyncio.to_thread(
        find_candidates, results, want)
    text, shown = format_find_report(want, candidates, rung, counts, generated_at)

    from datetime import datetime

    await asyncio.to_thread(append_find_log, {
        "at": datetime.now().isoformat(), "want": want, "rung": rung,
        "counts": counts, "found": len(shown),
        "enough": len(candidates) >= want, "source": "find",
        "sweep_generated_at": generated_at,
    })

    for part in split_for_telegram(text):
        try:
            await update.message.reply_text(part, parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(part)

    if not shown:
        return

    await _save_as_queue_list(
        update, shown, "find",
        refilter_warn=("⚠️ Don't run /refilter before /queue — it will recompute "
                       "the list on default thresholds and undo the relaxation."))


async def _deliver_morning(bot, chat, text, shown, stale):
    """
    Sends the morning build result. One function for the scheduled task
    and for the manual /morning now — otherwise the two paths diverge, and checking with a manual
    run what will arrive in the morning becomes pointless.
    """
    if text is None:
        await bot.send_message(
            chat, "❌ Morning build: the market pass failed, and there is no saved "
                  "data from a previous sweep either. The list was not built.")
        return

    head = "🌅 *Morning list*\n"
    if stale:
        # Loudly, not as a footnote: a list on yesterday's prices looks exactly
        # like a fresh one, and they cannot be told apart by content.
        head += ("⚠️ Today's market pass FAILED — the list was built from "
                 "the previous sweep's data. Prices and levels are not today's.\n")
    for part in split_for_telegram(head + "\n" + text):
        try:
            await bot.send_message(chat, part, parse_mode="Markdown")
        except Exception:
            await bot.send_message(chat, part)

    if shown:
        await bot.send_message(
            chat,
            f"💾 The /queue list was replaced ({len(shown)} tickers).\n"
            f"⚠️ Don't run /refilter before /queue — it will recompute the list on "
            f"default thresholds and undo the relaxation.")


async def morning_job(context):
    """
    The daily task of the morning loop.

    Registered for EVERY day, not for weekdays via the days parameter:
    weekday numbering in run_daily changed between versions of
    python-telegram-bot (0=Monday vs. 0=Sunday), and a mistake there
    would show up as silence on Mondays that nobody would notice for a week.
    The trading-day filter is inside, and it also accounts for exchange holidays.

    Progress is deliberately NOT sent to the chat: the task runs at 7:45 while the person
    is asleep, and forty minutes of "checked 2000/6950" is spam, not information.
    Everything goes to the log.
    """
    from agents.morning_scan import load_config, run_morning_scan
    from utils.market_hours import is_trading_day

    cfg = load_config()
    if not cfg.get("enabled"):
        return
    if not is_trading_day():
        print("🌅 [morning] not a trading day — skipping", flush=True)
        return

    from config import MY_CHAT_ID
    chat = cfg.get("chat_id") or MY_CHAT_ID
    if not chat:
        print("🌅 [morning] nobody to send to: chat_id not set", flush=True)
        return

    want = cfg.get("want", 10)
    print(f"\n🌅 [morning] start, collecting {want} candidates", flush=True)

    def on_status(text):
        print(f"   [morning] {text}", flush=True)

    from config import TELEGRAM_SCHEDULED_QUIET
    try:
        text, shown, stale = await asyncio.to_thread(
            run_morning_scan, want, on_status, None)
    except Exception as exc:
        print(f"🌅 [morning] failed: {exc}", flush=True)
        if not TELEGRAM_SCHEDULED_QUIET:
            await context.bot.send_message(chat, f"❌ Morning build failed: {exc}")
        return

    if TELEGRAM_SCHEDULED_QUIET:
        # The list is already saved for /queue inside run_morning_scan; nothing is sent to the chat
        # (the 18:00 daily digest will say whether there was a sweep).
        print(f"🌅 [morning] done: {len(shown or [])} candidates"
              + (" (from the previous sweep's data!)" if stale else "")
              + "\n" + (text or "list not built"), flush=True)
        return
    await _deliver_morning(context.bot, chat, text, shown, stale)


async def intraday_resolve_job(context):
    """
    Task at 15:35 ET: closes all of today's signals by the daily rule.

    WHY 15:35 AND NOT 15:30. The bar closing at 15:30 exists only
    after 15:30, and yfinance returns it with a delay. Five minutes of margin is cheaper
    than systematically losing the last bar of every trade.

    WHY ON A SCHEDULE AND NOT LAZILY ON /stats. yfinance keeps 5-minute bars for
    about 60 days — an outcome not computed today cannot be recovered in a quarter.
    The old lazy scheme was fine for hourly data, but not
    for this.
    """
    from utils.market_hours import is_trading_day
    from utils.signal_history import resolve_intraday_today, resolve_no_trade_backlog

    if not is_trading_day():
        return

    from config import MY_CHAT_ID
    from agents.morning_scan import load_config
    chat = load_config().get("chat_id") or MY_CHAT_ID

    print("\n🔔 [15:35] resolving today's signals by the daily rule", flush=True)
    try:
        summary = await asyncio.to_thread(resolve_intraday_today)
    except Exception as exc:
        print(f"🔔 [15:35] resolution failed: {exc}", flush=True)
        summary = {"resolved": [], "excluded": []}

    # NO_TRADE counterfactual — silently to the log, no chat message: this is background
    # maintenance of the control group, not an event worth pinging about
    # every day. The limit is small so as not to stretch the task itself to
    # dozens of minutes — a backlog of hundreds of old records is caught up over
    # several days in a row; a one-off catch-up of the history is done by hand
    # (resolve_no_trade_backlog(limit=None)).
    try:
        cf_counts = await asyncio.to_thread(resolve_no_trade_backlog, None, 150)
        if cf_counts["processed"]:
            print(f"   [NO_TRADE counterfactual] processed {cf_counts['processed']} "
                 f"(resolved={cf_counts['resolved']}, excluded={cf_counts['excluded']})",
                 flush=True)
    except Exception as exc:
        print(f"   [NO_TRADE counterfactual] failed: {exc}", flush=True)

    if not summary["resolved"] and not summary["excluded"]:
        print("   there were no unresolved signals today", flush=True)
        return

    lines = ["🔔 *Today's signal results*"]
    labels = {"target_1": "✅ target 1", "target_2": "✅ target 2",
              "stop": "❌ stop", "close": "⏹ closed at 15:30"}
    for item in summary["resolved"]:
        ret = item["return_pct"]
        lines.append(f"{labels.get(item['outcome'], item['outcome'])} "
                     f"*{item['ticker']}* "
                     + (f"{ret:+.2f}%" if ret is not None else "")
                     + f" — {item['day']}")
    for item in summary["excluded"]:
        # Show these too: a silently vanished signal cannot be told apart from
        # a signal that never existed.
        lines.append(f"⚪ *{item['ticker']}* outside the statistics — {item['day']}")

    from config import TELEGRAM_SCHEDULED_QUIET
    if not chat or TELEGRAM_SCHEDULED_QUIET:      # results will come in the 18:00 daily digest
        print("\n".join(lines), flush=True)
        return
    for part in split_for_telegram("\n".join(lines)):
        try:
            await context.bot.send_message(chat, part, parse_mode="Markdown")
        except Exception:
            await context.bot.send_message(chat, part)


def build_daily_digest(day=None):
    """
    The daily digest is the bot's only scheduled message to the chat (the
    owner's decision of 2026-10-09, see TELEGRAM_SCHEDULED_QUIET in config.py).

    Besides signal results it also carries a liveness check: for 9 days, 2026-09-30 to
    10-09, the hourly autopilot silently skipped every tick because of a stuck flag,
    and it became noticeable only when analysing the data. A line "not a single
    check today" would have shown it on the very first evening.
    """
    import glob
    import json as _json
    from datetime import datetime as _dt
    from zoneinfo import ZoneInfo
    from config import DEEPSEEK_VERDICTS_FILE
    from utils.signal_history import (RULE_INTRADAY, _trade_return_pct,
                                      load_no_trade_history, load_signal_history)

    day = day or _dt.now(ZoneInfo("America/New_York")).date()
    ds = day.isoformat()
    sig = [(t, r) for t, lst in load_signal_history().items() for r in lst]
    nt = [(t, r) for t, lst in load_no_trade_history().items() for r in lst]
    sig_today = sorted(((t, r) for t, r in sig if (r.get("timestamp") or "")[:10] == ds),
                       key=lambda x: x[1].get("timestamp", ""))
    nt_today = [(t, r) for t, r in nt if (r.get("timestamp") or "")[:10] == ds]

    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sweep = glob.glob(os.path.join(app_dir, "reports", "sweeps", f"sweep_{ds}_*.json"))
    verdicts = 0
    try:
        with open(DEEPSEEK_VERDICTS_FILE, encoding="utf-8") as f:
            verdicts = sum(1 for v in _json.load(f).values()
                           if (v.get("signal_timestamp") or v.get("asked_at") or "")[:10] == ds)
    except Exception:
        pass

    lines = [f"📊 *day-trader bot — results for {day:%d.%m}*",
             f"Morning sweep: {'yes' if sweep else '⚠️ none'}",
             f"Stocks checked: {len(sig_today) + len(nt_today)} "
             f"(signals {len(sig_today)}, NO_TRADE {len(nt_today)}), "
             f"DeepSeek verdicts: {verdicts}"]
    if not sig_today and not nt_today:
        lines.append("⚠️ Not a single check during the trading day — the autopilot seems "
                     "to be stuck. Check the container logs.")

    labels = {"target_1": "target 1", "target_2": "target 2", "stop": "stop",
              "close": "closed at 15:30"}
    rets = []
    for t, r in sig_today:
        ret = _trade_return_pct(r)
        if ret is not None:
            rets.append(ret)
        outcome = labels.get(r.get("outcome"), r.get("outcome") or "unresolved")
        lines.append(f"• {r.get('direction')} *{t}* at {(r.get('timestamp') or '')[11:16]}, "
                     f"entry {r.get('entry_price')} → {outcome}"
                     + (f" {ret:+.2f}%" if ret is not None else ""))
    if rets:
        lines.append(f"Today: {len(rets)} trades, average {sum(rets) / len(rets):+.2f}%, "
                     f"positive {sum(x > 0 for x in rets)} of {len(rets)}")

    allr = [_trade_return_pct(r) for _t, r in sig if r.get("resolution_rule") == RULE_INTRADAY]
    allr = [x for x in allr if x is not None]
    if allr:
        lines.append(f"Since tracking began: {len(allr)} trades, average "
                     f"{sum(allr) / len(allr):+.2f}%, positive {sum(x > 0 for x in allr) / len(allr):.0%}")
    return "\n".join(lines)


async def daily_digest_job(context):
    """Daily digest at DAILY_DIGEST_TIME (ET), trading days only."""
    from utils.market_hours import is_trading_day
    from config import MY_CHAT_ID
    if not is_trading_day() or not MY_CHAT_ID:
        return
    try:
        text = await asyncio.to_thread(build_daily_digest)
    except Exception as exc:
        text = f"⚠️ day-trader bot: could not build the daily digest: {exc}"
    print(f"\n📊 [daily digest]\n{text}", flush=True)
    for part in split_for_telegram(text):
        try:
            await context.bot.send_message(MY_CHAT_ID, part, parse_mode="Markdown")
        except Exception:
            await context.bot.send_message(MY_CHAT_ID, part)


async def cmd_morning(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /morning            — status
    /morning on | off   — enable/disable the pre-open auto-build
    /morning n 15       — how many candidates to collect
    /morning now        — run right now (40-50 minutes)

    State lives in a separate file (agents/morning_scan.py::MORNING_CONFIG_FILE),
    so it survives a container rebuild — no need to re-enable it after every
    deploy.
    """
    print("\n🌅 [/morning] command received", flush=True)
    from agents.morning_scan import load_config, run_morning_scan, save_config
    from agents.screener import load_find_log
    from utils.market_hours import is_trading_day

    args = [a.lower() for a in (context.args or [])]
    cfg = load_config()

    if args[:1] == ["on"]:
        cfg["enabled"] = True
        # Remember the chat right when enabling: MY_CHAT_ID is set by an environment
        # variable and may turn out empty, leaving nowhere to send.
        cfg["chat_id"] = update.effective_chat.id
        save_config(cfg)
        await update.message.reply_text(
            f"✅ Morning auto-build enabled: every trading day at "
            f"{cfg['hour']:02d}:{cfg['minute']:02d} ET, {cfg['want']} candidates.\n"
            f"The market pass takes 40-50 minutes — the message will arrive before the open "
            f"at 9:30 ET. Weekends and exchange holidays are skipped.")
        return

    if args[:1] == ["off"]:
        cfg["enabled"] = False
        save_config(cfg)
        await update.message.reply_text("🛑 Morning auto-build disabled.")
        return

    if args[:1] == ["n"]:
        try:
            cfg["want"] = max(1, min(50, int(args[1])))
        except (IndexError, ValueError):
            await update.message.reply_text("❌ /morning n 15 — how many to collect")
            return
        save_config(cfg)
        await update.message.reply_text(
            f"✅ Will collect {cfg['want']} candidates.")
        return

    if args[:1] == ["now"]:
        await update.message.reply_text(
            f"🌅 Starting the build right now: market pass 40-50 minutes, "
            f"will send {cfg['want']} candidates when ready.")

        def on_status(text):
            print(f"   [morning/now] {text}", flush=True)

        try:
            text, shown, stale = await asyncio.to_thread(
                run_morning_scan, cfg["want"], on_status, None)
        except Exception as exc:
            await update.message.reply_text(f"❌ Build failed: {exc}")
            return
        await _deliver_morning(context.bot, update.effective_chat.id,
                               text, shown, stale)
        return

    if args:
        await update.message.reply_text(
            "❌ /morning [on|off|now|n NUMBER] — or no arguments for status")
        return

    last = next((e for e in reversed(load_find_log())
                 if e.get("source") == "morning"), None)
    lines = [
        f"🌅 *Morning auto-build:* {'enabled' if cfg['enabled'] else 'disabled'}",
        f"Time: {cfg['hour']:02d}:{cfg['minute']:02d} ET, trading days",
        f"Candidates per run: {cfg['want']}",
        f"Today: {'trading day' if is_trading_day() else 'exchange closed'}",
    ]
    if last:
        lines.append(
            f"Last run: {last['at'][:16].replace('T', ' ')} — "
            f"{last['found']} at step {last['rung']}"
            + (" (from yesterday's data)" if last.get("stale") else ""))
    else:
        lines.append("Last run: none yet")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


AUTOPILOT_MAX_ATTEMPTS_MULT = 5   # attempt cap = min(N * this, ABS cap)
AUTOPILOT_MAX_ATTEMPTS_ABS = 30
AUTOPILOT_TIME_CAP_MIN = 60
# The _autopilot_running flag lives in state.json and is cleared only at the end of a hunt.
# A hunt interrupted by an exception or a container restart left it set
# forever: from 2026-09-30 to 2026-10-09 the hourly autopilot skipped EVERY tick
# ("a hunt is already running"), and the bot did no daily checks for 7 trading days. Now
# the flag has a start time, and a flag older than the longest hunt limit (+margin)
# is considered stuck. Plus a reset at process start (main.py, register_menu).
AUTOPILOT_STALE_MIN = AUTOPILOT_TIME_CAP_MIN + 10


def autopilot_busy(state):
    """True only if a hunt is really running. Clears a stuck flag."""
    from datetime import datetime as _dt, timedelta as _td
    from utils.helpers import save_state
    if not state.get("_autopilot_running"):
        return False
    started = state.get("_autopilot_started_at")
    try:
        if _dt.now() - _dt.fromisoformat(started) < _td(minutes=AUTOPILOT_STALE_MIN):
            return True
    except (TypeError, ValueError):
        pass                      # a flag without a start time — left over from old code
    print(f"⚠️ [autopilot] the hunt flag is stuck (set {started or 'at an unknown time'}), clearing it",
          flush=True)
    state["_autopilot_running"] = False
    save_state(state)
    return False


async def cmd_autopilot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /autopilot N [force] — looks for N REAL LONG signals (not candidates),
    running the expensive full analysis over the LONG+entry_ok pool (without the
    apply_prefilter limits — see agents/screener.py::FULL_POOL_OVERRIDES), until
    N are confirmed by the full analysis, the pool runs out or
    a safety stop fires.
    /autopilot stop — stop the current hunt after the current attempt.

    HOW IT DIFFERS FROM /find. /find is a cheap measuring instrument: seconds,
    no network or models, the step number as an honest measure of market stinginess.
    /autopilot is an expensive executor: minutes to hours, an LLM on every attempt, and
    it does NOT search for a combination of thresholds — combinations are not needed, because any
    /refilter gives only a subset of the same LONG+entry_ok pool.
    It is simpler to take the whole pool once, sorted by gap, and
    leave the "signal or not" decision to the expensive analysis rather than to threshold
    tuning — that tuning has already been tested and loses to chance (regime
    switching on 37 sweeps lost to a coin).

    SAFETY STOPS — THREE INDEPENDENT ONES, not one instead of another:
    - the goal is reached (success);
    - the candidate pool is exhausted (an honest "the market gave fewer than N today");
    - an attempt cap (min(5N, 30)) OR a time cap (60 min) — whichever
      comes first. The attempt cap does not protect against a rate limit or
      a model reload stretching each attempt; the time cap
      does not protect against "big pool, stingy market". Both are needed.
    Plus a manual stop — /autopilot stop sets a flag in state.json,
    checked before EACH next attempt (an attempt already started is not interrupted).

    FORCE. By default tickers already analysed TODAY (a record exists in
    signal_history or no_trade_history — see tickers_seen_today)
    are skipped: an expensive call to confirm an already known outcome of the same
    day is a waste of time without a new observation. force disables this
    check. The very point of skipping is tied to real trading hours (in
    a closed market/on synthetic runs "today" does not guarantee the
    same day for yfinance data) — this will really show
    only on live morning runs; force is a workaround for the time being,
    until this is verified live.

    NO_TRADE is not sent to the chat for each attempt (that would be noise — the request was for
    signals, not the process) — it is only counted. These runs go into no_trade_history
    as usual, automatically, inside the pipeline itself.
    """
    print("\n🤖 [/autopilot] command received", flush=True)
    from datetime import datetime as _dt, timedelta as _td

    from agents.scanner import format_single_result
    from agents.screener import (FULL_POOL_OVERRIDES, append_find_log,
                                 load_raw_sweep_results, rank_long_candidates)
    from utils.helpers import age_str, load_state, save_state
    from utils.signal_history import tickers_seen_today

    args = [a.lower() for a in (context.args or [])]

    if args[:1] == ["stop"]:
        state = load_state()
        state["_autopilot_stop"] = True
        save_state(state)
        running = state.get("_autopilot_running", False)
        await update.message.reply_text(
            "🛑 Stopping after the current attempt..." if running else
            "🛑 Stop flag set. No hunt is running now — it will take effect "
            "if one is started before the flag is cleared."
        )
        return

    force = "force" in args
    num_args = [a for a in args if a != "force"]
    try:
        n = int(num_args[0]) if num_args else 5
    except ValueError:
        await update.message.reply_text(
            "❌ /autopilot N [force] — e.g. /autopilot 5\n"
            "   /autopilot stop — stop the current hunt")
        return
    n = max(1, min(20, n))

    results, generated_at = await asyncio.to_thread(load_raw_sweep_results)
    if not results:
        await update.message.reply_text(
            "❌ No saved sweep data — run /autoscan at least once first.")
        return

    pool, _ = await asyncio.to_thread(
        rank_long_candidates, results, FULL_POOL_OVERRIDES, "gap")
    if not pool:
        await update.message.reply_text(
            "❌ The candidate pool is empty even without limits — the market "
            "literally has nothing to offer today.")
        return

    skipped_seen = 0
    if not force:
        seen = await asyncio.to_thread(tickers_seen_today)
        before = len(pool)
        pool = [c for c in pool if c["ticker"] not in seen]
        skipped_seen = before - len(pool)

    max_attempts = min(n * AUTOPILOT_MAX_ATTEMPTS_MULT, AUTOPILOT_MAX_ATTEMPTS_ABS)
    time_cap = _td(minutes=AUTOPILOT_TIME_CAP_MIN)
    started = _dt.now()

    state = load_state()
    state["_autopilot_stop"] = False
    state["_autopilot_running"] = True
    state["_autopilot_started_at"] = started.isoformat()
    save_state(state)

    status_msg = await update.message.reply_text(
        f"🎯 Looking for {n} real LONG signals among {len(pool)} candidates "
        f"(sweep from {generated_at[:16].replace('T', ' ') if generated_at else '?'}, "
        f"{age_str(generated_at) if generated_at else 'age unknown'}).\n"
        f"Limit: {max_attempts} attempts / {AUTOPILOT_TIME_CAP_MIN} min.\n"
        + (f"Skipped as already analysed today: {skipped_seen}.\n" if skipped_seen else "")
        + "/autopilot stop — stop."
    )

    found_tickers = []
    no_trade_count = 0
    attempts = 0
    stop_reason = None

    for candidate in pool:
        if len(found_tickers) >= n:
            break
        if attempts >= max_attempts:
            stop_reason = "attempts"
            break
        if _dt.now() - started >= time_cap:
            stop_reason = "time"
            break
        state = load_state()
        if state.get("_autopilot_stop"):
            stop_reason = "manual"
            break

        ticker = candidate["ticker"]
        attempts += 1
        elapsed_min = int((_dt.now() - started).total_seconds() // 60)
        try:
            await status_msg.edit_text(
                f"🎯 Attempt {attempts}/{max_attempts} | "
                f"{elapsed_min}/{AUTOPILOT_TIME_CAP_MIN} min\n"
                f"Now analysing: {ticker}\n"
                f"Found: {len(found_tickers)}/{n}"
                + (f" — {', '.join(found_tickers)}" if found_tickers else "") + "\n"
                f"NO_TRADE along the way: {no_trade_count}\n"
                f"/autopilot stop — stop"
            )
        except Exception:
            pass

        try:
            result = await asyncio.to_thread(scan_single_ticker, ticker, None, attempts, len(pool))
        except Exception as e:
            print(f"   ⚠️ [/autopilot] {ticker}: {e}", flush=True)
            continue

        direction = result.get("direction", "NO_TRADE")
        if direction in ("LONG", "STRONG_LONG"):
            found_tickers.append(ticker)
            text = f"✅ *Signal {len(found_tickers)}/{n}:*\n" + format_single_result(result)
            for part in split_for_telegram(text):
                try:
                    await update.message.reply_text(part, parse_mode="Markdown")
                except Exception:
                    await update.message.reply_text(part)
        else:
            no_trade_count += 1

        await asyncio.sleep(2)

    state = load_state()
    state["_autopilot_running"] = False
    save_state(state)

    if stop_reason is None:
        stop_reason = "success" if len(found_tickers) >= n else "exhausted"

    labels = {
        "success": "✅ Done",
        "exhausted": "😐 The candidate pool ran out before the goal",
        "attempts": "⏱ Hit the attempt limit",
        "time": "⏱ Hit the time limit",
        "manual": "🛑 Stopped manually",
    }
    elapsed_min = int((_dt.now() - started).total_seconds() // 60)
    await update.message.reply_text(
        f"{labels[stop_reason]}: found {len(found_tickers)}/{n}\n"
        f"Attempts: {attempts}/{max_attempts} | NO_TRADE: {no_trade_count} | "
        f"time: {elapsed_min} min"
        + (f"\nSignals: {', '.join(found_tickers)}" if found_tickers else "")
    )

    await asyncio.to_thread(append_find_log, {
        "at": _dt.now().isoformat(), "source": "autopilot",
        "want": n, "found": len(found_tickers), "tickers": found_tickers,
        "attempts": attempts, "max_attempts": max_attempts,
        "no_trade": no_trade_count, "skipped_seen": skipped_seen,
        "stop_reason": stop_reason, "elapsed_sec": int((_dt.now() - started).total_seconds()),
        "sweep_generated_at": generated_at,
    })


# Its own attempt/time limit (not shared with the interactive /autopilot) — the hourly
# task must fit into ITS hour, otherwise the next tick may overlap
# the previous one that has not finished yet (overlap protection is the flag below, but it is better
# not to create a situation where it is needed at all). N=5 and the time cap are
# what was explicitly requested; the interactive /autopilot (60 min) is designed for N up to 20 and
# for a person waiting consciously, which is not the case here.
AUTOPILOT_HOURLY_N = 5
AUTOPILOT_HOURLY_TIME_CAP_MIN = 20


async def hourly_autopilot_job(context):
    """
    Once an hour while the trading session is open — the same hunt as /autopilot 5,
    but without a manual start.

    WHY. The morning auto-build gives at most one candidate list a day —
    with v4_stumpy_flat2 (and any later formula version) statistics
    start accumulating from scratch, and waiting weeks for 5-10 observations a day,
    when the market during the day gives more material than one morning
    run can process, is a waste of time. /stats can already show
    a window (7d/30d/today) precisely so that such fresh statistics can
    be looked at separately, without waiting weeks.

    WHY NOT A NEW SWEEP EVERY HOUR. full_market_sweep is ~7000 tickers
    and Yahoo's rate limit, dozens of minutes; running it every hour just so
    the candidate list is "really fresh" costs more than it gives — the
    LONG+entry_ok pool does not go stale within an hour the way individual prices
    do inside the expensive analysis (which recomputes them for every ticker anyway).
    The same saved sweep is used as by /autopilot and the morning build.

    WHY THIS DOES NOT DUPLICATE THE MORNING BUILD. tickers_seen_today (the same function
    as in /autopilot without force) excludes tickers already analysed TODAY —
    each hourly run naturally gets new names from the pool instead of
    recomputing the same 5 that were there at 7:45.

    OVERLAP PROTECTION. Reads the same state["_autopilot_running"] flag as
    /autopilot: if a manual hunt (or the previous hourly run that did not
    fit into its 20 minutes) is still running — the tick is silently skipped, it does not
    queue up and does not send a "market closed" message (it is not closed, just
    busy).
    """
    from datetime import datetime as _dt, timedelta as _td
    from config import MY_CHAT_ID, TELEGRAM_SCHEDULED_QUIET
    from utils.market_hours import is_trading_day, MARKET_OPEN
    from agents.scanner import format_single_result
    from agents.screener import (FULL_POOL_OVERRIDES, append_find_log,
                                 load_raw_sweep_results, rank_long_candidates)
    from utils.helpers import load_state, save_state
    from utils.signal_history import tickers_seen_today, INTRADAY_CUTOFF

    # EXPLICITLY the regular session (09:30-15:00), NOT is_within_market_hours() — that one
    # honestly accounts for EXTENDED_HOURS_ENABLED=True (a 04:00-20:00 ET window,
    # needed for live pre-market/after-hours signals), but here it made
    # the hourly autopilot fire from 4 a.m.: found on 2026-09-18 in the logs
    # (04:21, 05:22, 06:22, 07:21 ET). And after INTRADAY_CUTOFF (15:00)
    # found signals are marked "too_late" anyway and do not get into the
    # statistics (signal_history.py) — running the hunt later than that is pointless.
    now = _dt.now()
    if not (is_trading_day(now) and MARKET_OPEN <= now.time() < INTRADAY_CUTOFF):
        return

    state = load_state()
    if autopilot_busy(state):
        print("⏭️ [hourly autopilot] skip: a hunt is already running (manual or previous tick)", flush=True)
        return

    results, generated_at = await asyncio.to_thread(load_raw_sweep_results)
    if not results:
        print("⏭️ [hourly autopilot] skip: no saved sweep — run /autoscan first", flush=True)
        return

    pool, _ = await asyncio.to_thread(rank_long_candidates, results, FULL_POOL_OVERRIDES, "gap")
    if not pool:
        print("⏭️ [hourly autopilot] skip: the candidate pool is empty", flush=True)
        return

    seen = await asyncio.to_thread(tickers_seen_today)
    pool = [c for c in pool if c["ticker"] not in seen]
    if not pool:
        print("⏭️ [hourly autopilot] skip: the whole pool has already been analysed today", flush=True)
        return

    n = AUTOPILOT_HOURLY_N
    max_attempts = min(n * AUTOPILOT_MAX_ATTEMPTS_MULT, AUTOPILOT_MAX_ATTEMPTS_ABS)
    time_cap = _td(minutes=AUTOPILOT_HOURLY_TIME_CAP_MIN)
    started = _dt.now()

    state["_autopilot_stop"] = False
    state["_autopilot_running"] = True
    state["_autopilot_started_at"] = started.isoformat()
    save_state(state)

    found_tickers, no_trade_count, attempts = [], 0, 0
    stop_reason = None

    for candidate in pool:
        if len(found_tickers) >= n:
            break
        if attempts >= max_attempts:
            stop_reason = "attempts"
            break
        if _dt.now() - started >= time_cap:
            stop_reason = "time"
            break
        state = load_state()
        if state.get("_autopilot_stop"):
            stop_reason = "manual"
            break

        ticker = candidate["ticker"]
        attempts += 1
        try:
            result = await asyncio.to_thread(scan_single_ticker, ticker, None, attempts, len(pool))
        except Exception as e:
            print(f"   ⚠️ [hourly autopilot] {ticker}: {e}", flush=True)
            continue

        direction = result.get("direction", "NO_TRADE")
        if direction in ("LONG", "STRONG_LONG"):
            found_tickers.append(ticker)
            text = f"🕐 *Hourly autopilot — signal {len(found_tickers)}/{n}:*\n" + format_single_result(result)
            if TELEGRAM_SCHEDULED_QUIET:              # the signal will go into the daily digest
                print(text, flush=True)
            else:
                for part in split_for_telegram(text):
                    try:
                        await context.bot.send_message(MY_CHAT_ID, part, parse_mode="Markdown")
                    except Exception:
                        await context.bot.send_message(MY_CHAT_ID, part)
        else:
            no_trade_count += 1

        await asyncio.sleep(2)

    state = load_state()
    state["_autopilot_running"] = False
    save_state(state)

    if stop_reason is None:
        stop_reason = "success" if len(found_tickers) >= n else "exhausted"

    print(f"🎯 [hourly autopilot] found {len(found_tickers)}/{n}, "
          f"attempts {attempts}, NO_TRADE {no_trade_count}, stop reason: {stop_reason}", flush=True)

    # A chat message only if something was actually analysed (otherwise a quiet
    # "pool empty" tick stays silent above and does not spam), but without the live status of
    # the interactive command: the hourly tick runs in the background, not on request.
    if attempts and not TELEGRAM_SCHEDULED_QUIET:
        try:
            await context.bot.send_message(
                MY_CHAT_ID,
                f"🕐 Hourly autopilot: {len(found_tickers)}/{n} signals in "
                f"{attempts} attempts ({no_trade_count} NO_TRADE — also recorded in the statistics)."
            )
        except Exception:
            pass

    await asyncio.to_thread(append_find_log, {
        "at": _dt.now().isoformat(), "source": "hourly_autopilot",
        "want": n, "found": len(found_tickers), "tickers": found_tickers,
        "attempts": attempts, "max_attempts": max_attempts,
        "no_trade": no_trade_count, "stop_reason": stop_reason,
        "elapsed_sec": int((_dt.now() - started).total_seconds()),
        "sweep_generated_at": generated_at,
    })


async def cmd_queue(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /queue [N=10] — takes the next batch of LONG candidates from the list
    saved by the last /autoscan (agents/screener.py::save_autoscan_candidates,
    file reports/autoscan_candidates.json), and runs the full analysis on them,
    like /scan. The cursor is persistent (state.json) — each call continues from
    where the previous one stopped and wraps around to the start when
    the list is exhausted. The whole list is overwritten on every new
    /autoscan — the old queue loses its meaning.
    """
    print("\n📋 [/queue] command received", flush=True)
    from agents.screener import next_autoscan_queue_batch

    args = context.args or []
    try:
        n = int(args[0]) if args else 10
    except ValueError:
        n = 10
    n = max(1, min(10, n))

    batch, cursor_start, cursor_end, total, generated_at = await asyncio.to_thread(
        next_autoscan_queue_batch, n
    )
    if not batch:
        await update.message.reply_text(
            "❌ The candidate list is empty — run /autoscan first."
        )
        return

    tickers = [c["ticker"] for c in batch]
    await update.message.reply_text(
        f"📋 Candidates {cursor_start+1}-{cursor_end} of {total} "
        f"(list from {generated_at}): {', '.join(tickers)}\n"
        f"Starting the full analysis..."
    )

    report = await scan_tickers(tickers, update, context)
    parts = [report[i:i + 3800] for i in range(0, len(report), 3800)]
    for idx, part in enumerate(parts, 1):
        for attempt in range(3):
            try:
                await update.message.reply_text(part, parse_mode="Markdown")
                break
            except BadRequest:
                try:
                    await update.message.reply_text(part)
                    break
                except Exception as e2:
                    print(f"   ⚠️ Could not send part {idx}: {e2}")
            except Exception as e:
                print(f"   ⚠️ Error sending part {idx}/{len(parts)} (attempt {attempt+1}/3): {e}")
                await asyncio.sleep(2 ** attempt)
        await asyncio.sleep(0.5)


def _parse_stats_window(token):
    """
    Parses the window token for /stats: "7d"/"30d" (the last N days, including
    today), "today" (today), "YYYY-MM-DD" (a specific day).
    Returns (since, until, label) or None if the token does not look like a window
    (then the calling code treats it as a ticker).
    """
    import re
    from datetime import date, timedelta

    token_l = token.lower()
    m = re.fullmatch(r"(\d+)d", token_l)
    if m:
        n = max(1, int(m.group(1)))
        since = (date.today() - timedelta(days=n - 1)).isoformat()
        return since, None, f"last {n} d."
    if token_l == "today":
        today = date.today().isoformat()
        return today, today, "today"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", token):
        return token, token, token
    return None


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Risk-adjusted statistics over the accumulated signal history (signal_history.json):
    win rate, average return, Sharpe per trade, profit factor.

    /stats                 — the whole history (the "is there an edge" question, needs all of n)
    /stats TICKER          — for a specific ticker
    /stats 7d | 30d        — the last N days (the "how have things been
                             lately" question — operational monitoring, NOT a test
                             of the edge: the window is shorter, there is no significance in it)
    /stats today           — today
    /stats 2026-08-25      — a specific day
    Ticker and window can be combined in any order: /stats AAPL 7d.
    """
    print("\n📊 [/stats] command received", flush=True)
    from utils.signal_history import (RULE_INTRADAY, RULE_TWO_SESSIONS,
                                      LEGACY_SCORING_VERSION,
                                      compute_no_trade_counterfactual_stats,
                                      format_counterfactual_comparison)
    from utils.signals import SCORING_VERSION

    ticker = None
    since = until = window_label = None
    for arg in (context.args or []):
        window = _parse_stats_window(arg)
        if window:
            since, until, window_label = window
        else:
            ticker = arg.upper()

    # Series are shown SEPARATELY and never added up — along THREE axes:
    # the resolution rule (what counts as a win), the score formula version (what the score
    # means) and now the date window (which time slice we look at — a
    # separate question from the first two, see the compute_performance_stats docstring).
    msg = format_performance_stats(
        compute_performance_stats(ticker, RULE_INTRADAY, SCORING_VERSION, since, until))
    for older in (
        compute_performance_stats(ticker, RULE_INTRADAY, LEGACY_SCORING_VERSION, since, until),
        compute_performance_stats(ticker, RULE_TWO_SESSIONS, since=since, until=until),
    ):
        if older.get("resolved") or older.get("pending"):
            msg += "\n\n" + "─" * 20 + "\n" + format_performance_stats(older)

    # The NO_TRADE counterfactual and the score spread are diagnostics of the system's CALIBRATION
    # as a whole, not "how are things today". Shown only in the full summary without
    # a window and without a ticker: the window asks a different question, mixing them in one answer
    # would only confuse.
    if not ticker and not window_label:
        msg += format_counterfactual_comparison(compute_no_trade_counterfactual_stats())
        msg += "\n\n" + format_score_distribution(compute_score_distribution())
        from utils.deepseek_verdict import compute_deepseek_stats, format_deepseek_stats
        msg += format_deepseek_stats(compute_deepseek_stats())

    for part in split_for_telegram(msg):
        try:
            await update.message.reply_text(part, parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(part)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Help for all commands, grouped by stage of work rather than as a flat
    list: the funnel commands (/autoscan → /refilter → /queue) make no sense
    individually, and the order in the text is the only place where it is written down for
    the user.

    TWO MARKUP RULES that are easy to break here (both have already broken the command):
    1. Any token with '_' must be in backticks. Telegram in
       parse_mode="Markdown" (legacy) treats '_' as an opening italic, and with
       an odd number of underscores answers 400 "can't find end of the
       entity" — the message does not go through AT ALL. That is exactly what happened: five
       underscores (`atr_min`, `atr_max`, `gap_pct`, `no_earnings`) outside
       backticks, /help silently did not answer. Inside code markup entities are not
       parsed, so the backticks cure the problem for the future, not only for
       the current text.
    2. Do not put ']' right next to '(' — that is link syntax and will break parsing
       just as quietly.

    Plain-text fallback — as in all other handlers: the user needs the help
    more than its formatting.
    """
    text = (
        "🤖 *Day Trader v2*\n"
        "Main path: /autoscan → /refilter → /queue. Selection and thresholds "
        "are computed by Python, the model only describes the result.\n"
        "\n"
        "*Whole-market funnel*\n"
        "/autoscan — a pass over ~6900 stocks (40-50 min). Only computes and "
        "stops: shows how many passed the entry filter and on which "
        "criterion the rest were screened out. Analyses nothing by itself.\n"
        "/refilter [params] — recompute thresholds on the last "
        "/autoscan's data in seconds, without a new market pass.\n"
        "   • thresholds: `volume=` `rr=` `price=` `atr_min=` `atr_max=` `dv=` "
        "`gap_pct=`\n"
        "   • earnings: `earnings=N` — only those reporting in ≤N d.; "
        "`no_earnings=N` — the opposite, hide them\n"
        "   • sort: `sort=` gap | `volume_ratio` | `dollar_volume` | "
        "`rr1` | `atr_pct`\n"
        "   • example: `/refilter volume=1.5 rr=2 sort=dollar_volume`\n"
        "/find [N=10] — collect N candidates by automatically stepping down "
        "the relaxation ladder on the last /autoscan's data. Seconds, no "
        "network and no models. The key thing in the answer is the step number: step zero "
        "means the market is generous, step three — that N was collected only at the cost of "
        "relaxed thresholds. It does not go below the last step.\n"
        "/autopilot N [force] — looks for N REAL LONG signals (not "
        "candidates), running the expensive full analysis over the whole "
        "`LONG+entry_ok` pool until N are collected, the pool runs out or "
        "a safety stop fires (attempt/time limit). Minutes to hours, "
        "not seconds. Each found signal arrives right away, without waiting "
        "for the hunt to end. force — do not skip tickers already analysed "
        "today. /autopilot stop — stop.\n"
        "/queue [N=10] — the next N candidates from the list for full analysis "
        "(at most 10 at a time). The cursor remembers its place between calls.\n"
        "/morning [on|off|now|n NUMBER] — the same, but on its own: every trading "
        "day at 07:45 ET it passes the market and sends a ready list before the open. "
        "Skips weekends and exchange holidays; if the pass fails, it takes "
        "the previous sweep and says so explicitly. Without arguments — status.\n"
        "\n"
        "*Single-ticker analysis*\n"
        "/now TICKER — full analysis of one stock (4-5 model calls, "
        "a couple of minutes)\n"
        "/scan TICKER1 TICKER2 ... — the same for 2-10 tickers\n"
        "/go TICKER N — monitor every N minutes (N ≥ 5); outside the trading "
        "session ticks are skipped\n"
        "/status — active monitors\n"
        "/stop TICKER or /stop all — stop\n"
        "\n"
        "*Quick screen (no models)*\n"
        "/screen [source] [N=15] — sources: `trending`, `actives`, "
        "`gainers`, `losers`, `small_gainers`; N from 3 to 30\n"
        "/mass [N=100] [batch=10] — a cursor pass over the market, continues from "
        "where the last call stopped; N from 10 to 200, batch from 5 to 25\n"
        "\n"
        "*Raw lists for manual analysis*\n"
        "/earnings N — stocks reporting in the next N days across the market\n"
        "/earnings N M — earnings in a window from N to M days. No scoring or "
        "ranking, reads a ready cache of dates.\n"
        "/insiders [days=2] [amount=25000] [role] — open-market insider "
        "purchases (Form 4, code=P, no 10b5-1) across the market; days from 1 to "
        "30. Example: /insiders 3 50000 CEO\n"
        "   • both accept `save` — the found tickers become the list "
        "for /queue, and the full analysis runs on them. There is one list for all: "
        "the previous queue from /autoscan will be overwritten, and /refilter does not "
        "work on such a list — it has no sweep fields.\n"
        "\n"
        "*Validation and statistics*\n"
        "/stats [TICKER] [window] — win rate, return, Sharpe and profit factor over "
        "the accumulated signal history. Without a window — the whole history (the \"is there "
        "an edge\" question, needs all of n). Window — `7d`/`30d` (the last N days), "
        "`today`, a date `YYYY-MM-DD`: the \"how are things now\" question, not an "
        "edge test — on a small n it shows a warning instead of a conclusion. Ticker and "
        "window combine in any order.\n"
        "/lab [days=5] — whether filter thresholds give a real edge on the "
        "sweep archive. The answer deliberately does not look like a recommendation: with few "
        "sweeps it will honestly say it is too early for conclusions.\n"
        "/news — whole-market overview: indices, VIX, sectors, news background\n"
        "/news TICKER — news for a specific stock\n"
        "\n"
        "/help or /start — this help"
    )
    for part in split_for_telegram(text):
        try:
            await update.message.reply_text(part, parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(part)