# -*- coding: utf-8 -*-
"""
DEEPSEEK vs THE BOT'S REAL PAST SIGNALS — a retrospective, not a backtest replay.

A separate branch from the live pipeline and from agents/historical_backtest_llm.py.
The difference from historical_backtest_llm.py is fundamental: that experiment fed
the LLM ANONYMISED data on a REPLAY of the formula over historical prices (no VWAP,
no news, no real ticker) — there an LLM with real context could not be told apart
from an LLM without it, and the verdict predictably did not differ from a coin. Here
it is the opposite: the bot's REAL past signals are taken from signal_history.json
(the real ticker, the real features at signal time, an already known outcome) — that
is, the same data format the live path sees in utils/deepseek_verdict.py,
only not forward but over trades already played out in the past period. Leakage
here is of exactly the same type as in the live path (DeepSeek may remember what happened
to the company next) — this is a CONSCIOUS trade-off for speed: run a month
of statistics in one evening instead of waiting weeks for the live path to accumulate
as much. The result still has to be checked against the live path's conclusions, not
replace it.

REWRITTEN 2026-09-21: previously DeepSeek saw the bot's finished decision (direction,
score, entry/stop/targets) and answered "agree/disagree" — per the
owner's instruction this was the wrong setup, DeepSeek was only reacting to someone else's conclusion.
Now _build_prompt (utils/deepseek_verdict.py) gives it THE SAME raw
features the algorithm sees, WITHOUT direction/scores/entry/stop/target — it
gives a fully independent verdict of its own (LONG/SHORT/NO_TRADE). Here,
in summary(), this independent decision is compared (a) with what the bot actually
did (agrees_with_bot) and (b) with the real outcome via the return of
DeepSeek's hypothetical position (deepseek_return_pct) vs. the bot's return
on the same signals (bot_return_pct) — so what is measured is not "agreement" but "would it have won
more or less than the bot". Old records under the previous methodology (n=20,
field direction_lean) are archived in reports/deepseek_historical_verdict_v1_agree_disagree.json.bak
— not comparable with the new format, do not mix.

RUN DIRECTLY ON THE HOST, NOT VIA DOCKER. signal_history.json is an ordinary file on
disk (the same one mounted into the container), and Chrome with DeepSeek is also on
the host. No need for host.docker.internal + a Host header — that was only
needed inside the container; here Chrome and Python are on the same network
(localhost), no workaround is needed at all.

Run (from the project root):
    python agents/deepseek_historical_verdict.py                  LONG over 30 days
    python agents/deepseek_historical_verdict.py --direction SHORT --days 14
    python agents/deepseek_historical_verdict.py --summary          summary of what is already collected
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from utils.deepseek_verdict import (_ChromeDeepSeek, _build_prompt, _parse_verdict,  # noqa: E402
                                    tab_lock, HOST_TAB_LOCK_FILE)

for _s in (sys.stdout, sys.stderr):
    try:
        # line_buffering=True — without it, on some terminals / when output
        # is not a console, print() accumulates in the buffer and is not shown until the buffer
        # fills or the process exits: it looked "stuck"
        # although the script was actually working and waiting for DeepSeek's answer.
        _s.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

SIGNAL_HISTORY_PATH = os.path.join(ROOT, "signal_history.json")
OUTPUT_PATH = os.path.join(ROOT, "reports", "deepseek_historical_verdict.json")
# Same exclusions as utils/signal_history.py EXCLUDED_OUTCOMES — signals
# without a meaningful outcome (unresolved, outside session, too late) carry
# no information either for DeepSeek or for the comparison.
EXCLUDED_OUTCOMES = ("too_late", "unresolved", "out_of_session", "premarket", None, "")
NEW_CHAT_EVERY = 5


def _return_pct(direction, entry, outcome_price):
    if entry is None or outcome_price is None or entry == 0:
        return None
    if direction == "LONG":
        return round((outcome_price - entry) / entry * 100, 2)
    return round((entry - outcome_price) / entry * 100, 2)


def _actual_direction(entry, outcome_price):
    """Which direction would actually have been right by the price — not the bot's
    opinion and not DeepSeek's, but what physically happened to the price."""
    if entry is None or outcome_price is None or entry == 0:
        return None
    if outcome_price > entry:
        return "LONG"
    if outcome_price < entry:
        return "SHORT"
    return None


def collect_signals(direction, days):
    with open(SIGNAL_HISTORY_PATH, encoding="utf-8") as f:
        signal_history = json.load(f)
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    out = []
    for ticker, records in signal_history.items():
        for r in records:
            if r.get("direction") != direction:
                continue
            if (r.get("timestamp") or "") < cutoff:
                continue
            if r.get("outcome") in EXCLUDED_OUTCOMES:
                continue
            out.append((ticker, r))
    out.sort(key=lambda x: x[1].get("timestamp") or "")
    return out


def _as_trader_plan(record):
    """signal_history.json stores total_score, while _build_prompt (like the live
    trader_plan from agents/trader.py) expects direction_score — the same rename
    that append_signal does in the opposite direction when writing."""
    plan = dict(record)
    plan["direction_score"] = record.get("total_score")
    return plan


def run(direction, days, limit=None):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    pending = collect_signals(direction, days)
    if limit:
        pending = pending[:limit]
    print("%s signals over %d days with a known outcome: %d" % (direction, days, len(pending)))
    if not pending:
        return

    existing = {}
    if os.path.isfile(OUTPUT_PATH):
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            existing = json.load(f)

    ctl = _ChromeDeepSeek(debug_host="localhost", debug_port=9222)
    # A SHARED tab with the live built-in path (utils/deepseek_verdict._worker_loop)
    # — the same way it does, via _attach_to_tab. A separate tab via
    # /json/new was tried (open_new=True) — in this particular Chrome (a temporary
    # profile from start_chrome_debug.bat) a freshly created tab
    # consistently crashes with Inspector.targetCrashed; it is not a timing race but
    # a systemic environment problem. So the restriction is simple: do NOT run
    # this pass while the live path is actually sending something (the hourly autopilot
    # during market hours, a manual /now) — otherwise two clients will type into the same input
    # field at once. Safest of all — outside trading hours.
    with tab_lock(HOST_TAB_LOCK_FILE, owner="historical"):
        if not ctl.connect(open_new=False):
            print("Chrome/DeepSeek unavailable: %s" % ctl.ready_reason)
            print("Open Chrome with --remote-debugging-port=9222 and log in to chat.deepseek.com")
            return
        ctl.new_chat()

    done = skipped = failed = consecutive_failures = 0
    for i, (ticker, record) in enumerate(pending):
        key = "%s|%s" % (ticker, record.get("timestamp"))
        if key in existing:
            skipped += 1
            continue
        plan = _as_trader_plan(record)
        prompt = _build_prompt(ticker, plan)
        t0 = time.time()
        # Print BEFORE sending and flush right away — previously the output for one ticker
        # was printed in ONE chunk only after the full answer (up to 300s+ of silence,
        # looked like a hang). Now you can see the question went out without waiting
        # for the answer.
        print("  %-8s sending question..." % ticker, end="", flush=True)
        # Lock per ticker, not for the whole run — otherwise a multi-hour run
        # would never let the live bot into the tab.
        with tab_lock(HOST_TAB_LOCK_FILE, owner="historical"):
            if i > 0 and done > 0 and done % NEW_CHAT_EVERY == 0:
                ctl.new_chat()
            reply = ctl.send_message(prompt)
        print("\r  %-8s answer arrived in %.0fs" % (ticker, time.time() - t0), flush=True)
        if reply is None:
            # One failed record must not stop the whole run —
            # the goal is "run everything there is" (the owner's request). Mark it
            # as failed (it will not be recomputed on the next
            # run) and move on, with a fresh chat just in case. But
            # if MANY tickers in a row fail, that is no longer one stubborn
            # ticker but a real problem (account limit, Chrome dropped) —
            # there is no point hammering further, abort.
            consecutive_failures += 1
            failed += 1
            print("  %-8s no answer — marking as failed, moving on (failures in a row: %d)"
                 % (ticker, consecutive_failures), flush=True)
            existing[key] = {"ticker": ticker, "signal_timestamp": record.get("timestamp"),
                             "bot_direction": direction, "failed": True,
                             "asked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")}
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=1)
            if consecutive_failures >= 3:
                print("  3 failures in a row — looks like a systemic problem (limit/Chrome), "
                     "aborting the run", flush=True)
                break
            with tab_lock(HOST_TAB_LOCK_FILE, owner="historical"):
                ctl.new_chat()
            time.sleep(5)
            continue
        consecutive_failures = 0
        v = _parse_verdict(reply)
        ds_dir = v.get("deepseek_direction_norm")
        bot_return = _return_pct(direction, record.get("entry_price"), record.get("outcome_price"))
        # Hypothetical return of DeepSeek's OWN choice — regardless of
        # what the bot did. NO_TRADE/unparsed answers do not take part
        # in this comparison (no trade — no return).
        ds_return = (_return_pct(ds_dir, record.get("entry_price"), record.get("outcome_price"))
                    if ds_dir in ("LONG", "SHORT") else None)
        actual_dir = _actual_direction(record.get("entry_price"), record.get("outcome_price"))
        # "Is DeepSeek right" — compared with what ACTUALLY happened to the price,
        # not with the bot's opinion. NO_TRADE/unparsed answers are neither right nor
        # wrong and do not count in the percentage (no bet — nothing to score).
        ds_correct = (ds_dir == actual_dir) if (ds_dir in ("LONG", "SHORT") and actual_dir) else None
        rec_out = {
            "ticker": ticker, "signal_timestamp": record.get("timestamp"),
            "bot_direction": direction, "bot_outcome": record.get("outcome"),
            "bot_return_pct": bot_return,
            "deepseek_direction": v.get("deepseek_direction"),
            "deepseek_direction_norm": ds_dir,
            "deepseek_return_pct": ds_return,
            "actual_direction": actual_dir,
            "deepseek_correct": ds_correct,
            "agrees_with_bot": (ds_dir == direction) if ds_dir else None,
            "confidence": v.get("confidence"),
            "tail_risk": v.get("tail_risk"), "tail_risk_why": v.get("tail_risk_why"),
            "reasoning": v.get("reasoning"), "raw": v.get("raw"),
            "asked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        }
        existing[key] = rec_out
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=1)
        done += 1
        correct_mark = "?" if ds_correct is None else ("✅right" if ds_correct else "❌wrong")
        print("  %-8s %-17s bot=%-6s deepseek=%-8s actual=%-6s %-8s tail=%-4s (%.0fs)"
             % (ticker, (record.get("timestamp") or "")[:16], direction, ds_dir or v.get("deepseek_direction"),
                actual_dir, correct_mark, v.get("tail_risk"), time.time() - t0))
        time.sleep(2)

    ctl.close()
    print("Done: processed %d, skipped (already there) %d, failed %d" % (done, skipped, failed))


def summary():
    if not os.path.isfile(OUTPUT_PATH):
        print("File %s does not exist yet" % OUTPUT_PATH)
        return
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        data = json.load(f)
    rows = list(data.values())
    print("Total collected: %d" % len(rows))
    with_return = [r for r in rows if r.get("bot_return_pct") is not None]
    print("With a known bot return: %d" % len(with_return))
    if not with_return:
        return

    def avg(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    # MAIN PERCENTAGE: is DeepSeek right by the actual price (not by the bot's opinion).
    graded = [r for r in with_return if r.get("deepseek_correct") is not None]
    correct = [r for r in graded if r["deepseek_correct"] is True]
    print()
    if graded:
        print("DeepSeek called the direction right (by actual price): %d/%d (%.0f%%)"
             % (len(correct), len(graded), 100.0 * len(correct) / len(graded)))
    else:
        print("DeepSeek called the direction right (by actual price): no directional answers to evaluate")

    agreed = [r for r in with_return if r.get("agrees_with_bot") is True]
    disagreed = [r for r in with_return if r.get("agrees_with_bot") is False]
    print()
    print("DeepSeek agreed with the bot's direction: %d/%d (%.0f%%)"
         % (len(agreed), len(with_return), 100.0 * len(agreed) / len(with_return)))

    # MAIN COMPARISON: return of DeepSeek's OWN choice vs. the bot's return
    # on THE SAME signals — not "did it agree" but "would it have won more".
    ds_directional = [r for r in with_return if r.get("deepseek_return_pct") is not None]
    print()
    print("Of these, DeepSeek made a directional call (not NO_TRADE/unparsed): %d" % len(ds_directional))
    if ds_directional:
        print("  Bot return on the same signals:           %s%%" % avg([r["bot_return_pct"] for r in ds_directional]))
        print("  Hypothetical return of DeepSeek's choice:  %s%%" % avg([r["deepseek_return_pct"] for r in ds_directional]))

    tail_yes = [r for r in with_return if (r.get("tail_risk") or "").lower().startswith("y")]
    tail_no = [r for r in with_return if not (r.get("tail_risk") or "").lower().startswith("y")]
    print()
    print("Flagged TAIL_RISK=yes (n=%d): actual bot return %s%%"
         % (len(tail_yes), avg([r["bot_return_pct"] for r in tail_yes])))
    print("Not flagged (n=%d): actual bot return %s%%"
         % (len(tail_no), avg([r["bot_return_pct"] for r in tail_no])))
    print()
    print("MAIN CHECK (metric fixed in advance, see the docstring of")
    print("utils/deepseek_verdict.py): if the hypothetical return of DeepSeek's OWN")
    print("choice is noticeably and consistently better than the bot's return on the same")
    print("signals — there is value. If comparable or worse — a negative")
    print("result, not 'too little data'. Separately: if the return for")
    print("tail_risk=yes is noticeably worse than for tail_risk=no — DeepSeek catches")
    print("the catastrophic tail. n=%d — probably too little for a final" % len(with_return))
    print("conclusion; check against what the live path accumulates, not only this run.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="LONG", choices=["LONG", "SHORT"])
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary()
    else:
        run(args.direction, args.days, args.limit)
        summary()
