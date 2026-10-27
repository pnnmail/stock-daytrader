# -*- coding: utf-8 -*-
"""
A parallel web-DeepSeek verdict on live LONG/SHORT signals. NOT PART
OF SCORING — a background queue thread; the result never goes back into
run_trader/compute_factor_scores and does not affect direction/entry. Only
the DEEPSEEK_VERDICTS_FILE plus a separate Telegram message.

WHY A LIVE VERDICT AND NOT ANOTHER RUN ON ANONYMISED HISTORY. Already
tested (agents/historical_backtest_llm.py, 2026-09-12): the LLM verdict on 300
anonymised historical observations gave 51.7% vs. 52.0% for Python —
indistinguishable. The reason is not that the model is weak, but that anonymisation
tests exactly the dimension (guessing direction from bare numbers) where
DeepSeek is obviously not needed — Python already computes that more precisely. The only reason
to call an LLM at all is context that is not in the numbers:
a specific company, a specific news item, a specific risk right now. That
dimension is removed by anonymisation by construction, so it cannot be tested on
history even in theory — not "imprecisely" but outright impossible. Hence:
forward only, on real tickers, without anonymisation (there is no LLM memory leakage here
and cannot be — the outcome has not happened yet).

SUCCESS METRIC (fixed BEFORE data collection so as not to fit it after the fact).
Not directional hit rate — that is already measured and gave a coin. We look at: among
trades where DeepSeek agreed with the bot's direction, do those flagged
tail_risk=yes have a worse mean return/failure share than the rest? If yes
and consistently on a sufficient sample — DeepSeek catches the catastrophic tail,
which is valuable. If there is no difference — it is closed the same way as
short_ratio (utils/short_volume.py): a negative result on a sufficient
sample, not "too little data, let's wait". Compare no earlier than after 8-12 weeks
of accumulation (5-6 weeks was too early even for the much simpler
short_ratio).

WHY VIA host.docker.internal AND NOT A SEPARATE PROCESS ON THE HOST. Chrome with
chat.deepseek.com logged in must be open on the HOST (the controller does not store cookies
and does not log in by itself). The bot lives in the Docker network and physically cannot
see the host's localhost, but Docker Desktop on Windows forwards
host.docker.internal. The only obstacle is Chrome itself: since 2026
CDP rejects requests with a Host header other than "localhost"/IP (protection
against DNS rebinding), and host.docker.internal is not one. Workaround — connect
via host.docker.internal but explicitly set Host: localhost:9222 in HTTP and
WebSocket requests. Verified live on 2026-09-21: a full REST /json + WS
Runtime.evaluate cycle against a real chat.deepseek.com tab on the host succeeded
from inside the bot's container.

WHY ONE BACKGROUND QUEUE THREAD AND NOT A THREAD PER SIGNAL. Web DeepSeek is
one tab, one conversation: two simultaneous send_message calls would mangle
each other (type text over half-typed text). The hourly autopilot can
find up to 5 signals per tick — they go into a queue and are processed one
at a time rather than in parallel. An answer can take up to 180s — this must NOT
block either the bot's main thread or the Telegram handlers, so
request_verdict_background() only enqueues the task and returns
immediately; all network work happens in a daemon thread.

WHAT IT DOES IF CHROME IS UNAVAILABLE. Silently skips this attempt and logs
a warning — the bot's signal has already been sent and is not affected at all
(the call happens AFTER the main message). Nothing is retried
forever: a missing verdict is not a disaster, it is diagnostics, not
a trading decision.
"""
import contextlib
import json
import logging
import os
import queue
import re
import threading
import time
from datetime import datetime, timezone

import requests

from config import (DEEPSEEK_VERDICT_ENABLED, DEEPSEEK_DEBUG_HOST, DEEPSEEK_DEBUG_PORT,
                    DEEPSEEK_RESPONSE_TIMEOUT, DEEPSEEK_VERDICTS_FILE, DEEPSEEK_NEW_CHAT_EVERY,
                    DEEPSEEK_TAB_LOCK_FILE, TELEGRAM_BOT_TOKEN, MY_CHAT_ID,
                    TELEGRAM_SCHEDULED_QUIET)

logger = logging.getLogger(__name__)

# The same file as DEEPSEEK_TAB_LOCK_FILE inside the container (./reports is mounted
# at /app/reports) — the path for scripts that talk to Chrome directly from the host.
HOST_TAB_LOCK_FILE = os.environ.get("DEEPSEEK_HOST_TAB_LOCK_FILE", os.path.join("reports", "deepseek_tab.lock"))
# Longer than the longest legitimate hold (parsing the insider list: answer timeout
# up to 900s, hard ceiling twice that) — only then is the lock considered
# abandoned by a crashed process and taken over.
TAB_LOCK_STALE_SECONDS = 2400


@contextlib.contextmanager
def tab_lock(path: str = None, owner: str = "stock_agent2"):
    """One DeepSeek tab for several independent processes: the live bot in
    the container and scripts on the host (historical run, picking from the insider
    list). Two clients in one chat type over each other and
    navigate the page in the middle of someone else's answer — so the whole cycle
    "connect/new chat/ask/wait" holds a file lock in a shared
    mounted folder. O_CREAT|O_EXCL over a Docker Desktop bind mount is not
    guaranteed to be atomic, but holds last minutes, not milliseconds —
    a race at the same instant does not happen in practice."""
    path = path or DEEPSEEK_TAB_LOCK_FILE
    waiting = path + ".waiting"
    waited = False
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write("%s %s" % (owner, datetime.now().isoformat(timespec="seconds")))
            break
        except FileExistsError:
            waited = True
            try:
                with open(waiting, "w") as f:
                    f.write(owner)
            except OSError:
                pass
            try:
                age = time.time() - os.path.getmtime(path)
            except OSError:
                continue
            if age > TAB_LOCK_STALE_SECONDS:
                logger.warning("DeepSeek: tab lock abandoned (%.0fs) — taking it over", age)
                try:
                    os.remove(path)
                except OSError:
                    pass
                continue
            time.sleep(2)
    if waited:
        try:
            os.remove(waiting)
        except OSError:
            pass
    try:
        yield
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
        # Fairness: the live bot processes the verdict queue back to back and without a pause
        # would grab the tab right after each of its own answers — a script on
        # the host could wait for hours. If someone registered as waiting in the last
        # few seconds — yield to them before taking the tab again.
        try:
            if time.time() - os.path.getmtime(waiting) < 10:
                time.sleep(6)
        except OSError:
            pass

DEEPSEEK_URL = "https://chat.deepseek.com/"

INPUT_SELECTORS = [
    'textarea#chat-input', 'textarea[placeholder]',
    'div[contenteditable="true"]', 'textarea',
]
ASSISTANT_MSG_SELECTORS = [
    '[class*="ds-markdown"]', '[class*="_4f9bf79"]',
    '[class*="message-content"]', '[class*="markdown"]',
]
ROOTS_JS = ("[...document.querySelectorAll(s)]"
           ".filter(n => !(n.parentElement && n.parentElement.closest(s)))")
SEEN_ATTR = "data-ds-agent-seen"
GENERATING_SELECTORS = ['[class*="stop"]', 'div[class*="loading"] svg',
                        '[class*="generating"]', 'button[aria-label*="Stop"]',
                        'button[aria-label*="停止"]']
LOGIN_MARKERS = ["log in", "sign in", "登录", "log in with"]
TROUBLE_MARKERS = ["rate limit", "too many requests", "try again later",
                   "服务器繁忙", "try again later", "server is busy"]
CONTINUE_TEXTS = ["continue generat", "continue", "继续生成", "继续"]
SUBMIT_WAIT = 2.5
ANSWER_FALLBACK_QUIET_SECONDS = 180


class _ChromeDeepSeek:
    """A trimmed copy of a standalone chrome_controller.py. By default it is configured
    for access from the container (host.docker.internal + Host header), but
    host/port are parameterised — when running directly on the host (e.g.
    agents/deepseek_historical_verdict.py) pass debug_host="localhost":
    then host_header is not needed at all, Chrome lets localhost in anyway."""

    def __init__(self, debug_host: str = None, debug_port: int = None):
        self.debug_host = debug_host or DEEPSEEK_DEBUG_HOST
        self.debug_port = debug_port or DEEPSEEK_DEBUG_PORT
        # The Host header for bypassing Chrome's DNS rebinding protection is needed only
        # if we connect NOT via localhost/IP (see the module docstring) —
        # with debug_host="localhost" Chrome itself won't require it.
        self._host_header = "localhost:%d" % self.debug_port
        self.ws = None
        self.command_id = 0
        self.ready_reason = ""
        self.last_complete = False

    def connect(self, open_new: bool = False) -> bool:
        """open_new=True — do not look for an existing tab, open our own.
        Needed for parallel clients (e.g. the one-off historical run
        agents/deepseek_historical_verdict.py running AT THE SAME TIME as the live
        built-in path): DeepSeek is one conversation per tab, two clients on
        one tab will type over each other (see the module docstring
        about "one background queue thread"). Different tabs = different conversations,
        they don't interfere."""
        if not open_new and self._attach_to_tab():
            return self._verify_ready()
        if self._open_tab():
            return self._verify_ready()
        logger.error("DeepSeek tab not found/did not open (%s:%d)",
                     self.debug_host, self.debug_port)
        self.ready_reason = "DeepSeek tab not found"
        return False

    def _verify_ready(self) -> bool:
        ok, reason = self.check_ready()
        self.ready_reason = reason
        if ok:
            self._enable_search_deepthink()
        return ok

    def _force_active(self):
        """THE ROOT CAUSE of all of today's "no answer" — found and confirmed
        live on 2026-09-22. The Chrome tab is almost always in the background on the host screen
        (the user keeps it open but not active) — document.hidden=
        true. DeepSeek visually renders the answer fully and quickly (confirmed
        with Page.captureScreenshot screenshots — the compositor paints the frame
        regardless of tab visibility), but the React app itself apparently
        commits the final DOM structure of the message (.ds-message/.ds-markdown)
        via a deferred/throttled path (requestIdleCallback or similar),
        which Chrome holds back hard for background tabs — because of this
        querySelectorAll saw 0 elements for 300+ seconds although the answer was long
        ready. Reproduced and fixed: without this method a trivial
        question was not detected for 300+s, with it — 11.2s. Page.setWebLifecycleState
        + Emulation.setFocusEmulationEnabled make the tab consider itself
        active/focused for the CDP target regardless of the real window state
        on the user's screen."""
        self._send("Page.setWebLifecycleState", {"state": "active"})
        self._send("Emulation.setFocusEmulationEnabled", {"enabled": True})

    def _enable_search_deepthink(self):
        """DeepThink and Search must ALWAYS be ON — per the owner's direct
        instruction (2026-09-22): deep analysis and web search are part of what
        DeepSeek should do here, not extra load. (Previously it was
        the opposite — both toggles were switched off, mistaking their long answer delay for
        the cause of "no answer"; the real cause turned out to be not them but
        the background tab — see _force_active(). We keep both on and
        rely on _force_active() + an increased timeout to have time
        to wait for a slower but more substantive answer.)
        The toggles do not always return to the on state by themselves — just
        in case we re-sync them on every connection and new_chat()."""
        self._eval("""(() => { const targets = ['DeepThink', 'Search'];
          const btns = [...document.querySelectorAll('[role="button"], button, div[class*="button"]')];
          for (const b of btns) {
            const t = (b.innerText || '').trim();
            if (targets.includes(t) && b.getAttribute('aria-pressed') === 'false') b.click();
          }
        })()""")

    def _list_tabs(self):
        r = requests.get("http://%s:%d/json" % (self.debug_host, self.debug_port),
                         headers={"Host": self._host_header}, timeout=5)
        r.raise_for_status()
        return r.json()

    def _attach_to_tab(self) -> bool:
        try:
            tabs = self._list_tabs()
        except requests.RequestException as e:
            logger.warning("CDP unavailable at %s:%d: %s", self.debug_host, self.debug_port, e)
            return False
        for tab in tabs:
            if "deepseek" in tab.get("url", "").lower() and tab.get("webSocketDebuggerUrl"):
                ws_url = tab["webSocketDebuggerUrl"].replace(
                    "ws://localhost:%d" % self.debug_port,
                    "ws://%s:%d" % (self.debug_host, self.debug_port))
                return self._connect_ws(ws_url)
        return False

    def _open_tab(self) -> bool:
        """Creates a tab via /json/new and explicitly sends Page.navigate — new
        Chrome versions do not always honour auto-navigation via the query parameter
        /json/new?URL (verified live on 2026-09-21: the tab was created
        empty, about:blank, without navigation)."""
        try:
            r = requests.put("http://%s:%d/json/new" % (self.debug_host, self.debug_port),
                             headers={"Host": self._host_header}, timeout=5)
            r.raise_for_status()
            tab = r.json()
        except requests.RequestException as e:
            logger.error("Could not open a new DeepSeek tab: %s", e)
            return False
        # Race: connecting to the socket right after creating the tab sometimes does not
        # make it — the tab itself is not ready to accept a WS client yet (verified
        # live on 2026-09-21: without a pause — handshake timeout, with a 1s pause —
        # connects reliably).
        time.sleep(1)
        ws_url = (tab.get("webSocketDebuggerUrl") or "").replace(
            "ws://localhost:%d" % self.debug_port, "ws://%s:%d" % (self.debug_host, self.debug_port))
        if not self._connect_ws(ws_url, navigate_first=True):
            return False
        time.sleep(4)  # extra time for the DeepSeek page to draw the input field
        return True

    def _connect_ws(self, ws_url: str, navigate_first: bool = False) -> bool:
        """navigate_first=True — for a FRESHLY CREATED (via /json/new) empty
        tab. Verified live on 2026-09-21: if Runtime.enable is sent to a bare about:blank
        right after creation, Chrome does not answer at all
        (hangs on recv() until timeout) — the target really
        "comes alive" for CDP only once it has somewhere to navigate to.
        On an already loaded existing tab (_attach_to_tab) this problem
        does not reproduce — there navigate_first is neither needed nor passed."""
        import websocket
        try:
            self.ws = websocket.create_connection(ws_url, timeout=30, suppress_origin=True,
                                                  host=self._host_header)
            if navigate_first:
                self._send("Page.navigate", {"url": DEEPSEEK_URL})
                time.sleep(3)
            self._send("Runtime.enable", {})
            self._send("Page.enable", {})
            self._force_active()
            return True
        except Exception as e:  # noqa: BLE001
            logger.error("WebSocket error to DeepSeek on the host: %s", e)
            return False

    def send_message(self, message: str, timeout: int = None):
        if not self.ws:
            return None
        # Another tab client (see tab_lock) may have navigated the page between
        # our messages — after navigation the activity state may
        # reset, and without it the answer is not visible in the DOM (see _force_active).
        self._force_active()
        prev_text = self._last_assistant_text()
        self._mark_seen_answers()
        self._focus_input()
        self._clear_input()
        self._insert_text(message)
        time.sleep(0.4)
        if not self._submit():
            logger.error("Message not sent: the input field did not clear")
            return None
        return self._wait_for_response(prev_text=prev_text, timeout=timeout)

    def _focus_input(self):
        self._eval("""(() => { const sels = %s;
          for (const s of sels) { const el = document.querySelector(s);
            if (el) { el.focus(); el.click(); return s; } } return null; })()""" % json.dumps(INPUT_SELECTORS))

    def _insert_text(self, text: str):
        self._send("Input.insertText", {"text": text})
        got = self._eval("""(() => { const sels = %s;
          for (const s of sels) { const el = document.querySelector(s);
            if (el) return (el.value || el.innerText || '').length; } return 0; })()"""
                         % json.dumps(INPUT_SELECTORS))
        if isinstance(got, int) and got == 0:
            self._eval("""(() => { const sels = %s; const txt = %s;
              for (const s of sels) { const el = document.querySelector(s);
                if (!el) continue;
                if ('value' in el) el.value = txt; else el.innerText = txt;
                el.dispatchEvent(new InputEvent('input', {bubbles:true}));
                return true; } return false; })()""" % (json.dumps(INPUT_SELECTORS), json.dumps(text)))

    def _input_length(self) -> int:
        val = self._eval("""(() => { const sels = %s;
          for (const s of sels) { const el = document.querySelector(s);
            if (el) return ((el.value || el.innerText || '').trim()).length; } return 0; })()"""
                         % json.dumps(INPUT_SELECTORS))
        return val if isinstance(val, int) else 0

    def _clear_input(self):
        self._eval("""(() => { const sels = %s;
          for (const s of sels) { const el = document.querySelector(s);
            if (!el) continue;
            if ('value' in el) el.value = ''; else el.innerText = '';
            el.dispatchEvent(new InputEvent('input', {bubbles:true}));
            return true; } return false; })()""" % json.dumps(INPUT_SELECTORS))

    def _click_send(self) -> bool:
        return bool(self._eval("""(() => { const sels = %s;
          let el = null; for (const s of sels) { el = document.querySelector(s); if (el) break; }
          if (!el) return false;
          let box = el; for (let i = 0; i < 5 && box.parentElement; i++) box = box.parentElement;
          const btns = [...box.querySelectorAll('[role="button"], button')]
            .filter(b => /ds-button--primary/.test((b.className||'').toString()) && b.offsetParent !== null);
          if (!btns.length) return false;
          btns[btns.length - 1].click(); return true; })()""" % json.dumps(INPUT_SELECTORS)))

    def _submit(self) -> bool:
        self._press_enter()
        if self._wait_empty(SUBMIT_WAIT):
            self._clear_input()
            return True
        self._click_send()
        if self._wait_empty(SUBMIT_WAIT):
            self._clear_input()
            return True
        return False

    def _wait_empty(self, seconds: float) -> bool:
        deadline = time.time() + seconds
        while time.time() < deadline:
            if self._input_length() == 0:
                return True
            time.sleep(0.3)
        return self._input_length() == 0

    def _press_enter(self):
        for ev_type in ("keyDown", "char", "keyUp"):
            params = {"type": ev_type, "key": "Enter", "code": "Enter",
                     "windowsVirtualKeyCode": 13, "nativeVirtualKeyCode": 13}
            if ev_type == "char":
                params["text"] = "\r"
            self._send("Input.dispatchKeyEvent", params)
            time.sleep(0.05)

    def _mark_seen_answers(self):
        self._eval("""(() => { const sels = %s; const attr = %s;
          for (const s of sels) { const roots = %s;
            if (roots.length) { roots.forEach(n => n.setAttribute(attr, '1')); return roots.length; } }
          return 0; })()""" % (json.dumps(ASSISTANT_MSG_SELECTORS), json.dumps(SEEN_ATTR), ROOTS_JS))

    def _is_generating(self) -> bool:
        return bool(self._eval("""(() => { const sels = %s;
          return sels.some(s => document.querySelector(s) !== null); })()"""
                               % json.dumps(GENERATING_SELECTORS)))

    def _last_assistant_text(self, only_new: bool = False) -> str:
        js = """(() => {
          const sels = %s;
          let box = null;
          for (const s of sels) { const roots = %s; if (roots.length) { box = roots[roots.length - 1]; break; } }
          if (!box) return '';
          if (%s && box.getAttribute(%s) === '1') return '';
          const parts = [];
          const emitPre = (pre) => {
            const code = pre.querySelector('code') || pre;
            const m = (code.className || pre.className || '').match(/language-([\\w+-]+)/);
            const lang = m ? m[1] : '';
            parts.push('\\n```' + lang + '\\n' + (code.textContent || '').replace(/\\n$/, '') + '\\n```\\n');
          };
          const walk = (node) => {
            for (const ch of node.childNodes) {
              if (ch.nodeType === 3) { parts.push(ch.textContent); continue; }
              if (ch.nodeType !== 1) continue;
              const nn = ch.nodeName;
              if (nn === 'PRE') { emitPre(ch); continue; }
              const inner = ch.querySelectorAll ? ch.querySelectorAll('pre') : [];
              if (inner.length) { inner.forEach(emitPre); continue; }
              if (nn === 'CODE') { parts.push('`' + (ch.textContent || '') + '`'); }
              else if (nn === 'STRONG' || nn === 'B') { parts.push('__' + (ch.textContent || '') + '__'); }
              else if (nn === 'EM' || nn === 'I') { parts.push('_' + (ch.textContent || '') + '_'); }
              else if (nn === 'BR') { parts.push('\\n'); }
              else { walk(ch); if (/^(P|DIV|LI|H[1-6]|TR|UL|OL|BLOCKQUOTE)$/.test(nn)) parts.push('\\n'); }
            }
          };
          walk(box);
          return parts.join('').trim();
        })()""" % (json.dumps(ASSISTANT_MSG_SELECTORS), ROOTS_JS,
                   'true' if only_new else 'false', json.dumps(SEEN_ATTR))
        val = self._eval(js)
        return val if isinstance(val, str) else ""

    def _fresh_text(self, prev_text: str) -> str:
        text = self._last_assistant_text(only_new=True)
        if text:
            return text
        text = self._last_assistant_text()
        return text if text and text != prev_text else ""

    def new_chat(self) -> bool:
        self._eval("location.href = %s" % json.dumps(DEEPSEEK_URL))
        for _ in range(20):
            time.sleep(1.0)
            ok, _ = self.check_ready()
            if ok:
                # location.href is a real navigation (not an SPA push-state), it
                # creates a new execution context and may reset the target's
                # activity/focus state set by _force_active() in
                # connect() — re-apply it unconditionally on every new_chat().
                self._force_active()
                self._enable_search_deepthink()
                return True
        return False

    def check_ready(self):
        if not self.ws:
            return False, "no connection to Chrome (CDP)"
        js = """(() => {
          const sels = %s;
          let el = null; for (const s of sels) { el = document.querySelector(s); if (el) break; }
          const body = (document.body ? document.body.innerText || '' : '');
          const low = body.toLowerCase();
          const has = (list) => list.find(w => low.includes(w)) || '';
          return JSON.stringify({
            state: document.readyState, url: location.href, hasInput: !!el,
            blocked: el ? !!(el.disabled || el.readOnly || el.getAttribute('aria-disabled') === 'true') : false,
            messages: document.querySelectorAll('.ds-message').length,
            login: has(%s), trouble: has(%s), bodyLen: body.length
          });
        })()""" % (json.dumps(INPUT_SELECTORS), json.dumps(LOGIN_MARKERS), json.dumps(TROUBLE_MARKERS))
        raw = self._eval(js)
        if not isinstance(raw, str):
            return False, "the page does not respond to CDP requests"
        try:
            d = json.loads(raw)
        except ValueError:
            return False, "could not read the page state"
        if "deepseek" not in (d.get("url") or "").lower():
            return False, "the tab is not DeepSeek: %s" % d.get("url")
        if d.get("state") != "complete":
            return False, "the page is still loading"
        if not d.get("hasInput"):
            reason = "input field not found"
            if d.get("login"):
                reason += " — looks like it logged out"
            return False, reason
        if d.get("blocked"):
            return False, "input field is disabled"
        if d.get("trouble"):
            return False, "the page reports a problem: %s" % d["trouble"]
        return self._probe_input()

    def _probe_input(self):
        if self._input_length() > 0:
            return True, "ready (the field already has text)"
        self._focus_input()
        self._send("Input.insertText", {"text": "."})
        got = self._input_length()
        self._clear_input()
        if got <= 0:
            return False, "the input field does not accept text"
        return True, "ready"

    def _click_continue(self) -> bool:
        return bool(self._eval("""(() => { const texts = %s;
          const btns = [...document.querySelectorAll('button, [role="button"], div[class*="button"]')];
          for (const b of btns) {
            const t = (b.innerText || b.textContent || b.getAttribute('aria-label') || '').trim().toLowerCase();
            if (!t || t.length > 40) continue;
            if (texts.some(x => t.includes(x))) { if (b.offsetParent === null) continue; b.click(); return t; }
          } return null; })()""" % json.dumps(CONTINUE_TEXTS)))

    def _wait_for_response(self, prev_text: str = "", timeout: int = None):
        """A HARD ceiling hard_deadline — without it every click on the
        "continue" button pushed the deadline out by another full timeout, up to 20 times
        in a row (max_continues). If _click_continue() falsely fired on
        some page element with similar text (the selector is broad —
        the whole document, not just the answer area), this stretched the wait
        to hours without a single sign of progress — reproduced live on
        2026-09-21. hard_deadline physically cannot be pushed out by clicks."""
        timeout = timeout or DEEPSEEK_RESPONSE_TIMEOUT
        t_start = time.time()
        hard_deadline = t_start + timeout * 2
        deadline = t_start + timeout
        while time.time() < deadline:
            if self._fresh_text(prev_text) or self._is_generating():
                break
            time.sleep(1)
        else:
            return None
        last_text, stable, last_growth, first_seen = "", 0, 0.0, 0.0
        poll, continues, max_continues, len_at_continue = 1.5, 0, 20, -1
        last_heartbeat = 0.0
        while time.time() < deadline:
            generating = self._is_generating()
            text = self._fresh_text(prev_text)
            if text and not first_seen:
                first_seen = time.time()
            # A "live" status every ~15s — without it a 300s wait looks like
            # a hang, even if the answer has long been ready on screen and our
            # "text stopped growing" detector simply has not collected enough polls
            # in a row yet (see the class docstring above about false triggers).
            if time.time() - last_heartbeat >= 15:
                last_heartbeat = time.time()
                print("      ...waiting %ds, generating=%s, chars captured=%d, silence=%ds"
                     % (int(time.time() - t_start), generating, len(text),
                        int(time.time() - last_growth) if last_growth else 0), flush=True)
            if (continues < max_continues and len(text) > len_at_continue
                    and self._click_continue()):
                continues += 1
                len_at_continue = len(text)
                stable = 0
                deadline = min(time.time() + timeout, hard_deadline)
                time.sleep(2.5)
                continue
            if text and text == last_text:
                stable += 1
            else:
                stable = 0
                last_text = text
                last_growth = time.time()
            quiet = time.time() - last_growth
            # Readiness — the row of buttons under the answer (copy/regenerate/
            # rate), it appears only for a finished message. Previously
            # we trusted "8s of silence", but with DeepThink+Search the model writes a line like
            # "Now let me search" and stays silent for a minute while searching — such an intermediate
            # phrase was taken for the whole answer (2026-09-30, picking from the insider
            # list: 12 characters out of 9550 captured).
            if text and stable >= 1 and self._answer_done():
                self.last_complete = True
                return text
            # Fallback exit in case the button layout changes: only after a very
            # long silence, and the answer is marked incomplete — the calling code
            # decides whether to trust it.
            if text and stable >= 3 and quiet >= ANSWER_FALLBACK_QUIET_SECONDS:
                self.last_complete = False
                return text
            time.sleep(poll)
        self.last_complete = False
        return last_text or None

    def _answer_done(self) -> bool:
        """The last message in the chat is the model's answer, and it already has an action row
        under it. Our own message has two buttons (copy,
        edit), a finished answer has six; while the answer is being written
        there is no row or it has one button (verified live on 2026-09-30)."""
        return bool(self._eval("""(() => {
          const msgs = [...document.querySelectorAll('.ds-message')];
          const last = msgs[msgs.length - 1];
          if (!last || !last.parentElement) return false;
          return [...last.parentElement.children].some(c => c !== last &&
            c.querySelectorAll('[role="button"], button, [class*="icon-button"]').length >= 4);
        })()"""))

    def _send(self, method: str, params: dict) -> dict:
        self.command_id += 1
        cid = self.command_id
        self.ws.send(json.dumps({"id": cid, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == cid:
                return msg

    def _eval(self, expression: str):
        resp = self._send("Runtime.evaluate",
                          {"expression": expression, "returnByValue": True, "awaitPromise": True})
        if "error" in resp:
            logger.error("CDP error: %s", resp["error"])
            return None
        return resp.get("result", {}).get("result", {}).get("value")

    def close(self):
        if self.ws:
            try:
                self.ws.close()
            except Exception:  # noqa: BLE001
                pass
            self.ws = None


# ---------------------------------------------------------------- prompt
#
# REWRITTEN 2026-09-21 — previously DeepSeek was shown the bot's finished decision
# (direction/score/entry/stop/targets) and asked "do you agree". Per the
# owner's instruction this is not what is needed — DeepSeek was only reacting to
# someone else's conclusion instead of analysing by itself. Now it gets THE SAME raw
# inputs as our algorithm (features from trader_plan — the same thing
# that goes into compute_factor_scores), but WITHOUT direction/scores/entry/stop/
# target — it gives a fully independent verdict of its own (LONG/SHORT/NO_TRADE
# + confidence + tail risk), while agreement with the bot and the hypothetical
# return of its own choice are computed AFTERWARDS, in code (see
# _worker_loop._send_telegram and agents/deepseek_historical_verdict.summary),
# rather than asked of the model in text.
PROMPT_TMPL = """You are a senior trader-analyst with 20 years of experience in equities, specialising in
small/mid caps and special situations. Below is the raw data on ticker
{ticker} as of now — exactly what our algorithm sees before
making a decision. You are NOT given a ready-made decision: work through everything yourself,
from scratch, as your own analysis, and decide whether to enter a position, and if so,
in which direction.

The price and all technical figures below were taken from the exchange just now — treat them
as given, do not check them against your idea of where the price "should be",
and do not spend part of the answer on that: if it seems far off from what you
remember, it is your data that is stale, not the input data.

Before answering, briefly (2-4 points, no filler) go over the substance,
not just the numbers below:
- Do you know anything about {ticker} from recent weeks/months: earnings,
  news, insider trades (Form 4 — clustered purchases/sales by
  directors and major shareholders), legal/regulatory stories?
- Is there a catalyst or date you know of (earnings, FDA decision,
  court, licence) that may land in the next few days and blow the price
  either way?
- Does the current move look like something meaningful (a reaction to a real
  event) or like noise/a pump without a reason?
If you have little data on the company or it may be stale — say so
plainly, do not invent numbers or events.

Inputs for {ticker}:
Price: {price}
RSI: {rsi} | MACD: {macd} (signal line: {macd_signal}) | Bollinger Bands: [{bb_lower} .. {bb_upper}]
Volume vs. average: {volume_ratio}x | Position in the day's range: {position_in_range_pct}%
VWAP: {vwap_position}, distance {distance_to_vwap}% | Trend 5m: {trend_5m}, 15m: {trend_15m}
ATR (hourly): {atr_hourly}
STUMPY (similar historical patterns): win_rate {stumpy_win_rate}%, average return {stumpy_avg_return}% (n={stumpy_n})
News: {news_sentiment}, catalysts {news_catalysts_n}, risk flags {news_risk_flags_n}

After the analysis above — give YOUR independent verdict, STRICTLY in this format,
each field on a new line:
DIRECTION: <exactly one of: LONG / SHORT / NO_TRADE — your own decision>
CONFIDENCE: <a number from 0 to 100>
TAIL_RISK: <yes / no — is there reason to expect a sharp move (either way) right now: upcoming earnings, a known catalyst, regulatory risk, a lawsuit, anything specific to this company>
TAIL_WHY: <if yes — briefly what the risk is; if no — a dash>
REASONING: <1-3 sentences, based on the analysis above, not only on technicals>"""

FIELD_RE = {
    "deepseek_direction": re.compile(r"DIRECTION\s*:\s*(.+)", re.I),
    "confidence": re.compile(r"CONFIDENCE\s*:\s*(\d+)", re.I),
    "tail_risk": re.compile(r"TAIL_RISK\s*:\s*(.+)", re.I),
    "tail_risk_why": re.compile(r"TAIL_WHY\s*:\s*(.+)", re.I),
    "reasoning": re.compile(r"REASONING\s*:\s*(.+)", re.I),
}


def _fmt(x, suffix=""):
    return "no data" if x is None else "%s%s" % (x, suffix)


def _build_prompt(ticker, trader_plan):
    """Features only — nothing from the bot's direction/scores/entry/stop/target
    gets in here (see the PROMPT_TMPL docstring — on principle)."""
    feat = trader_plan.get("features") or {}
    return PROMPT_TMPL.format(
        ticker=ticker,
        price=_fmt(feat.get("price")),
        rsi=_fmt(feat.get("rsi")),
        macd=_fmt(feat.get("macd")), macd_signal=_fmt(feat.get("macd_signal")),
        bb_lower=_fmt(feat.get("bb_lower")), bb_upper=_fmt(feat.get("bb_upper")),
        volume_ratio=round(feat.get("volume_ratio") or 0, 2),
        position_in_range_pct=round(feat.get("position_in_range_pct") or 0, 1),
        vwap_position=_fmt(feat.get("vwap_position")),
        distance_to_vwap=_fmt(feat.get("distance_to_vwap")),
        trend_5m=_fmt(feat.get("trend_5m")), trend_15m=_fmt(feat.get("trend_15m")),
        atr_hourly=_fmt(feat.get("atr_hourly")),
        stumpy_win_rate=_fmt(feat.get("stumpy_win_rate")),
        stumpy_avg_return=_fmt(feat.get("stumpy_avg_return")),
        stumpy_n=_fmt(feat.get("stumpy_n")),
        news_sentiment=feat.get("news_sentiment", "no data"),
        news_catalysts_n=feat.get("news_catalysts_n", 0),
        news_risk_flags_n=feat.get("news_risk_flags_n", 0))


def _normalize_direction(raw):
    """LONG/SHORT/NO_TRADE from the free text of the answer — the model sometimes writes
    'Long', 'go long', 'no trade' etc. instead of exactly one of the three words."""
    t = (raw or "").strip().upper()
    if "SHORT" in t or "SELL" in t:
        return "SHORT"
    if "LONG" in t or "BUY" in t:
        return "LONG"
    if "NO" in t or "NO ENTRY" in t or "STAY OUT" in t or "PASS" in t:
        return "NO_TRADE"
    return None


def _parse_verdict(text):
    out = {"raw": text}
    for field, rx in FIELD_RE.items():
        m = rx.search(text or "")
        out[field] = m.group(1).strip() if m else None
    if out.get("confidence") is not None:
        try:
            out["confidence"] = int(out["confidence"])
        except ValueError:
            pass
    out["deepseek_direction_norm"] = _normalize_direction(out.get("deepseek_direction"))
    return out


def _esc(x):
    return str(x or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _send_telegram(ticker, trader_plan, v):
    # The verdict is already in DEEPSEEK_VERDICTS_FILE; the number of verdicts per day
    # is shown in the 18:00 digest (TELEGRAM_SCHEDULED_QUIET in config.py).
    if TELEGRAM_SCHEDULED_QUIET or not (TELEGRAM_BOT_TOKEN and MY_CHAT_ID):
        return
    bot_dir = trader_plan.get("direction")
    ds_dir = v.get("deepseek_direction_norm")
    if ds_dir is None:
        mark = "❓"
    elif ds_dir == bot_dir:
        mark = "✅"
    else:
        mark = "⚠️"
    tail = "🔴 YES" if (v.get("tail_risk") or "").lower().startswith("y") else "—"
    lines = ["🧠 <b>DeepSeek verdict</b> — <b>%s</b> %s" % (_esc(ticker), mark),
            "Bot: %s | DeepSeek (independent): %s | Confidence: %s"
            % (_esc(bot_dir), _esc(v.get("deepseek_direction") or "?"), _esc(v.get("confidence"))),
            "Catastrophic tail risk: %s" % tail]
    if v.get("tail_risk_why") and v["tail_risk_why"] not in ("-", "—"):
        lines.append("  ↳ %s" % _esc(v["tail_risk_why"]))
    if v.get("reasoning"):
        lines.append("Reasoning: %s" % _esc(v["reasoning"]))
    try:
        requests.post("https://api.telegram.org/bot%s/sendMessage" % TELEGRAM_BOT_TOKEN,
                      data={"chat_id": MY_CHAT_ID, "text": "\n".join(lines),
                            "parse_mode": "HTML", "disable_web_page_preview": "true"},
                      timeout=20)
    except Exception as e:  # noqa: BLE001
        logger.warning("DeepSeek verdict: could not send to Telegram: %s", e)


def _send_error_telegram(ticker, reason):
    """Browser/Chrome/DeepSeek-side errors are always sent to Telegram as a separate
    message, regardless of the NO_TRADE filter above — this is not a verdict
    but a signal "there will be no verdict, check Chrome on the host"."""
    logger.warning("DeepSeek verdict not received for %s: %s", ticker, reason)
    # With TELEGRAM_SCHEDULED_QUIET — log only; the 18:00 digest will show that
    # there are fewer verdicts for the day than signals.
    if TELEGRAM_SCHEDULED_QUIET or not (TELEGRAM_BOT_TOKEN and MY_CHAT_ID):
        return
    text = "⚠️ <b>DeepSeek verdict not received</b> — <b>%s</b>\n%s" % (_esc(ticker), _esc(reason))
    try:
        requests.post("https://api.telegram.org/bot%s/sendMessage" % TELEGRAM_BOT_TOKEN,
                      data={"chat_id": MY_CHAT_ID, "text": text, "parse_mode": "HTML"},
                      timeout=20)
    except Exception as e:  # noqa: BLE001
        logger.warning("DeepSeek verdict: could not send the error message to Telegram: %s", e)


def _append_verdict_file(key, record):
    data = {}
    if os.path.isfile(DEEPSEEK_VERDICTS_FILE):
        try:
            with open(DEEPSEEK_VERDICTS_FILE, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    data[key] = record
    tmp = DEEPSEEK_VERDICTS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, DEEPSEEK_VERDICTS_FILE)


# ---------------------------------------------------------------- queue
# Queue limit — 2026-09-21, per the owner's decision to call DeepSeek on EVERY
# ticker evaluation, including NO_TRADE (an order of magnitude more frequent than LONG/SHORT:
# in the hourly autopilot there can be dozens of NO_TRADE per 5 taken signals).
# Without a limit the queue would grow unboundedly under overload and keep
# older (and less and less interesting) tasks in memory. When it is full,
# new tasks are dropped with a warning in the log — this is diagnostics,
# not a trading decision; losing some is less important than hanging for hours.
_QUEUE_MAXSIZE = 40
_queue: "queue.Queue" = queue.Queue(maxsize=_QUEUE_MAXSIZE)
_worker_lock = threading.Lock()
_worker_started = False


def _worker_loop():
    ctl = _ChromeDeepSeek()
    connected = False
    msgs_since_new_chat = 0
    while True:
        ticker, trader_plan = _queue.get()
        try:
            with tab_lock():
                if not connected:
                    connected = ctl.connect()
                    if not connected:
                        logger.warning("DeepSeek verdict: Chrome unavailable (%s) — signal for %s skipped",
                                       ctl.ready_reason, ticker)
                        _send_error_telegram(ticker, "Chrome unavailable: %s" % ctl.ready_reason)
                        continue
                    # The msgs_since_new_chat counter lives only in this process's memory and
                    # resets on every container restart — while the Chrome tab
                    # keeps living and accumulating history across restarts. Without this,
                    # with frequent bot redeploys the conversation would quietly grow past the
                    # degradation threshold (see ChromeDeepSeek.new_chat) without the counter
                    # even suspecting it. So on every new connection —
                    # an unconditionally fresh conversation, not trusting what is already on screen.
                    ctl.new_chat()
                    msgs_since_new_chat = 0
                if msgs_since_new_chat >= DEEPSEEK_NEW_CHAT_EVERY:
                    ctl.new_chat()
                    msgs_since_new_chat = 0
                reply = ctl.send_message(_build_prompt(ticker, trader_plan))
            msgs_since_new_chat += 1
            if reply is None:
                logger.warning("DeepSeek verdict: no answer for %s — treating the connection as dead", ticker)
                _send_error_telegram(ticker, "DeepSeek did not answer in %ds (the page was silent) — "
                                     "treating the connection as dead, will reconnect on the next signal"
                                     % DEEPSEEK_RESPONSE_TIMEOUT)
                ctl.close()
                connected = False
                continue
            v = _parse_verdict(reply)
            v["ticker"] = ticker
            v["bot_direction"] = trader_plan.get("direction")
            v["signal_timestamp"] = trader_plan.get("_signal_timestamp")
            v["asked_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
            v["complete"] = ctl.last_complete
            key = "%s|%s" % (ticker, v["signal_timestamp"] or v["asked_at"])
            _append_verdict_file(key, v)
            # NO_TRADE accumulates silently in the file (for counterfactual comparison statistics),
            # Telegram only gets real LONG/SHORT — otherwise on a busy day
            # the autopilot would flood the chat with dozens of messages about trades that never happened.
            if trader_plan.get("direction") in ("LONG", "SHORT"):
                _send_telegram(ticker, trader_plan, v)
            logger.info("DeepSeek verdict %s: bot=%s deepseek=%s (conf. %s, tail %s)",
                       ticker, trader_plan.get("direction"), v.get("deepseek_direction"),
                       v.get("confidence"), v.get("tail_risk"))
        except Exception as e:  # noqa: BLE001 — a background thread must not take anyone down
            logger.exception("DeepSeek verdict: unhandled error for %s: %s", ticker, e)
            _send_error_telegram(ticker, "unhandled error: %s" % e)
            try:
                ctl.close()
            except Exception:
                pass
            connected = False
        finally:
            _queue.task_done()


def request_verdict_background(ticker: str, trader_plan: dict) -> None:
    """Fire-and-forget: enqueues the task and returns immediately.
    Call ONLY after the main signal message has already been sent —
    this is diagnostics, not part of the trading decision."""
    global _worker_started
    if not DEEPSEEK_VERDICT_ENABLED:
        return
    trader_plan = dict(trader_plan)
    trader_plan["_signal_timestamp"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    with _worker_lock:
        if not _worker_started:
            threading.Thread(target=_worker_loop, name="deepseek-verdict", daemon=True).start()
            _worker_started = True
    try:
        _queue.put_nowait((ticker, trader_plan))
    except queue.Full:
        logger.warning("DeepSeek verdict: queue full (>%d) — %s skipped",
                       _QUEUE_MAXSIZE, ticker)


# ---------------------------------------------------------------- statistics
# A separate summary of accumulated live DeepSeek verdicts — at the
# owner's request (2026-09-23), shown TOGETHER with /stats but as a separate block: this is
# diagnostics of DEEPSEEK'S QUALITY, not a bot metric (see the module docstring —
# the verdict never affects the bot's decision, so its own
# accuracy/return is computed and shown in isolation).
#
# WHY MATCHING BY NEAREST TIME AND NOT BY THE EXACT timestamp.
# v["signal_timestamp"] (utils.deepseek_verdict.request_verdict_background)
# and the record in signal_history.json/no_trade_history.json (append_signal/
# append_no_trade) are TWO separate datetime.now() calls within the same
# synchronous request (first append_signal, then request_verdict_background
# right after), so the seconds almost always differ by a fraction of a second —
# exact string matching would drop almost all records. We take the nearest-in-time
# record of the same ticker within 30s — enough not to
# confuse it with another signal of the same ticker (signals for one ticker do not come
# more often than once every few minutes).
def _match_bot_record(ticker, signal_timestamp, bot_direction):
    from utils.signal_history import load_signal_history, load_no_trade_history
    try:
        ts = datetime.fromisoformat(signal_timestamp)
    except (ValueError, TypeError):
        return None
    history = (load_signal_history() if bot_direction in ("LONG", "SHORT")
              else load_no_trade_history())
    best, best_diff = None, None
    for r in history.get(ticker, []):
        try:
            rts = datetime.fromisoformat(r.get("timestamp"))
        except (ValueError, TypeError):
            continue
        diff = abs((rts - ts).total_seconds())
        if diff <= 30 and (best_diff is None or diff < best_diff):
            best, best_diff = r, diff
    return best


def _entry_outcome_from_bot_record(r, bot_direction):
    """(entry, outcome_price) of the trade/NO_TRADE the verdict is tied to —
    a common shape for both branches, so DeepSeek's return is then computed
    with one formula regardless of whether the bot traded the idea or rejected it.

    For NO_TRADE there is no "outcome_price" of its own — instead we
    reconstruct the 15:30 close from the bot's already computed counterfactual
    (cf_return_pct/cf_direction — utils.signal_history._resolve_no_trade_
    counterfactual): it is THE SAME actual closing price, while the sign of
    DeepSeek's return is computed separately from its own direction
    below (_return_pct), not inherited from the bot."""
    from utils.signal_history import EXCLUDED_OUTCOMES, NO_TRADE_OUTCOMES
    if bot_direction in ("LONG", "SHORT"):
        outcome = r.get("outcome")
        if not outcome or outcome in NO_TRADE_OUTCOMES + EXCLUDED_OUTCOMES:
            return None, None
        entry = r.get("effective_entry_price")
        if entry is None:
            entry = r.get("entry_price")
        return entry, r.get("outcome_price")
    cf_outcome = r.get("cf_outcome")
    if cf_outcome not in ("would_win", "would_lose"):
        return None, None
    entry = (r.get("features") or {}).get("price")
    cf_direction, cf_return = r.get("cf_direction"), r.get("cf_return_pct")
    if entry is None or cf_direction not in ("LONG", "SHORT") or cf_return is None:
        return None, None
    if cf_direction == "LONG":
        return entry, entry * (1 + cf_return / 100)
    return entry, entry * (1 - cf_return / 100)


def _verdict_return_pct(direction, entry, outcome_price):
    if entry is None or outcome_price is None or entry == 0 or direction not in ("LONG", "SHORT"):
        return None
    if direction == "LONG":
        return (outcome_price - entry) / entry * 100
    return (entry - outcome_price) / entry * 100


def _verdict_actual_direction(entry, outcome_price):
    if entry is None or outcome_price is None:
        return None
    if outcome_price > entry:
        return "LONG"
    if outcome_price < entry:
        return "SHORT"
    return None


def compute_deepseek_stats():
    """Summary of DEEPSEEK_VERDICTS_FILE: how many LONG/SHORT/NO_TRADE verdicts
    it gave, whether it called the direction right by the actual price (not by the bot's opinion), and what
    return a bet on its own choice would have brought — on those signals
    (including NO_TRADE rejected by the bot) where the outcome is already known."""
    if not os.path.isfile(DEEPSEEK_VERDICTS_FILE):
        return {"n": 0}
    try:
        with open(DEEPSEEK_VERDICTS_FILE, encoding="utf-8") as f:
            rows = list(json.load(f).values())
    except Exception:
        return {"n": 0}
    if not rows:
        return {"n": 0}

    dir_counts = {"LONG": 0, "SHORT": 0, "NO_TRADE": 0, "n/a": 0}
    graded, returns = [], []
    for v in rows:
        ds_dir = v.get("deepseek_direction_norm")
        dir_counts[ds_dir if ds_dir in dir_counts else "n/a"] += 1
        if ds_dir not in ("LONG", "SHORT"):
            continue
        bot_record = _match_bot_record(v.get("ticker"), v.get("signal_timestamp"),
                                       v.get("bot_direction"))
        if not bot_record:
            continue
        entry, outcome_price = _entry_outcome_from_bot_record(bot_record, v.get("bot_direction"))
        actual_dir = _verdict_actual_direction(entry, outcome_price)
        if actual_dir:
            graded.append(ds_dir == actual_dir)
        ret = _verdict_return_pct(ds_dir, entry, outcome_price)
        if ret is not None:
            returns.append(ret)

    from utils.signal_history import SIMULATED_POSITION_USD
    correct_n = sum(1 for g in graded if g)
    return {
        "n": len(rows),
        "dir_counts": dir_counts,
        "graded_n": len(graded),
        "correct_n": correct_n,
        "accuracy_pct": round(100.0 * correct_n / len(graded), 1) if graded else None,
        "returns_n": len(returns),
        "avg_return_pct": round(sum(returns) / len(returns), 2) if returns else None,
        "total_usd": (round(sum(SIMULATED_POSITION_USD * r / 100 for r in returns), 2)
                     if returns else None),
    }


def format_deepseek_stats(stats):
    if stats.get("n", 0) == 0:
        return "\n\n🧠 *DeepSeek verdict*: no data yet."
    dc = stats["dir_counts"]
    na_part = " | n/a %d" % dc["n/a"] if dc["n/a"] else ""
    lines = ["\n\n🧠 *DeepSeek verdict — independent statistics*",
            "Total verdicts: %d  (LONG %d | SHORT %d | NO_TRADE %d%s)"
            % (stats["n"], dc["LONG"], dc["SHORT"], dc["NO_TRADE"], na_part)]
    if stats["graded_n"]:
        lines.append(f"Called the direction right (by actual price, not the bot's opinion): "
                     f"{stats['correct_n']}/{stats['graded_n']} ({stats['accuracy_pct']}%)")
    else:
        lines.append("Called the direction right: no directional calls with a known outcome yet")
    if stats["returns_n"]:
        sign = "+" if stats["avg_return_pct"] >= 0 else ""
        usign = "+" if stats["total_usd"] >= 0 else ""
        lines.append(f"Hypothetical return of its own choice: "
                     f"{sign}{stats['avg_return_pct']}%/trade "
                     f"({usign}${stats['total_usd']} over {stats['returns_n']} signals at $100, "
                     f"including ideas the bot rejected as NO_TRADE)")
    else:
        lines.append("Return of its own choice: no resolved directional calls yet")
    return "\n".join(lines)
