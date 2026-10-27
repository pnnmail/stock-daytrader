import os

# ============================================================
# OLLAMA
# ============================================================
OLLAMA_URL = "http://ollama:11434/api/generate"

# ============================================================
# MODELS
# ============================================================
# gemma4:e4b is the only official Gemma 4 E4B build that fits ENTIRELY into VRAM on an RTX 3050 (6 GB):
# measured 3.29 GB, 100% on GPU. The file size (9.6 GB) is
# misleading here — the E4B architecture keeps only the effective part of the weights resident.
# For comparison (measured under real load): bartowski Q6_K — 50% on GPU, qwen2.5:14b — 39%,
# i.e. half the model spills onto the CPU and calls slow down 2-3x.
# One model for all roles = always one LLM + an embedder in memory, no reloads in the pipeline.
JUNIOR_MODEL = NEWS_MODEL = EXTRACTOR_MODEL = TRADER_MODEL = LIBRARIAN_MODEL = "qwen3.5:4b"
DISTILLER_MODEL = "qwen3.5:4b"

# Model families with a "hybrid" reasoning mode (Qwen3 and its descendants,
# including qwen3.5), where thinking is ON BY DEFAULT. Without an explicit
# think=False in the Ollama request such a model goes into a <think> block and with
# structured output (json_schema) returns an EMPTY "response" — not
# broken JSON but literally an empty string (verified live on qwen3.5:4b,
# TRADER_SCHEMA). gemma4:e4b, currently used in all
# roles, does not have this problem: over a series of test calls it never fell
# into an empty answer without think, although Ollama formally lists the
# "thinking" capability for it too — so the capability itself cannot tell them apart,
# what differs is the default behaviour of the particular family.
#
# call_ollama/call_ollama_json (utils/ollama.py) check this list by
# model-name PREFIX and add think=False themselves — so switching
# a role to a model from another "thinking" family is done HERE, without edits
# in agents/*.py or utils/ollama.py. The list is deliberately narrow (only what
# was actually observed) rather than broad "just in case" — a false
# match on a model that was not going to think is harmless (it won't
# start thinking), but missing a model that really thinks reproduces the
# empty-answer bug again.
THINKING_MODEL_PREFIXES = ("qwen3",)
# ============================================================
# TELEGRAM
# ============================================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
MY_CHAT_ID = os.environ.get("MY_CHAT_ID", "")
# Owner's decision 2026-10-09: scheduled tasks (morning list, hourly
# autopilot, the 15:35 review, DeepSeek verdicts) do NOT post to Telegram — log
# only; data accumulates as before. The only scheduled message is the daily
# digest at DAILY_DIGEST_TIME (ET). Replies to your own commands (/scan, /now,
# /morning now ...) arrive as before. False restores the old behaviour.
TELEGRAM_SCHEDULED_QUIET = True
DAILY_DIGEST_TIME = (18, 0)
STATE_FILE = "/app/state.json"
SIGNAL_HISTORY_FILE = "/app/signal_history.json"
# Rejected signals (NO_TRADE) are the CONTROL GROUP. A separate file rather than
# shared with signal_history: NO_TRADE is an order of magnitude more frequent than trades (a mass
# scan consists almost entirely of them), and in a shared store the per-ticker limit would push out
# the real signal history within a couple of days. Without this group the question "does the entry
# threshold distinguish anything at all?" is unanswerable in principle: there is nothing to compare
# score=4 outcomes with, because score=2 was never recorded.
NO_TRADE_HISTORY_FILE = "/app/no_trade_history.json"
AUTOSCAN_CANDIDATES_FILE = "/app/reports/autoscan_candidates.json"
AUTOSCAN_RAW_RESULTS_FILE = "/app/reports/autoscan_raw_results.json"
# Sweep archive: AUTOSCAN_RAW_RESULTS_FILE is overwritten by every /autoscan,
# so history does not accumulate and there is nothing to measure the filter on. A copy of
# each sweep with the date in its name goes here — this is the material for agents/filter_lab.py.
SWEEP_ARCHIVE_DIR = "/app/reports/sweeps"
# Quote cache for filter_lab measurements (per ticker, like price_cache in agent3).
FILTER_LAB_PRICE_CACHE = "/app/reports/lab_price_cache.json"
# Earnings date cache (agents/earnings_calendar.py). Lives in reports/ because
# that directory is mounted — otherwise the cache would be wiped on every rebuild,
# and it takes about 40 minutes to build.
EARNINGS_CALENDAR_CACHE = "/app/reports/earnings_calendar.json"
# Computed earnings events for the event study (agents/earnings_event_study.py):
# one compact record per event, raw quotes are not stored.
EARNINGS_EVENTS_FILE = "/app/reports/earnings_events.json"
# Historical bar cache for sweep replay (agents/replay_lab.py). In batches of
# 25 tickers — so the download resumes after a rate limit instead of starting
# over (Yahoo throttles regularly, verified).
REPLAY_CACHE_DIR = "/app/reports/replay_cache"
# Settings for the autonomous morning loop (on/off, how many candidates, what
# time to run). A separate file in the mounted reports/, not in
# state.json: state.json is rewritten wholesale from several places (queue
# cursors, ticker snapshots), and a setting the user switched on once
# must not depend on races over that file. It also survives a rebuild.
MORNING_CONFIG_FILE = "/app/reports/morning_config.json"
# /find log: at which relaxation step N candidates were collected and how many
# there were at each step passed. A separate file, not shared with the signal
# history: this is a record of the MARKET STATE at search time, not of a trade.
# The point of accumulating it — the step number is a free measure of how stingy
# the market was that day; on the accumulated series we can check whether it is related to
# outcomes, and also the discarded regime hypothesis (see the decision that
# regime switching lost to a coin on 37 sweeps).
FIND_LOG_FILE = "/app/reports/find_log.json"
# Raw Form 4 XML cache (agents/insider_scan.py) — the same logic as the
# other caches above: the mounted reports/, so filings already seen are not re-downloaded
# when date windows overlap between /insiders calls.
INSIDER_FORM4_CACHE_DIR = "/app/reports/form4_cache"

# Cache of daily FINRA short volume files (utils/short_volume.py) — one file
# for the whole market per day, cached in reports/ so it is not re-downloaded on
# every autopilot call within an hour.
SHORT_VOLUME_CACHE_DIR = "/app/reports/short_volume_cache"

# Result of the one-off check of the current formula (agents/historical_backtest.py)
# on historical daily data — a SEPARATE file from signal_history.json /
# no_trade_history.json. They must not be mixed: these are not the bot's real signals but
# a replay of the formula on past prices (no intraday bars and no news —
# see the disclaimer inside the file itself).
HISTORICAL_BACKTEST_FILE = "/app/reports/historical_backtest.json"
# Same idea, but for the "LLM gives a verdict" experiment on top of an anonymised
# sample from HISTORICAL_BACKTEST_FILE (agents/historical_backtest_llm.py).
HISTORICAL_BACKTEST_LLM_FILE = "/app/reports/historical_backtest_llm.json"
# Five targeted checks on top of the existing HISTORICAL_BACKTEST_FILE that do not
# repeat the expensive run (agents/historical_theories.py).
HISTORICAL_THEORIES_FILE = "/app/reports/historical_theories.json"

# ============================================================
# DEEPSEEK VERDICT (utils/deepseek_verdict.py) — 2026-09-21
# ============================================================
# A parallel web-DeepSeek verdict on live LONG/SHORT signals. NEVER
# affects scoring/direction — a background thread, the result goes as a separate
# Telegram message and into DEEPSEEK_VERDICTS_FILE. See the detailed rationale
# (why not on anonymised history but exactly this way) in the module docstring.
#
# Chrome with chat.deepseek.com logged in must be open on the HOST (not in
# the container) with --remote-debugging-port=9222. The container reaches it via
# host.docker.internal — but since 2026 Chrome blocks CDP requests with
# a Host header other than "localhost"/IP (DNS rebinding protection), and
# host.docker.internal is not one. Workaround: connect via
# host.docker.internal but EXPLICITLY set Host: localhost:9222 in HTTP and
# WebSocket requests — verified live on 2026-09-21 (full REST /json cycle +
# WS Runtime.evaluate against a real chat.deepseek.com tab on the host).
DEEPSEEK_VERDICT_ENABLED = True
DEEPSEEK_DEBUG_HOST = "host.docker.internal"
DEEPSEEK_DEBUG_PORT = 9222
# Live measurement 2026-09-21: at 180s almost every answer took 181-184s
# (hitting the ceiling); raised to 240 — and all 5 successful answers took EXACTLY
# 242s each, suspiciously identical. It looks like the "answer finished" detector
# (_wait_for_response: N polls of silence in a row) almost never really fires for long
# multi-paragraph answers and instead runs into the overall timeout,
# taking the last captured text as a fallback — i.e.
# the answer is actually ready earlier, it just isn't recognised. The content
# comes out complete and correct (verified on parsing), so this is not a
# correctness bug but an inefficiency + a risk of false cut-offs if the model did not
# finish exactly by the boundary. For now the ceiling was simply raised with a margin rather than fixing
# the detector itself — it does not block use, but next time in this
# file the JS silence detector should be sorted out instead of raising the number further.
# 2026-09-22: DeepThink and Search stay ON for the model (the
# owner's decision — deep analysis and web search are the reason we call
# DeepSeek at all, not extra load). The real cause of the series of "no answer" that
# day turned out to be not them but the fact that a background (inactive) Chrome tab
# held back committing the answer to the DOM for 300+ s — fixed in _force_active()
# (utils/deepseek_verdict.py). With this fix the delay is honest (real
# search+reasoning time), not artificial — we keep the ceiling with a margin.
DEEPSEEK_RESPONSE_TIMEOUT = 420
DEEPSEEK_VERDICTS_FILE = "/app/reports/deepseek_verdicts.json"
# Shared DeepSeek tab lock between the container and host scripts (see
# utils/deepseek_verdict.tab_lock) — in the mounted folder so both sides can see it.
DEEPSEEK_TAB_LOCK_FILE = "/app/reports/deepseek_tab.lock"
# Start a new conversation every this many messages — a long chat
# degrades (observed in the live DeepSeek UI in another project: after
# dozens of consecutive messages answers started arriving truncated).
DEEPSEEK_NEW_CHAT_EVERY = 5

# ============================================================
# ALPACA (real trade flow — utils/real_flow.py)
# ============================================================
# A free account at alpaca.markets (paper, no money and no brokerage account)
# gives keys for the Market Data API, IEX feed. Without keys real_flow silently
# returns {} — the bot works as before, just without this diagnostic
# factor (see the utils/real_flow.py header).
ALPACA_API_KEY_ID = os.environ.get("ALPACA_API_KEY_ID", "")
ALPACA_API_SECRET_KEY = os.environ.get("ALPACA_API_SECRET_KEY", "")
ALPACA_DATA_URL = "https://data.alpaca.markets"
# Windows in minutes. IEX is one exchange of many, with far fewer trades than
# the consolidated tape, so the windows are wider than in crypto_daytrader (1/5/15).
REAL_FLOW_WINDOWS_MIN = (5, 15, 30)

# ============================================================
# TRADING SESSION
# ============================================================
# Extended session (pre-market + after-hours). Brokers like Robinhood allow
# trading outside the regular 9:30-16:00 ET.
# IMPORTANT: this flag controls BOTH the trading window AND data loading (prepost in yfinance).
# They must not be split: if pre-market is treated as tradable but only the regular
# session is loaded, plans will be built on yesterday's close — exactly in the hours when
# gaps happen.
EXTENDED_HOURS_ENABLED = True
# Extended session bounds in ET. The US standard is 04:00-20:00.
# A particular broker's window may be narrower — put yours here.
PREMARKET_OPEN_HHMM = (4, 0)
AFTERHOURS_CLOSE_HHMM = (20, 0)

# ============================================================
# NEWS
# ============================================================
NEWS_TEXT_MAX_CHARS = 600
MAX_NEWS = 15

# Extra source — Google News RSS per ticker. The main source (Yahoo RSS per
# ticker) misses some wire press releases (PRNewswire/GlobeNewswire/
# Newsfile — e.g. law firm announcements of investigations after a drop),
# since they don't always make it into Yahoo's headline feed for a given ticker.
NEWS_WIRE_LOOKBACK_DAYS = 7
MAX_WIRE_NEWS = 10

# ============================================================
# RAG
# ============================================================
CHROMA_DB_PATH = "/app/chroma_db"
KNOWLEDGE_PATH = "/app/knowledge"
EMBED_MODEL = "nomic-embed-text"
RAG_N_RESULTS = 5
RAG_CHUNK_SIZE = 5000
RAG_MAX_CHARS = 2000
RAG_MAX_CHUNKS = 5000

# ============================================================
# LOGS
# ============================================================
MAX_LINE_LOG = 5000

# Decision thresholds.
#
# Lowered from ±4 to ±3 AT THE SAME TIME as compressing the momentum group in
# compute_factor_scores (see SCORING_VERSION in utils/signals.py). The reason is
# arithmetic, not taste: previously Order Flow, MACD and VWAP gave up to +3
# combined as three independent confirmations, now their agreement is worth +1.
# The reachable maximum fell from ±10 to ±7, and the ±4 threshold in its old place would cut
# twice as many signals — measured on 477 accumulated observations:
# |score|>=4 passed 8.0% under the old formula and 4.2% under the new one, |score|>=3 —
# 17.8%. The ±3 threshold brings the frequency roughly back to where it was.
#
# THRESHOLD AND FORMULA MUST CHANGE IN ONE COMMIT. Change only the formula and
# you get a quietly silent system that looks like "the market gives no
# setups" rather than a calibration error.
LONG_THRESHOLD = 3
SHORT_THRESHOLD = -3