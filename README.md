# stock-daytrader

A Telegram bot for US equity day trading that scans the whole market, scores
candidates with a deterministic Python formula, and uses a local LLM (Gemma via
Ollama) only to read news and write commentary. Every signal, and every
*rejected* signal, is logged and resolved against real 5-minute bars, so the
bot keeps measuring whether its own rules work.

**Short version of the results: they don't.** Across ~2,350 historical
replays, 254 live trades and a pre-registered control-group study, the score
does not separate good trades from bad ones. Details are in [Results](#results).

> Research project. It never places orders and has no broker connection.

---

## How it works

```
NASDAQ + NYSE/AMEX list (~7,000 stocks)
        │  batched yfinance downloads, rate-limit aware
        ▼
full_market_sweep ── deterministic screen, no LLM (seconds per batch)
        │  entry filter: R:R, volume, ATR, earnings within 3 days
        ▼
prefilter + ranking ── /refilter, /find (relaxation ladder)
        │
        ▼
full analysis per ticker (/now, /scan, /queue, hourly /autopilot)
   ├─ Python: indicators, pivots, VWAP, liquidity levels, STUMPY analogues
   ├─ Python: factor scores → direction, entry / stop / targets, size
   ├─ LLM (Gemma 4 E4B): news summary, historical analogy, risks, commentary
   └─ diagnostics only: real trade flow (Alpaca/IEX), FINRA short volume,
      independent DeepSeek verdict (web UI via Chrome CDP)
        │
        ▼
signal_history.json / no_trade_history.json
        │  15:35 ET job: stop / target / forced close at 15:30
        ▼
/stats, /lab, daily digest
```

**Design rule:** Python makes every decision (direction, levels, position
size). The LLM only does what needs language. An earlier version let an LLM
invent "order flow" from hourly candles; it hit 36.8% at a 45% base rate and
was removed.

### Score formula (`utils/signals.py`, `SCORING_VERSION = v4_stumpy_flat2`)

| Factor | Range | Source |
|---|---|---|
| Momentum (MACD + VWAP agree) | ±1 | Python |
| Technical (RSI + Bollinger, contrarian) | ±2 | Python |
| News | ±2 | LLM sentiment × count of catalysts / risk flags |
| STUMPY historical analogues | ±2 | `stumpy.match` on hourly bars |

Entry threshold ±3, reachable maximum ±7. Each formula change gets a new
version tag, and statistics are never mixed across versions.

### Measurement discipline built into the code

- Rejected signals are stored as a **control group** with a counterfactual
  outcome, so "does the threshold mean anything?" can be answered.
- One observation per ticker per session (deduplication).
- Out-of-session signals are re-anchored to the next open. Gaps that make the
  trade impossible (`gap_invalidated`) or kill the R:R (`gap_no_edge`) are
  excluded, not counted as wins.
- Inside a bar the stop is assumed to fire before the target, which biases
  results against the system.
- Pre-market and late signals (after 15:00) are kept out of the intraday
  statistics.

---

## Results

### 1. Live signals (2026-08-10 … 2026-10-26)

Daily rule: one session, stop / target on 5-minute bars, forced close at 15:30.

| | Trades | Win rate | Avg / trade | Median | Sharpe / trade | Profit factor |
|---|---|---|---|---|---|---|
| All versions | 254 | 44.5% | −0.02% | −0.14% | −0.01 | 0.96 |
| Current formula (v4) | 217 | 43.8% | −0.04% | −0.17% | −0.02 | 0.94 |

Outcomes: 145 forced closes at 15:30, 85 stops, 24 first targets. $100 per
signal would have ended at **−$5.6** over the whole period.

Counterfactual for rejected signals (n = 1,765):

| Rejected because | n | Would have won | Avg return |
|---|---|---|---|
| score below threshold | 1,374 | 48.4% | −0.09% |
| entry filter (R:R, volume, ATR) | 391 | 50.4% | +0.14% |

The entry filter rejected trades that did slightly *better* than the ones it
took.

### 2. Pre-registered control-group study (1,208 records, 29 sessions)

Same measure for both groups: entry at the decision price, exit at 15:30,
minus SPY over the same window.

| Hypothesis | Result | Verdict |
|---|---|---|
| H1: passing the threshold beats rejected | +0.08% vs −0.14%, diff 95% CI [−0.05; +0.48] | **FAIL** |
| H2: total score ranks outcomes | daily IC +0.03, t = 0.5 | **FAIL** |
| H3: any single factor (momentum / technical / news / STUMPY) | all |t| < 1.3 | not significant |

### 3. Historical replay of the formula (`agents/historical_backtest.py`)

2,350 observations, 25 tickers, 3 years of daily bars, 3-day close-to-close
outcome. Summary in [results/historical_backtest_summary.json](results/historical_backtest_summary.json).

| | n | Hit rate | 95% Wilson CI |
|---|---|---|---|
| Base rate (price up) | 2,350 | 49.2% | 47.2–51.3% |
| Trades taken by the formula | 187 | 51.9% | 44.7–58.9% |
| Rejected (counterfactual) | 2,163 | 51.1% | 49.0–53.2% |
| Technical factor alone | 985 | 52.3% | 49.2–55.4% |
| STUMPY alone | 1,558 | 50.9% | 48.4–53.4% |

Ten follow-up checks ([results/ten_targeted_checks.json](results/ten_targeted_checks.json)):
expectancy +0.23% per trade before costs (longs −0.09%, shorts +0.57%);
excess vs SPY 51.0%; gap continuation, earnings proximity and FINRA short volume
all show no effect (short-volume correlation r = 0.02). **0 of 40 confidence
intervals exclude 50%**, while about 2 would be expected by chance alone.

### 4. Does an LLM verdict add anything?

- **Anonymised history** (300 snapshots, qwen3.5:4b, ticker hidden):
  Python 52.0% vs LLM 51.7%. When they disagreed, Python was right 34 times and
  the LLM 33. The LLM chose NO_TRADE 97.7% of the time. See
  [results/llm_verdict_experiment_summary.json](results/llm_verdict_experiment_summary.json).
- **Live DeepSeek verdicts** (622 verdicts, 150 directional calls with a known
  outcome): direction right 46%, −0.22% per trade. Its "tail risk" flag
  correlated with the size of the move, but after normalising by ATR the
  effect disappeared (p = 0.71–0.78). It was repeating volatility already
  present in the prompt.

### Conclusion

The bot is a working measurement instrument, and it measured a negative
result: neither the factor score, nor the thresholds, nor an LLM layer gives
an edge in intraday US equities on free data. The one stable pattern noticed
(stocks picked by the sweep lag SPY by ~0.2% by 15:30) is a property of
candidate selection, not of the score, and was not pre-registered.

---

## Commands

| Stage | Commands |
|---|---|
| Whole-market funnel | `/autoscan` → `/refilter` → `/queue`, `/find N`, `/autopilot N`, `/morning on` |
| Single ticker | `/now TICKER`, `/scan T1 T2 …`, `/go TICKER N`, `/status`, `/stop` |
| Quick screens (no LLM) | `/screen`, `/mass` |
| Raw lists | `/earnings N [M]`, `/insiders [days] [min_usd] [role]` |
| Validation | `/stats [TICKER] [7d\|30d\|today]`, `/lab [days]`, `/news [TICKER]` |

## Project layout

```
main.py                 bot entry point, scheduler (morning sweep, hourly autopilot, 15:35 resolution, digest)
config.py               models, thresholds, file paths, feature flags
agents/                 screener, scanner, trader, prompts, market overview,
                        filter_lab / replay_lab / historical_* research tools
data/                   yfinance fetcher, indicators, STUMPY finder, value checklist
utils/                  scoring, signal history & resolution, market hours, DeepSeek bridge,
                        Alpaca trade flow, FINRA short volume, Ollama client
telegram_bot/           command handlers, report formatting
knowledge/              PDF → rules distillation into ChromaDB (RAG; currently disabled in the pipeline)
results/                summary JSON of the historical experiments
```

## Running

```bash
pip install -r requirements.txt
cp .env.example .env    # fill in TELEGRAM_BOT_TOKEN and MY_CHAT_ID
ollama pull gemma4:e4b
python main.py
```

Or with Docker: `docker build -t stock-daytrader .` and mount `reports/`,
`signal_history.json`, `no_trade_history.json` and `state.json` as volumes.

Optional: `ALPACA_API_KEY_ID` / `ALPACA_API_SECRET_KEY` for real trade flow,
`SEC_USER_AGENT` (an email address, required by SEC EDGAR) for `/insiders`,
and a Chrome started with `start_chrome_debug.bat` for DeepSeek verdicts.

## Stack

Python 3.11 · python-telegram-bot · yfinance · pandas / numpy · stumpy ·
Ollama (Gemma 4 E4B, runs fully on a 6 GB RTX 3050) · ChromaDB · Docker ·
Chrome DevTools Protocol
