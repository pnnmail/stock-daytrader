# -*- coding: utf-8 -*-
"""Real trade flow for a stock — replacing the LLM "Order Flow" with real data.

WHY THIS FILE. The old `agents/orderflow.py` reads neither the trade tape nor
quotes — it shows the model hourly OHLC candles and asks it to make up
`orderflow_bias`/`large_player_active`/`delta_divergence`. Candles physically
contain neither individual trades nor bid/ask, so that is not order flow but coherent
text based on price movement that is already visible in MACD/VWAP. Exactly this
momentum group (Order Flow+MACD+VWAP) on 169 signals with a counterfactual gave
36.8% hits at a 45% base rate — worse than a coin (see signals.py).

WHAT IS HERE INSTEAD. Alpaca Market Data (free tier, IEX feed —
one exchange, not the full consolidated book, but real trades and
quotes, not text). From them, buy/sell classification is computed from the
trade price relative to the quote (tick rule / quote rule), as in
`crypto_daytrader/utils/flow_features.py`.

ORDER BOOK DEPTH (bid/ask by level) IS NOT AVAILABLE on the free tier —
Alpaca only returns the latest NBBO(IEX) quote, not the full book. That is why
there are no `book_features` here, only trade flow. Real equity book depth
requires a paid feed (Polygon.io/Nasdaq TotalView/IBKR).

NONE OF THIS TAKES PART IN THE DECISION (see `utils/signals.py`). It is
an instrumentation snapshot for counterfactual measurement — following
crypto_daytrader: first measure predictive value on accumulated
data, then (with a separate SCORING_VERSION) decide whether to add it to the formula.

WITHOUT ALPACA KEYS THE MODULE SILENTLY RETURNS {} — the bot does not break for those who
have not set them up (see config.ALPACA_API_KEY_ID/ALPACA_API_SECRET_KEY).
"""
from datetime import datetime, timedelta, timezone

import requests

from config import (
    ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY, ALPACA_DATA_URL,
    REAL_FLOW_WINDOWS_MIN,
)

_session = requests.Session()
_MAX_PAGES = 20   # a liquid stock has several thousand IEX trades in 15 minutes
_TIMEOUT = 10


def _headers():
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY_ID,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET_KEY,
    }


def _configured() -> bool:
    return bool(ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY)


def _get(path: str, params: dict) -> dict:
    resp = _session.get(f"{ALPACA_DATA_URL}{path}", params=params,
                         headers=_headers(), timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _fetch_paginated(path: str, ticker: str, start: datetime, end: datetime,
                      data_key: str) -> list:
    rows, page_token = [], None
    params = {
        "symbols": ticker,
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "feed": "iex",
        "limit": 10000,
    }
    for _ in range(_MAX_PAGES):
        if page_token:
            params["page_token"] = page_token
        data = _get(path, params)
        rows.extend((data.get(data_key) or {}).get(ticker, []))
        page_token = data.get("next_page_token")
        if not page_token:
            break
    return rows


def _classify_trades(trades: list, quotes: list) -> list:
    """Assigns each trade a side (buy/sell) by the quote rule:
    price >= ask -> buy, price <= bid -> sell, otherwise by the previous price
    (the standard tick rule as a fallback, see Lee-Ready).

    merge_asof is needed exactly here: quotes and trades arrive as separate
    feeds, and each trade needs the quote IN EFFECT AT THAT MOMENT,
    not the nearest one by record number.
    """
    import pandas as pd

    if not trades:
        return []
    tdf = pd.DataFrame(trades)
    tdf["t"] = pd.to_datetime(tdf["t"], utc=True)
    tdf = tdf.sort_values("t")

    if quotes:
        qdf = pd.DataFrame(quotes)
        qdf["t"] = pd.to_datetime(qdf["t"], utc=True)
        qdf = qdf.sort_values("t")
        merged = pd.merge_asof(tdf, qdf[["t", "bp", "ap"]], on="t", direction="backward")
    else:
        merged = tdf
        merged["bp"] = None
        merged["ap"] = None

    sides = []
    prev_price = None
    for _, row in merged.iterrows():
        price = row["p"]
        bid, ask = row.get("bp"), row.get("ap")
        if ask and price >= ask:
            side = "buy"
        elif bid and price <= bid:
            side = "sell"
        elif prev_price is not None:
            side = "buy" if price >= prev_price else "sell"
        else:
            side = "unknown"
        sides.append(side)
        prev_price = price

    merged["side"] = sides
    return merged.to_dict("records")


def _window_stats(classified: list, minutes: int) -> dict:
    buy = sum(r["p"] * r["s"] for r in classified if r["side"] == "buy")
    sell = sum(r["p"] * r["s"] for r in classified if r["side"] == "sell")
    total = buy + sell
    return {
        "n_trades": len(classified),
        "buy_quote": round(buy, 2),
        "sell_quote": round(sell, 2),
        "delta_quote": round(buy - sell, 2),
        "delta_pct": round((buy - sell) / total, 4) if total else None,
        "window_min": minutes,
    }


def build_real_flow_features(ticker: str) -> dict:
    """Full real-flow block for the longest window in REAL_FLOW_WINDOWS_MIN;
    shorter windows are cut from it — one network request instead of N.

    Errors are not propagated: this is diagnostics, not a decision input (see the file
    header) — an Alpaca failure must not take down the main /now pipeline.
    """
    if not _configured():
        return {}

    try:
        longest = max(REAL_FLOW_WINDOWS_MIN)
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=longest)

        trades = _fetch_paginated("/v2/stocks/trades", ticker, start, end, "trades")
        if not trades:
            return {"source": "alpaca_iex", "n_trades": 0}

        quotes = _fetch_paginated("/v2/stocks/quotes", ticker, start, end, "quotes")
        classified = _classify_trades(trades, quotes)

        feats = {"source": "alpaca_iex"}
        import pandas as pd
        newest = max(pd.Timestamp(r["t"]) for r in classified)
        for m in REAL_FLOW_WINDOWS_MIN:
            cutoff = newest - pd.Timedelta(minutes=m)
            sub = [r for r in classified if pd.Timestamp(r["t"]) >= cutoff]
            if sub:
                feats[f"flow_{m}m"] = _window_stats(sub, m)
        return feats
    except Exception as e:
        return {"source": "alpaca_iex", "error": str(e)}
