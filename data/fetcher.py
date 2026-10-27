import time
import requests
import yfinance as yf
import pandas as pd
import feedparser
from datetime import datetime, timedelta
from urllib.parse import quote
from bs4 import BeautifulSoup
from config import (
    MAX_NEWS, NEWS_TEXT_MAX_CHARS, MAX_LINE_LOG, EXTENDED_HOURS_ENABLED,
    MAX_WIRE_NEWS, NEWS_WIRE_LOOKBACK_DAYS,
)

# A stretched backoff for the requests that most often hit "Too Many Requests"
# from Yahoo (volume/52W in fetch_relative_context, value_investing) — especially
# after a heavy /autoscan that has been hammering Yahoo for half an hour before.
# It used to be 3 attempts with a fixed 5s pause (10s in total) — not enough
# to survive a real ban. Now 4 attempts, with growing pauses: 10-15-30s.
RATE_LIMIT_BACKOFF = [10, 15, 30]


def bars_with_volume(df):
    """
    Returns only bars that have volume filled in.

    IMPORTANT: this is NOT a filter for "garbage" bars. With prepost=True Yahoo returns
    real OHLC for the extended session but does NOT fill in volume — it is 0 there.
    E.g. AMZN's 16:00 bar after earnings: O=235.50 H=264.35 C=254.54 with
    volume=0 — the move is real, the volume just isn't provided.

    So such bars must NOT be dropped from the price series (otherwise all
    out-of-session trading is lost, which is why pre-market was enabled), but volume
    statistics must NOT be computed from them either — zeros understate the average several times and break
    engulfing detection and volume_ratio.
    """
    if df is None or df.empty or "Volume" not in df.columns:
        return df
    return df[df["Volume"] > 0]


def fetch_price_data(ticker):
    """
    Returns hourly data for indicators AND daily data for volume.

    prepost=EXTENDED_HOURS_ENABLED — if pre-market/after-hours count as tradable,
    data for those hours must be included: otherwise at 07:00 the "current price"
    would be yesterday's close, and the plan would be built on a price at which one
    can no longer enter.
    """
    hourly = yf.download(ticker, period="5d", interval="1h", progress=False,
                         prepost=EXTENDED_HOURS_ENABLED)
    daily = yf.download(ticker, period="5d", interval="1d", progress=False)
    
    if hourly.empty: 
        hourly = yf.download(ticker, period="1mo", interval="1d", progress=False)
    if hourly.empty: 
        hourly = yf.Ticker(ticker).history(period="1mo")
    
    if isinstance(hourly.columns, pd.MultiIndex):
        hourly.columns = hourly.columns.get_level_values(0)
    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns = daily.columns.get_level_values(0)

    return hourly, daily


def fetch_previous_day_levels(ticker):
    """Gets the previous trading day's High/Low and stop levels."""
    try:
        df = yf.download(ticker, period="2d", interval="1d", progress=False)
        if not df.empty and len(df) >= 2:
            prev_day = df.iloc[-2]
            prev_high = float(prev_day["High"].values[0])  # ✅ .values[0]
            prev_low = float(prev_day["Low"].values[0])    # ✅ .values[0]
            stops_above = [round(prev_high * 1.005, 2), round(prev_high * 1.01, 2)] if prev_high else []
            stops_below = [round(prev_low * 0.995, 2), round(prev_low * 0.99, 2)] if prev_low else []
            return {
                "prev_high": prev_high,
                "prev_low": prev_low,
                "stops_above": stops_above,
                "stops_below": stops_below
            }
        
        info = yf.Ticker(ticker).info
        prev_high = info.get('previousClose')
        if prev_high:
            prev_high = float(prev_high)
            prev_low = round(prev_high * 0.98, 2)
            return {
                "prev_high": prev_high,
                "prev_low": prev_low,
                "stops_above": [round(prev_high * 1.005, 2), round(prev_high * 1.01, 2)],
                "stops_below": [round(prev_low * 0.995, 2), round(prev_low * 0.99, 2)]
            }
        
        return {"prev_high": None, "prev_low": None, "stops_above": [], "stops_below": []}
    except Exception as e:
        print(f"   ⚠️ Error getting prev day for {ticker}: {e}")
        return {"prev_high": None, "prev_low": None, "stops_above": [], "stops_below": []}

def fetch_news(ticker: str, limit: int = MAX_NEWS) -> list:
    print(f"\n📰 NEWS SEARCH FOR {ticker}")
    feed_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    feed = feedparser.parse(feed_url)
    print(f"   Found in RSS: {len(feed.entries)} entries")
    news_list = []
    seen_texts = set()
    
    for i, entry in enumerate(feed.entries[:limit]):
        title = entry.get("title", "Untitled")
        link = entry.get("link", "")
        summary = entry.get("description", entry.get("summary", ""))
        clean_summary = BeautifulSoup(summary, "html.parser").get_text(strip=True) if summary else ""
        print(f"   [{i+1}] {title[:MAX_LINE_LOG]}")
        
        full_text = ""
        try:
            resp = requests.get(link, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer", "aside", "iframe"]):
                tag.decompose()
            article_body = soup.find("div", class_="caas-body") or soup.find("article") or soup.find("div", attrs={"class": lambda x: x and "article" in str(x).lower()})
            if article_body:
                full_text = article_body.get_text(separator="\n", strip=True)
            else:
                full_text = "\n".join([p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 30])
            if full_text:
                full_text = ' '.join([l.strip() for l in full_text.split('\n') if len(l.strip()) > 50])
        except:
            pass
        
        text_to_analyze = full_text[:NEWS_TEXT_MAX_CHARS] if full_text else clean_summary[:NEWS_TEXT_MAX_CHARS]
        
        if text_to_analyze.startswith(title):
            text_to_analyze = text_to_analyze[len(title):].strip()
        
        import re
        text_to_analyze = re.sub(r'<[^>]+>', '', text_to_analyze)
        text_to_analyze = re.sub(r'\s{2,}', ' ', text_to_analyze)
        
        text_key = text_to_analyze[:100].strip()
        if text_key in seen_texts:
            print(f"   ⚠️ Skipped a duplicate (same text): {title[:80]}...")
            continue
        seen_texts.add(text_key)
        
        has_full = bool(full_text and len(full_text) > 200)
        news_list.append({
            "title": title,
            "text": text_to_analyze,
            "has_full_text": has_full
        })
        
        if has_full:
            print(f"      ✅ Full text retrieved ({len(full_text)} chars)")
        else:
            print(f"      ⚠️ RSS description only ({len(clean_summary)} chars)")
    
    yahoo_count = len(news_list)

    # Extra source: Yahoo RSS per ticker misses some wire press releases
    # (PRNewswire/GlobeNewswire/Newsfile — e.g. "a law firm has opened an
    # investigation after the stock fell") — they don't always make it into
    # the ticker's headline feed, although an ordinary broad
    # search finds them. Google News indexes/mirrors them much more fully.
    wire_news = fetch_wire_news(ticker)
    seen_titles = {n['title'].lower().strip() for n in news_list}
    wire_added = 0
    for w in wire_news:
        key = w['title'].lower().strip()
        # A rough dedup by headline (not by text — wire entries have no text):
        # an exact match or one headline contained in another.
        if any(key == t or key in t or t in key for t in seen_titles):
            continue
        news_list.append(w)
        seen_titles.add(key)
        wire_added += 1

    full_count = sum(1 for n in news_list if n['has_full_text'])
    print(
        f"   TOTAL: {len(news_list)} news items (Yahoo: {yahoo_count}, "
        f"Google News extra: {wire_added}; with full text: {full_count})\n"
    )
    return news_list


def fetch_wire_news(ticker: str, limit: int = MAX_WIRE_NEWS, days: int = NEWS_WIRE_LOOKBACK_DAYS) -> list:
    """
    An extra news source — Google News RSS per ticker (see fetch_news).

    Unlike fetch_news(), there is NO full article text here: Google News
    links are redirects via news.google.com, not direct publisher
    URLs, and pulling the article from there without extra hops and a risk of blocking
    is not possible. We take what the RSS entry itself gives — the headline and the source.
    For headlines like "Law Offices of X Investigates Y (TICK)" this
    is enough for the fact of publication to get into the analysis.
    """
    query = quote(f"{ticker} stock when:{days}d")
    feed_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    print(f"\n📰 EXTRA SOURCE (Google News RSS) FOR {ticker}")
    try:
        feed = feedparser.parse(feed_url)
    except Exception as e:
        print(f"   ⚠️ Google News RSS error: {e}")
        return []
    print(f"   Found in Google News: {len(feed.entries)} entries")

    news_list = []
    seen_titles = set()
    for entry in feed.entries:
        title = entry.get("title", "")
        # Google News appends " - Publisher" to the end of the headline — the source
        # comes as a separate field anyway, no need to duplicate it in the text.
        if " - " in title:
            title = title.rsplit(" - ", 1)[0].strip()
        if not title:
            continue
        key = title.lower().strip()
        if key in seen_titles:
            continue
        seen_titles.add(key)

        source = entry.get("source")
        source_name = source.get("title", "") if isinstance(source, dict) else ""

        news_list.append({
            "title": title,
            "text": f"(source: {source_name})" if source_name else "",
            "has_full_text": False,
        })
        if len(news_list) >= limit:
            break

    print(f"   TOTAL (Google News): {len(news_list)} entries\n")
    return news_list


def fetch_earnings_info(ticker):
    try:
        cal = yf.Ticker(ticker).calendar
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if not dates:
            return {"earnings_date": None, "days_until": None}
        ed = dates[0] if isinstance(dates, list) else dates
        if hasattr(ed, "date"):
            ed = ed.date()
        days = (ed - datetime.now().date()).days
        return {"earnings_date": str(ed), "days_until": days}
    except:
        return {"earnings_date": None, "days_until": None}


def format_earnings_info(e):
    if not e.get("earnings_date"):
        return "unknown"
    flag = " ⚠️ SOON" if e["days_until"] is not None and 0 <= e["days_until"] <= 21 else ""
    return f"{e['earnings_date']} ({e['days_until']} d.){flag}"


def fetch_relative_context(ticker, max_retries=None):
    """Gets context accounting for trading time, for a correct volume estimate."""
    vol_ratio = None
    h52 = l52 = pos = None
    max_retries = max_retries or len(RATE_LIMIT_BACKOFF) + 1

    for attempt in range(max_retries):
        try:
            hist = yf.download(ticker, period="3mo", interval="1d", progress=False)
            if not hist.empty:
                vol = hist["Volume"]
                if isinstance(vol, pd.DataFrame):
                    vol = vol.iloc[:, 0]
                
                avg_vol = float(vol.iloc[:-1].tail(20).mean())
                cur_vol = float(vol.iloc[-1])
                
                now = datetime.now()
                market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
                market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
                
                if now.weekday() < 5 and market_open <= now <= market_close:
                    minutes_passed = (now - market_open).total_seconds() / 60
                    total_minutes = 6.5 * 60
                    day_fraction = minutes_passed / total_minutes
                    
                    if day_fraction > 0.1:
                        estimated_full_day_vol = cur_vol / day_fraction
                        vol_ratio = estimated_full_day_vol / avg_vol if avg_vol else None
                        print(f"   🕐 {minutes_passed:.0f} of 390 min elapsed ({day_fraction:.0%} of the day)")
                        print(f"   📊 Volume now: {cur_vol:,.0f} → Full-day estimate: {estimated_full_day_vol:,.0f} (average: {avg_vol:,.0f})")
                    else:
                        vol_ratio = cur_vol / avg_vol if avg_vol else None
                        print(f"   🕐 Only {minutes_passed:.0f} min elapsed, volume for the day = {cur_vol:,.0f}")
                else:
                    vol_ratio = cur_vol / avg_vol if avg_vol else None
                
                break
        except Exception as e:
            print(f"   ⚠️ Volume error (attempt {attempt+1}/{max_retries}): {str(e)[:2000]}")
            if attempt < max_retries - 1:
                time.sleep(RATE_LIMIT_BACKOFF[attempt])

    for attempt in range(max_retries):
        try:
            info = yf.Ticker(ticker).info
            if info and info.get('symbol') == ticker:
                h52 = info.get("fiftyTwoWeekHigh")
                l52 = info.get("fiftyTwoWeekLow")
                if not h52 or not l52:
                    try:
                        hist_52w = yf.download(ticker, period="1y", interval="1d", progress=False)
                        if not hist_52w.empty:
                            h52 = float(hist_52w['High'].max())
                            l52 = float(hist_52w['Low'].min())
                            print(f"   📊 52W from history: High=${h52:.2f}, Low=${l52:.2f}")
                    except:
                        pass
                p = info.get("currentPrice")
                pos = ((p-l52)/(h52-l52)*100) if h52 and l52 and p and h52 != l52 else None
                break
        except Exception as e:
            print(f"   ⚠️ 52W error (attempt {attempt+1}/{max_retries}): {str(e)[:2000]}")
            if attempt < max_retries - 1:
                time.sleep(RATE_LIMIT_BACKOFF[attempt])

    return {
        "volume_ratio": vol_ratio,
        "high_52w": h52,
        "low_52w": l52,
        "position_in_range_pct": pos
    }


def format_relative_context(r):
    vl = "no data"
    if r.get("volume_ratio") is not None:
        d = (r["volume_ratio"]-1)*100
        vl = f"{r['volume_ratio']:.2f}x ({abs(d):.0f}% {'above' if d>0 else 'below'} normal)"
    rl = f"{r['position_in_range_pct']:.0f}% (low: {r['low_52w']} / high: {r['high_52w']})" if r.get("position_in_range_pct") is not None else "no data"
    return f"- Volume: {vl}\n- 52W position: {rl}"


def fetch_extended_history(ticker, max_retries=3):
    hourly = daily = None
    for attempt in range(max_retries):
        try:
            hourly = yf.download(ticker, period="5d", interval="1h", progress=False,
                                 prepost=EXTENDED_HOURS_ENABLED)
            if not hourly.empty:
                break
        except:
            time.sleep(5)
    for attempt in range(max_retries):
        try:
            daily = yf.download(ticker, period="1y", interval="1d", progress=False)
            if not daily.empty:
                break
        except:
            time.sleep(5)
    if hourly is None:
        hourly = pd.DataFrame()
    if daily is None:
        daily = pd.DataFrame()
    for df in (hourly, daily):
        if not df.empty and isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    return {"hourly": hourly, "daily": daily}


def fetch_signal_day_ohlc(ticker, signal_dt, buffer_days=6):
    """
    Loads hourly candles starting from the signal day and several days ahead
    (a margin for weekends/holidays) to check whether the stop/target fired
    the same day or the next trading day.
    """
    start = signal_dt.date()
    end = start + timedelta(days=buffer_days)
    try:
        df = yf.download(
            ticker,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval="1h",
            progress=False,
            # The stop/target may be touched in pre-market or after-hours — if
            # we trade in those hours, outcome resolution must account for them
            prepost=EXTENDED_HOURS_ENABLED
        )
        if df.empty:
            return df
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        print(f"   ⚠️ Error loading hourly data to check the signal for {ticker}: {str(e)[:150]}")
        return pd.DataFrame()


def fetch_signal_day_intraday(ticker, signal_dt):
    """
    Five-minute candles of THE SIGNAL DAY — for daily outcome resolution (one session,
    forced close at 15:30).

    WHY NOT HOURLY, as in fetch_signal_day_ohlc. Between 09:30 and 15:30
    there are six hourly bars, and a signal at 13:00 has only two. Within one bar
    the order "stop before target" is an assumption, and at that scale it
    decides the outcome of almost every trade. With five-minute bars there are 72, and
    the assumption stops being decisive.

    IMPORTANT ABOUT DATA LIFETIME: yfinance returns 5-minute bars only for the last
    ~60 days. So resolving an outcome retroactively a quarter later is NOT possible —
    the verdict must be saved into the record itself on the signal day, not
    recomputed on demand, as the lazy scheme does for hourly bars.
    """
    start = signal_dt.date()
    try:
        df = yf.download(
            ticker,
            start=start.isoformat(),
            end=(start + timedelta(days=1)).isoformat(),
            interval="5m",
            progress=False,
            # The daily rule lives inside the regular session, but prepost is kept
            # consistent with the rest of the pipeline: extra bars are filtered out
            # by time during resolution, while inconsistency between sources would not be.
            prepost=EXTENDED_HOURS_ENABLED,
        )
        if df.empty:
            return df
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        print(f"   ⚠️ Error loading 5-minute data for {ticker}: {str(e)[:150]}")
        return pd.DataFrame()


def get_company_name(ticker):
    """
    Gets the company name for a ticker via Yahoo Finance.
    Cached in a dict so as not to make a request every time.
    """
    if not hasattr(get_company_name, 'cache'):
        get_company_name.cache = {}
    
    if ticker in get_company_name.cache:
        return get_company_name.cache[ticker]
    
    try:
        info = yf.Ticker(ticker).info
        name = info.get('shortName') or info.get('longName') or ''
        if name:
            import re
            name = re.sub(r'\.com,?', '', name, flags=re.IGNORECASE)
            name = re.sub(r'\b(Inc\.?|Corp\.?|Ltd\.?|LLC|PLC|S\.A\.|AG|SE)\b', '', name, flags=re.IGNORECASE)
            name = re.sub(r'[,\-].*', '', name)
            name = name.strip()
            
            words = name.split()
            for word in words:
                clean_word = re.sub(r'[^a-zA-Z]', '', word).lower()
                if len(clean_word) > 3:
                    get_company_name.cache[ticker] = clean_word
                    print(f"   📊 Company {ticker}: {clean_word} (from '{name}')")
                    return clean_word
            
            first_word = words[0].lower() if words else name.lower()
            first_word = re.sub(r'[^a-zA-Z]', '', first_word)
            get_company_name.cache[ticker] = first_word
            print(f"   📊 Company {ticker}: {first_word}")
            return first_word
    except Exception as e:
        print(f"   ⚠️ Error getting the name for {ticker}: {str(e)[:80]}")
    
    get_company_name.cache[ticker] = ''
    return ''