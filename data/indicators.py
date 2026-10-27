import pandas as pd
import numpy as np
from datetime import timedelta
from utils.market_hours import is_within_market_hours, is_trading_holiday

# If today's open differs from the Pivot of the base (previous) session
# by more than this percentage, R1/R2/S1/S2 are computed from a level that is
# no longer relevant today (the classic case: an earnings gap), and are unusable as targets/stops.
PIVOT_GAP_THRESHOLD_PCT = 5.0


def compute_indicators(df_hourly, df_daily=None):
    """Computes indicators from hourly data, volume from daily."""
    if df_hourly.empty or len(df_hourly) < 2:
        return {"price": 0, "price_change": 0, "rsi": 50.0, "macd": 0, "macd_signal": 0, 
                "bb_upper": 0, "bb_lower": 0, "bb_position_pct": 50.0, "volume": 0}
    
    close = df_hourly["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.astype(float).squeeze()
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    
    # ✅ COMPUTE last_c AND prev_c RIGHT AWAY
    last_c = float(close.iloc[-1])
    prev_c = float(close.iloc[-2]) if len(close) > 1 else last_c
    chg = (last_c - prev_c)/prev_c*100 if prev_c else 0
    
    source = df_daily if (df_daily is not None and not df_daily.empty) else df_hourly
    volume_series = source["Volume"]
    if isinstance(volume_series, pd.DataFrame):
        volume_series = volume_series.iloc[:, 0]
    volume_series = volume_series.astype(float).squeeze()
    if isinstance(volume_series, pd.DataFrame):
        volume_series = volume_series.iloc[:, 0]
    # Take the last bar WITH volume, not just the last one. Yahoo returns an unclosed
    # daily bar (and bars of halted stocks) with volume=0: REPL on 07/30 showed
    # zero, and the report said "Volume: 0", although 14.3M had traded the day before.
    nonzero = volume_series[volume_series > 0]
    daily_volume = int(nonzero.iloc[-1]) if not nonzero.empty else 0
    
    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean()
    for i in range(14, len(avg_gain)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1]*13 + gain.iloc[i])/14
        avg_loss.iloc[i] = (avg_loss.iloc[i-1]*13 + loss.iloc[i])/14
    rsi = 100 - 100/(1 + avg_gain/avg_loss.replace(0, 1))
    rsi_v = float(rsi.iloc[-1])
    if pd.isna(rsi_v) or rsi_v <= 0 or rsi_v >= 100:
        rsi_v = 50.0
    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    bb_mid = close.rolling(20, min_periods=1).mean()
    bb_std = close.rolling(20, min_periods=1).std()
    bb_upper = float(bb_mid.iloc[-1] + 2*bb_std.iloc[-1])
    bb_lower = float(bb_mid.iloc[-1] - 2*bb_std.iloc[-1])
    
    # ✅ last_c IS NOW DEFINED
    if bb_upper != bb_lower:
        bb_position_pct = (last_c - bb_lower) / (bb_upper - bb_lower) * 100
    else:
        bb_position_pct = 50.0
    
    return {
        "price": round(last_c, 2),
        "price_change": round(chg, 2),
        "rsi": round(rsi_v, 1),
        "macd": round(float(macd.iloc[-1]), 3),
        "macd_signal": round(float(macd_signal.iloc[-1]), 3),
        "bb_upper": round(bb_upper, 2),
        "bb_lower": round(bb_lower, 2),
        "bb_position_pct": round(bb_position_pct, 1),
        "volume": daily_volume
    }

def check_data_quality(df_hourly):
    """
    Checks whether the last price can be trusted.

    Distinguishes normal out-of-session trading from corrupted data:
    extended-session bars have no volume filled in (that is normal), but
    THE LAST COMPLETED SESSION must have volume. If it does not, the stock
    was halted or the data was not updated, and the "current price"
    is not a trade price.

    Example: REPL on 07/30 got 4 bars with zero volume and a price of $12.51,
    while the last real trade was on 07/29 at $5.41 — analysing such
    a stock gave VWAP +139% and two-day-old pivots.

    Returns (ok: bool, warning: str).
    """
    if df_hourly is None or df_hourly.empty:
        return False, "no data"
    if "Volume" not in df_hourly.columns:
        return True, ""
    traded = df_hourly[df_hourly["Volume"] > 0]
    if traded.empty:
        return False, "not a single bar with volume — data unavailable"
    try:
        last_traded_day = traded.index[-1].date()
        last_bar_day = df_hourly.index[-1].date()
        # Count MISSED trading sessions, not calendar days. Otherwise at night and
        # in pre-market (when the last trade was "yesterday") literally every stock would
        # come under suspicion — that is the normal state, not a failure.
        skipped = 0
        day = last_traded_day + timedelta(days=1)
        while day < last_bar_day:
            if day.weekday() < 5 and not is_trading_holiday(day):
                skipped += 1
            day += timedelta(days=1)
    except (AttributeError, TypeError):
        return True, ""
    if skipped >= 1:
        last_price = float(df_hourly["Close"].iloc[-1])
        last_traded_price = float(traded["Close"].iloc[-1])
        drift = ((last_price - last_traded_price) / last_traded_price * 100
                 if last_traded_price else 0)
        return False, (
            f"the last trade was on {last_traded_day} at ${last_traded_price:.2f}, "
            f"while the current price ${last_price:.2f} ({drift:+.1f}%) comes from bars without "
            f"volume — probably a trading halt or incomplete data"
        )
    return True, ""


def compute_intraday_levels(df_hourly):
    if df_hourly.empty:
        return {}

    # Classic (floor-trader) pivots are computed from the PREVIOUS SESSION.
    # Previously High/Low were taken over the whole loaded period (5 days, ~85 bars),
    # which made the range several times wider than a day's, and S1/R1 drifted 7-10% from the price.
    # That is where too-wide stops and unreachable targets came from, which we
    # were catching with R:R filters. If splitting by day fails (too little data) —
    # we fall back to the old behaviour.
    c = float(df_hourly['Close'].iloc[-1])
    h = l = None
    try:
        dates = sorted({ts.date() for ts in df_hourly.index})
        # Pivots describe levels for the session we are PLANNING:
        #   market open   -> planning the current session -> base = the previous one
        #   market closed -> planning the next one        -> base = the last completed one
        # Without this fork, on the evening of the 30th the pivots would be computed from the 29th, although
        # the session of the 30th was already over, i.e. the data would be a day stale.
        base_idx = -2 if is_within_market_hours() else -1
        if len(dates) >= abs(base_idx):
            base_session = df_hourly[df_hourly.index.date == dates[base_idx]]
            if not base_session.empty:
                h = float(base_session['High'].max())
                l = float(base_session['Low'].min())
                # The classic formula uses the base session's close
                c = float(base_session['Close'].iloc[-1])
    except (AttributeError, TypeError):
        pass
    if h is None or l is None:
        h = float(df_hourly['High'].max())
        l = float(df_hourly['Low'].min())

    pp = (h + l + c) / 3
    r1 = 2 * pp - l
    s1 = 2 * pp - h
    r2 = pp + (h - l)
    s2 = pp - (h - l)
    r3 = h + 2 * (pp - l)
    s3 = l - 2 * (h - pp)

    # Gap detection: R1/R2/S1/S2 are built on the H/L/C of the BASE session. If today's
    # open is already far from this Pivot (an earnings gap etc.), these levels are not
    # a forecast but an extrapolation of a range the market does not recognise today.
    # There was a case: Pivot $94 (computed from a day that itself was -5.5% on
    # earnings), today's open ~$79 — R1/R2 drifted 20%+ from the price.
    # The reference is today's open; if the session has not started yet (market closed,
    # planning the next one), the last known price is used as a proxy.
    today_ref_price = None
    try:
        today_date = dates[-1]
        if today_date != dates[base_idx]:
            today_bars = df_hourly[df_hourly.index.date == today_date]
            if not today_bars.empty:
                today_ref_price = float(today_bars['Open'].iloc[0])
    except (NameError, AttributeError, TypeError, IndexError):
        pass
    if today_ref_price is None:
        today_ref_price = float(df_hourly['Close'].iloc[-1])
    pivot_gap_pct = abs(today_ref_price - pp) / pp * 100 if pp else 0
    pivot_reliable = pivot_gap_pct <= PIVOT_GAP_THRESHOLD_PCT

    # VWAP is INTRADAY: computed over the current session and reset every day.
    # Previously the cumsum ran over the whole loaded period (5 days), so it was
    # not VWAP but a weekly weighted average: the price was almost always far
    # from it, and the VWAP factor gave ±1 simply because of the weekly trend.
    df = df_hourly.copy()
    try:
        today = df.index[-1].date()
        session = df[df.index.date == today]
        if session.empty or session['Volume'].sum() <= 0:
            # The last date may have no volume (extended session only or
            # a trading halt). Fall back to the LAST SESSION WITH VOLUME, not to
            # the whole period: for REPL the full period gave VWAP $6.12 at a price of $12.52,
            # because it averaged the week instead of the trading day.
            traded = df[df['Volume'] > 0]
            if not traded.empty:
                last_traded_day = traded.index[-1].date()
                session = traded[traded.index.date == last_traded_day]
            else:
                session = df
    except (AttributeError, TypeError):
        session = df
    vol_sum = float(session['Volume'].sum())
    vwap = (float((session['Close'] * session['Volume']).sum()) / vol_sum
            if vol_sum > 0 else float(df_hourly['Close'].iloc[-1]))
    atr_h = compute_atr(df_hourly, period=14)
    prev_day = df_hourly.iloc[-2] if len(df_hourly) > 1 else None
    prev_high = float(prev_day['High']) if prev_day is not None else h
    prev_low = float(prev_day['Low']) if prev_day is not None else l
    prev_close = float(prev_day['Close']) if prev_day is not None else c
    first_2h = df_hourly.head(2)
    or_high = float(first_2h['High'].max()) if not first_2h.empty else h
    or_low = float(first_2h['Low'].min()) if not first_2h.empty else l
    
    # IMPORTANT: c above is the close of the PREVIOUS session (needed for the pivot formula).
    # All distances and comparisons with VWAP must be computed from the CURRENT price.
    price_now = float(df_hourly['Close'].iloc[-1])
    distance_to_vwap = (price_now - vwap) / vwap * 100 if vwap else 0
    distance_to_r1 = (r1 - price_now) / price_now * 100 if price_now else 0
    distance_to_s1 = (s1 - price_now) / price_now * 100 if price_now else 0

    if price_now > vwap * 1.001:
        vwap_position = "above"
    elif price_now < vwap * 0.999:
        vwap_position = "below"
    else:
        vwap_position = "at"
    
    def calc_trend(df_slice):
        if len(df_slice) < 2:
            return "neutral"
        first_close = float(df_slice['Close'].iloc[0])
        last_close = float(df_slice['Close'].iloc[-1])
        change = (last_close - first_close) / first_close * 100
        if change > 0.2:
            return "bullish"
        elif change < -0.2:
            return "bearish"
        else:
            return "neutral"
    
    last_5 = df_hourly.tail(5)
    last_15 = df_hourly.tail(15)
    trend_5m = calc_trend(last_5)
    trend_15m = calc_trend(last_15)
    
    return {
        'pivot': round(pp, 2), 'r1': round(r1, 2), 'r2': round(r2, 2), 'r3': round(r3, 2),
        's1': round(s1, 2), 's2': round(s2, 2), 's3': round(s3, 2),
        'pivot_reliable': pivot_reliable, 'pivot_gap_pct': round(pivot_gap_pct, 2),
        'vwap': round(vwap, 2), 'price_above_vwap': price_now > vwap,
        'atr_hourly': round(atr_h, 2) if atr_h else None,
        'prev_high': round(prev_high, 2), 'prev_low': round(prev_low, 2),
        'prev_close': round(prev_close, 2),
        'opening_range_high': round(or_high, 2), 'opening_range_low': round(or_low, 2),
        'vwap_position': vwap_position,
        'distance_to_vwap': round(distance_to_vwap, 2),
        'distance_to_r1': round(distance_to_r1, 2),
        'distance_to_s1': round(distance_to_s1, 2),
        'trend_5m': trend_5m,
        'trend_15m': trend_15m,
        # A direct divergence (bullish vs bearish, not just "one neutral") —
        # the same pattern seen earlier on NTLA. The flag is needed so the
        # trader/junior prompts can explicitly require mentioning it in reasoning, instead of
        # silently showing two numbers side by side that are easy to miss.
        'trend_divergence': {trend_5m, trend_15m} == {"bullish", "bearish"},
    }


def compute_volume_profile(df_hourly):
    if df_hourly.empty:
        return {}
    price_range = float(df_hourly['High'].max()) - float(df_hourly['Low'].min())
    if price_range == 0:
        return {'poc': float(df_hourly['Close'].iloc[-1]), 'value_area_high': None, 'value_area_low': None}
    step = price_range / 20
    levels = {}
    low = float(df_hourly['Low'].min())
    for i in range(20):
        level_low = low + i * step
        level_high = low + (i + 1) * step
        mask = (df_hourly['Close'] >= level_low) & (df_hourly['Close'] < level_high)
        levels[round((level_low + level_high) / 2, 2)] = float(df_hourly.loc[mask, 'Volume'].sum())
    if levels:
        poc = max(levels, key=levels.get)
        total_vol = sum(levels.values())
        sorted_levels = sorted(levels.items(), key=lambda x: abs(x[0] - poc))
        cum_vol = 0
        vah = val = poc
        for price, vol in sorted_levels:
            cum_vol += vol
            if cum_vol >= total_vol * 0.35:
                break
            vah = max(vah, price)
            val = min(val, price)
        return {'poc': round(poc, 2), 'value_area_high': round(vah, 2), 'value_area_low': round(val, 2), 'total_volume': int(total_vol)}
    return {'poc': float(df_hourly['Close'].iloc[-1]), 'value_area_high': None, 'value_area_low': None}


def detect_liquidity_levels(df_hourly):
    if df_hourly.empty:
        return {}
    highs = df_hourly['High'].values
    lows = df_hourly['Low'].values
    equal_highs = []
    for i in range(len(highs) - 1):
        if abs(highs[i] - highs[i+1]) / highs[i] < 0.001:
            equal_highs.append(round(float(highs[i]), 2))
    equal_lows = []
    for i in range(len(lows) - 1):
        if abs(lows[i] - lows[i+1]) / lows[i] < 0.001:
            equal_lows.append(round(float(lows[i]), 2))
    # Only clusters of equal extremes (where other traders' stops sit) — that is the
    # function's purpose. The prev_high/prev_low keys were REMOVED from here: they contained
    # highs[-2] — the high of the previous HOURLY bar, while the consumer
    # (calculate_entry_levels) treats prev_low as the previous DAY's low and
    # takes it as a stop candidate. Because of this /now placed the stop several times closer
    # than /scan on the same ticker. The real daily levels come from
    # fetch_previous_day_levels().
    return {
        'stops_above': list(set(equal_highs))[:3],
        'stops_below': list(set(equal_lows))[:3],
    }


def compute_atr(df, period=14):
    if df.empty or len(df) < period:
        return None
    h = df["High"].squeeze()
    l = df["Low"].squeeze()
    c = df["Close"].squeeze()
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return float(tr.rolling(period, min_periods=1).mean().iloc[-1])


def compute_moving_averages(daily):
    if daily.empty:
        return {}
    close = daily["Close"].squeeze()
    result = {"sma20": float(close.rolling(20, min_periods=1).mean().iloc[-1]), "sma50": float(close.rolling(50, min_periods=1).mean().iloc[-1])}
    if len(close) >= 200:
        result["sma200"] = float(close.rolling(200, min_periods=1).mean().iloc[-1])
    else:
        result["sma200"] = None
    return result


def compute_trade_metrics(trader_plan, intraday_levels, orderflow=None, junior_analysis=None):
    """
    Python computes ALL numeric metrics of the trade.
    """
    direction = trader_plan.get('direction', 'OUT')
    entry = trader_plan.get('entry_price', 0)
    stop = trader_plan.get('stop_loss', 0)
    target_1 = trader_plan.get('target_1', 0)
    target_2 = trader_plan.get('target_2', 0)
    position = trader_plan.get('position_size_pct', 2.0)
    atr = intraday_levels.get('atr_hourly', entry * 0.01) if intraday_levels else entry * 0.01
    
    warnings = []
    fixed = False
    
    if direction == 'LONG' and stop >= entry:
        stop = round(entry - atr, 2)
        trader_plan['stop_loss'] = stop
        warnings.append(f"Stop corrected: ${stop}")
        fixed = True
    elif direction == 'SHORT' and stop <= entry:
        stop = round(entry + atr, 2)
        trader_plan['stop_loss'] = stop
        warnings.append(f"Stop corrected: ${stop}")
        fixed = True
    
    if direction == 'LONG':
        if target_1 <= entry:
            target_1 = round(entry + atr * 1.5, 2)
            trader_plan['target_1'] = target_1
            warnings.append(f"Target 1 corrected: ${target_1}")
            fixed = True
        if target_2 <= target_1:
            target_2 = round(entry + atr * 3.0, 2)
            trader_plan['target_2'] = target_2
            warnings.append(f"Target 2 corrected: ${target_2}")
            fixed = True
    elif direction == 'SHORT':
        if target_1 >= entry:
            target_1 = round(entry - atr * 1.5, 2)
            trader_plan['target_1'] = target_1
            warnings.append(f"Target 1 corrected: ${target_1}")
            fixed = True
        if target_2 >= target_1:
            target_2 = round(entry - atr * 3.0, 2)
            trader_plan['target_2'] = target_2
            warnings.append(f"Target 2 corrected: ${target_2}")
            fixed = True
    
    if direction == 'LONG' and stop < entry:
        trader_plan['stop_pct'] = round((entry - stop) / entry * 100, 2)
        trader_plan['target_1_pct'] = round((target_1 - entry) / entry * 100, 2)
        trader_plan['target_2_pct'] = round((target_2 - entry) / entry * 100, 2)
        trader_plan['risk_reward_1'] = round((target_1 - entry) / (entry - stop), 1)
        trader_plan['risk_reward_2'] = round((target_2 - entry) / (entry - stop), 1)
    elif direction == 'SHORT' and stop > entry:
        trader_plan['stop_pct'] = round((stop - entry) / entry * 100, 2)
        trader_plan['target_1_pct'] = round((entry - target_1) / entry * 100, 2)
        trader_plan['target_2_pct'] = round((entry - target_2) / entry * 100, 2)
        trader_plan['risk_reward_1'] = round((entry - target_1) / (stop - entry), 1)
        trader_plan['risk_reward_2'] = round((entry - target_2) / (stop - entry), 1)
    else:
        trader_plan['stop_pct'] = 0
        trader_plan['target_1_pct'] = 0
        trader_plan['target_2_pct'] = 0
        trader_plan['risk_reward_1'] = 0
        trader_plan['risk_reward_2'] = 0
    
    confidence = 0.5
    
    if orderflow:
        of_bias = orderflow.get('orderflow_bias', '')
        if (direction == 'LONG' and of_bias == 'bullish') or (direction == 'SHORT' and of_bias == 'bearish'):
            confidence += 0.2
        elif (direction == 'LONG' and of_bias == 'bearish') or (direction == 'SHORT' and of_bias == 'bullish'):
            confidence -= 0.2
    
    if junior_analysis:
        junior_bias = junior_analysis.get('intraday_bias', '')
        if (direction == 'LONG' and junior_bias == 'long') or (direction == 'SHORT' and junior_bias == 'short'):
            confidence += 0.15
    
    rr1 = trader_plan.get('risk_reward_1', 0)
    if rr1 >= 2.0:
        confidence += 0.1
    elif rr1 < 1.0:
        confidence -= 0.1
    
    trader_plan['confidence'] = round(max(0.1, min(1.0, confidence)), 2)
    
    if position < 1 or position > 5:
        trader_plan['position_size_pct'] = 2.0
        warnings.append("Position size corrected to 2%")
        fixed = True
    
    if warnings:
        print(f"   ⚠️ VALIDATION: {'; '.join(warnings)}")
    
    return trader_plan, fixed