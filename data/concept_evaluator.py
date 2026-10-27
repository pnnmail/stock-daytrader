# data/concept_evaluator.py
import pandas as pd
import numpy as np

class ConditionEvaluator:
    """
    Checks the book's atomic concepts against the current market.
    market_data = {
        'df': DataFrame (hourly candles, at least 30 rows),
        'indicators': dict (result of compute_indicators),
        'intraday_levels': dict (result of compute_intraday_levels)
    }
    """

    # -----------------------------------------------------------------
    # HELPER FUNCTIONS
    # -----------------------------------------------------------------
    @staticmethod
    def _is_bullish(df, i=-1):
        return df['Close'].iloc[i] > df['Open'].iloc[i]

    @staticmethod
    def _is_bearish(df, i=-1):
        return df['Close'].iloc[i] < df['Open'].iloc[i]

    @staticmethod
    def _body(df, i=-1):
        return abs(df['Close'].iloc[i] - df['Open'].iloc[i])

    @staticmethod
    def _upper_shadow(df, i=-1):
        return df['High'].iloc[i] - max(df['Close'].iloc[i], df['Open'].iloc[i])

    @staticmethod
    def _lower_shadow(df, i=-1):
        return min(df['Close'].iloc[i], df['Open'].iloc[i]) - df['Low'].iloc[i]

    @staticmethod
    def _body_ratio(df, i=-1, total_range=None):
        if total_range is None:
            total_range = df['High'].iloc[i] - df['Low'].iloc[i]
        if total_range == 0:
            return 0
        return ConditionEvaluator._body(df, i) / total_range

    @staticmethod
    def _prev_highs(df, n=4):
        """The last n highs for divergence detection."""
        return df['High'].iloc[-n:].values

    @staticmethod
    def _prev_lows(df, n=4):
        return df['Low'].iloc[-n:].values

    # -----------------------------------------------------------------
    # CONCEPT MAP -> CONDITIONS
    # -----------------------------------------------------------------
    CONCEPT_CONDITIONS = {
        # ------------------- CANDLESTICK PATTERNS -------------------
        "hammer": [
            lambda df: ConditionEvaluator._is_bullish(df),
            lambda df: ConditionEvaluator._lower_shadow(df) > ConditionEvaluator._body(df) * 2,
            lambda df: ConditionEvaluator._upper_shadow(df) < ConditionEvaluator._body(df) * 0.5,
        ],
        "inverted_hammer": [
            lambda df: ConditionEvaluator._is_bullish(df),
            lambda df: ConditionEvaluator._upper_shadow(df) > ConditionEvaluator._body(df) * 2,
            lambda df: ConditionEvaluator._lower_shadow(df) < ConditionEvaluator._body(df) * 0.5,
        ],
        "bullish_engulfing": [
            lambda df: ConditionEvaluator._is_bearish(df, -2),
            lambda df: ConditionEvaluator._is_bullish(df, -1),
            lambda df: df['Open'].iloc[-1] < df['Close'].iloc[-2],
            lambda df: df['Close'].iloc[-1] > df['Open'].iloc[-2],
        ],
        "bearish_engulfing": [
            lambda df: ConditionEvaluator._is_bullish(df, -2),
            lambda df: ConditionEvaluator._is_bearish(df, -1),
            lambda df: df['Open'].iloc[-1] > df['Close'].iloc[-2],
            lambda df: df['Close'].iloc[-1] < df['Open'].iloc[-2],
        ],
        "piercing": [
            lambda df: ConditionEvaluator._is_bearish(df, -2),
            lambda df: ConditionEvaluator._is_bullish(df, -1),
            lambda df: df['Close'].iloc[-1] > (df['Open'].iloc[-2] + df['Close'].iloc[-2]) / 2,
            lambda df: df['Open'].iloc[-1] < df['Close'].iloc[-2],
        ],
        "dark_cloud_cover": [
            lambda df: ConditionEvaluator._is_bullish(df, -2),
            lambda df: ConditionEvaluator._is_bearish(df, -1),
            lambda df: df['Close'].iloc[-1] < (df['Open'].iloc[-2] + df['Close'].iloc[-2]) / 2,
            lambda df: df['Open'].iloc[-1] > df['Close'].iloc[-2],
        ],
        "doji": [
            lambda df: ConditionEvaluator._body(df) < (df['High'].iloc[-1] - df['Low'].iloc[-1]) * 0.1,
        ],
        "morning_star": [  # simplified: 3 candles
            lambda df: len(df) >= 3,
            lambda df: ConditionEvaluator._is_bearish(df, -3),
            lambda df: ConditionEvaluator._body(df, -2) < ConditionEvaluator._body(df, -3) * 0.3,
            lambda df: ConditionEvaluator._is_bullish(df, -1),
            lambda df: df['Close'].iloc[-1] > (df['Open'].iloc[-3] + df['Close'].iloc[-3]) / 2,
        ],
        "evening_star": [
            lambda df: len(df) >= 3,
            lambda df: ConditionEvaluator._is_bullish(df, -3),
            lambda df: ConditionEvaluator._body(df, -2) < ConditionEvaluator._body(df, -3) * 0.3,
            lambda df: ConditionEvaluator._is_bearish(df, -1),
            lambda df: df['Close'].iloc[-1] < (df['Open'].iloc[-3] + df['Close'].iloc[-3]) / 2,
        ],

        # ------------------- INDICATORS -------------------
        "oversold": [
            lambda df, indicators: indicators.get('rsi', 50) < 30,
        ],
        "overbought": [
            lambda df, indicators: indicators.get('rsi', 50) > 70,
        ],
        "rsi_bullish_divergence": [
            lambda df, indicators: ConditionEvaluator._check_rsi_divergence(df, indicators, 'bullish'),
        ],
        "rsi_bearish_divergence": [
            lambda df, indicators: ConditionEvaluator._check_rsi_divergence(df, indicators, 'bearish'),
        ],
        "macd_bullish_cross": [
            lambda df, indicators: indicators.get('macd', 0) > indicators.get('macd_signal', 0),
        ],
        "macd_bearish_cross": [
            lambda df, indicators: indicators.get('macd', 0) < indicators.get('macd_signal', 0),
        ],
        "volume_confirmation": [
            lambda df: df['Volume'].iloc[-1] > df['Volume'].iloc[:-1].mean() * 1.5,
        ],

        # ------------------- TRENDS -------------------
        "downtrend": [
            lambda df: len(df) >= 4 and all(df['Low'].iloc[-i] < df['Low'].iloc[-i-1] for i in range(1, 4)),
            lambda df, indicators: df['Close'].iloc[-1] < df['Close'].rolling(20).mean().iloc[-1],
        ],
        "uptrend": [
            lambda df: len(df) >= 4 and all(df['High'].iloc[-i] > df['High'].iloc[-i-1] for i in range(1, 4)),
            lambda df, indicators: df['Close'].iloc[-1] > df['Close'].rolling(20).mean().iloc[-1],
        ],

        # ------------------- LEVELS / VWAP / ATR -------------------
        "support_bounce": [
            lambda df, indicators, intraday_levels: df['Low'].iloc[-1] <= intraday_levels.get('s1', 0) * 1.01,
            lambda df: df['Close'].iloc[-1] > df['Open'].iloc[-1],  # bounce up
        ],
        "resistance_rejection": [
            lambda df, indicators, intraday_levels: df['High'].iloc[-1] >= intraday_levels.get('r1', 0) * 0.99,
            lambda df: df['Close'].iloc[-1] < df['Open'].iloc[-1],  # pullback down
        ],
        "above_vwap": [
            lambda df, indicators, intraday_levels: indicators['price'] > intraday_levels.get('vwap', 0),
        ],
        "below_vwap": [
            lambda df, indicators, intraday_levels: indicators['price'] < intraday_levels.get('vwap', 0),
        ],
        "high_volatility": [
            lambda df, indicators, intraday_levels: intraday_levels.get('atr_hourly', 0) / indicators['price'] > 0.02,
        ],
        "low_volatility": [
            lambda df, indicators, intraday_levels: intraday_levels.get('atr_hourly', 0) / indicators['price'] < 0.005,
        ],

        # ------------------- BREAKOUTS / GAPS -------------------
        "breakout": [
            lambda df, indicators, intraday_levels: df['Close'].iloc[-1] > intraday_levels.get('prev_high', 0),
        ],
        "breakdown": [
            lambda df, indicators, intraday_levels: df['Close'].iloc[-1] < intraday_levels.get('prev_low', 0),
        ],
        "gap_up": [
            lambda df: df['Low'].iloc[-1] > df['High'].iloc[-2],
        ],
        "gap_down": [
            lambda df: df['High'].iloc[-1] < df['Low'].iloc[-2],
        ],
    }

    # -----------------------------------------------------------------
    # CHECK METHODS
    # -----------------------------------------------------------------
    @staticmethod
    def _check_rsi_divergence(df, indicators, direction):
        """A simplified divergence check over the last 10 candles."""
        rsi_series = indicators.get('rsi_series', None)
        if rsi_series is None or len(rsi_series) < 10:
            return False
        closes = df['Close'].iloc[-10:].values
        rsis = rsi_series[-10:].values
        if direction == 'bullish':
            # Price makes a new low, RSI does not
            price_min_idx = closes.argmin()
            rsi_min_idx = rsis.argmin()
            return price_min_idx > rsi_min_idx and closes[price_min_idx] < closes[rsi_min_idx] and rsis[price_min_idx] > rsis[rsi_min_idx]
        else:  # bearish
            price_max_idx = closes.argmax()
            rsi_max_idx = rsis.argmax()
            return price_max_idx > rsi_max_idx and closes[price_max_idx] > closes[rsi_max_idx] and rsis[price_max_idx] < rsis[rsi_max_idx]

    @staticmethod
    def evaluate_concept(concept_id, market_data):
        """
        Returns (is_met: bool, quality: float, details: str).
        """
        if concept_id not in ConditionEvaluator.CONCEPT_CONDITIONS:
            return False, 0.0, f"Unknown concept: {concept_id}"

        conditions = ConditionEvaluator.CONCEPT_CONDITIONS[concept_id]
        passed = 0
        total = len(conditions)
        details = []
        df = market_data['df']
        indicators = market_data.get('indicators', {})
        intraday_levels = market_data.get('intraday_levels', {})

        for cond in conditions:
            try:
                # Try calling with a different number of arguments
                result = cond(df, indicators, intraday_levels)
            except TypeError:
                try:
                    result = cond(df, indicators)
                except TypeError:
                    result = cond(df)
            if result:
                passed += 1
            details.append(f"{'✓' if result else '✗'} condition")

        quality = passed / total if total > 0 else 0.0
        return passed == total, round(quality, 2), "; ".join(details)

    @staticmethod
    def evaluate_hammer_quality(df):
        """Rates the quality of a hammer (0..1)."""
        c = df['Close'].iloc[-1]
        o = df['Open'].iloc[-1]
        h = df['High'].iloc[-1]
        l = df['Low'].iloc[-1]
        body = abs(c - o)
        lower_shadow = o - l if c > o else c - l
        upper_shadow = h - c if c > o else h - o
        if body == 0:
            return 0.5
        ratio = lower_shadow / body
        quality = min(1.0, ratio / 2.0) * 0.7 + (0.3 if upper_shadow < body else 0)
        return round(quality, 2)

    @staticmethod
    def evaluate_engulfing_quality(df):
        """Quality of a bullish engulfing (body size and overlap)."""
        body1 = abs(df['Close'].iloc[-2] - df['Open'].iloc[-2])
        body2 = abs(df['Close'].iloc[-1] - df['Open'].iloc[-1])
        if body1 == 0:
            return 0.8
        ratio = body2 / body1
        overlap = (df['Close'].iloc[-1] - df['Open'].iloc[-2]) / body1 if body1 else 0
        quality = min(1.0, ratio / 2) * 0.5 + min(1.0, overlap) * 0.5
        return round(quality, 2)