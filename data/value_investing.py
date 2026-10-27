"""
Value investing module (Graham, Buffett, Lynch).
Computes fundamental metrics and a checklist.
"""
import time
import numpy as np
import yfinance as yf
from data.fetcher import RATE_LIMIT_BACKOFF


def _fetch_info_with_retry(ticker):
    """
    yf.Ticker(ticker).info used to be called without a single retry — any
    "Too Many Requests" from Yahoo (especially after a heavy /autoscan) immediately
    zeroed the whole checklist. The same stretched backoff as in
    data/fetcher.py::fetch_relative_context (10-15-30s, 4 attempts).
    """
    max_retries = len(RATE_LIMIT_BACKOFF) + 1
    last_error = None
    for attempt in range(max_retries):
        try:
            return yf.Ticker(ticker).info
        except Exception as e:
            last_error = e
            print(f"   ⚠️ value_investing info error (attempt {attempt+1}/{max_retries}): {str(e)[:200]}")
            if attempt < max_retries - 1:
                time.sleep(RATE_LIMIT_BACKOFF[attempt])
    raise last_error


def compute_value_investing_score(ticker, indicators, fundamentals=None):
    """
    Computes the value investor's checklist.
    Returns a dict with the results and an overall score (0-10).
    """
    try:
        info = _fetch_info_with_retry(ticker)

        # Extract fundamental data
        eps = info.get('trailingEps')  # Earnings per share (TTM)
        bvps = info.get('bookValue')   # Book value per share
        pe = info.get('trailingPE')    # P/E (TTM)
        pb = info.get('priceToBook')   # P/B
        roe = info.get('returnOnEquity')  # ROE (in percent)
        current_ratio = info.get('currentRatio')  # Current ratio
        peg = info.get('pegRatio')     # PEG
        growth = info.get('earningsGrowth')  # Earnings growth
        
        price = indicators['price']
        score = 0
        details = []
        
        # --- 1. Graham number ---
        graham_number = None
        margin_of_safety = None
        if eps and bvps and eps > 0 and bvps > 0:
            graham_number = round(np.sqrt(22.5 * eps * bvps), 2)
            margin_of_safety = round((graham_number - price) / graham_number * 100, 1)
            if price < graham_number:
                score += 2
                details.append(f"✅ Graham number: ${graham_number:.2f} (price is {abs(margin_of_safety)}% below — margin of safety)")
            else:
                details.append(f"❌ Graham number: ${graham_number:.2f} (price is {abs(margin_of_safety)}% above)")
        else:
            details.append("⚠️ Graham number: not enough data (negative EPS or BVPS)")
        
        # --- 2. Graham liquidity test ---
        if current_ratio:
            if current_ratio >= 2.0:
                score += 1
                details.append(f"✅ Liquidity: current ratio {current_ratio:.1f} ≥ 2.0")
            else:
                details.append(f"❌ Liquidity: current ratio {current_ratio:.1f} < 2.0")
        else:
            details.append("⚠️ Liquidity: no data")
        
        # --- 3. P/E and P/B ---
        pe_ok = False
        pb_ok = False
        if pe and pe > 0:
            if pe <= 15:
                score += 1
                pe_ok = True
                details.append(f"✅ P/E: {pe:.1f} ≤ 15")
            else:
                details.append(f"❌ P/E: {pe:.1f} > 15")
        else:
            details.append("⚠️ P/E: negative or no data")
        
        if pb and pb > 0:
            if pb <= 1.5:
                score += 1
                pb_ok = True
                details.append(f"✅ P/B: {pb:.1f} ≤ 1.5")
            else:
                details.append(f"❌ P/B: {pb:.1f} > 1.5")
        else:
            details.append("⚠️ P/B: negative or no data")
        
        # Combined P/E × P/B test
        if pe_ok and pb_ok and pe and pb:
            combo = pe * pb
            if combo <= 22.5:
                score += 1
                details.append(f"✅ P/E × P/B: {combo:.1f} ≤ 22.5")
            else:
                details.append(f"❌ P/E × P/B: {combo:.1f} > 22.5")
        
        # --- 4. ROE (Buffett) ---
        if roe:
            if roe >= 0.15:  # 15%
                score += 2
                details.append(f"✅ ROE: {roe*100:.1f}% ≥ 15% (economic moat)")
            else:
                details.append(f"❌ ROE: {roe*100:.1f}% < 15%")
        else:
            details.append("⚠️ ROE: no data")
        
        # --- 5. PEG (Lynch) ---
        if peg and peg > 0:
            if peg < 1.0:
                score += 1
                details.append(f"✅ PEG: {peg:.2f} < 1.0 (growth justifies the price)")
            else:
                details.append(f"❌ PEG: {peg:.2f} ≥ 1.0")
        elif growth and pe and pe > 0:
            peg_calc = pe / (growth * 100) if growth > 0 else None
            if peg_calc and peg_calc < 1.0:
                score += 1
                details.append(f"✅ PEG (calc.): {peg_calc:.2f} < 1.0")
            elif peg_calc:
                details.append(f"❌ PEG (calc.): {peg_calc:.2f} ≥ 1.0")
        else:
            details.append("⚠️ PEG: no data")
        
        return {
            'ticker': ticker,
            'score': min(score, 10),
            'graham_number': graham_number,
            'margin_of_safety': margin_of_safety,
            'pe': pe,
            'pb': pb,
            'roe': roe,
            'peg': peg,
            'current_ratio': current_ratio,
            'details': details,
        }
    
    except Exception as e:
        print(f"   ⚠️ value_investing error for {ticker}: {str(e)[:100]}")
        return {
            'ticker': ticker,
            'score': 0,
            'details': [f"⚠️ Could not compute: {str(e)[:80]}"],
        }


def format_value_investing(result):
    """Formats the result for inclusion in a prompt."""
    if not result:
        return ""
    
    lines = []
    lines.append(f"\n📊 *VALUE INVESTOR CHECKLIST ({result['ticker']})*")
    lines.append(f"• Overall score: {result['score']}/10")
    
    if result.get('graham_number'):
        lines.append(f"• Graham number: ${result['graham_number']:.2f}")
        if result.get('margin_of_safety') is not None:
            margin = abs(result['margin_of_safety'])
            if result['margin_of_safety'] > 0:
                lines.append(f"• Margin of safety: +{margin}% (price below the estimate)")
            else:
                lines.append(f"• Margin of safety: -{margin}% (price above the estimate)")
    
    lines.append("\n📋 *Details:*")
    for detail in result.get('details', []):
        lines.append(f"  {detail}")
    
    # Interpretation
    score = result.get('score', 0)
    has_data = any([
        result.get('pe') is not None,
        result.get('pb') is not None,
        result.get('roe') is not None,
        result.get('graham_number') is not None,
    ])
    
    if not has_data:
        lines.append(f"\n⚠️ *Conclusion:* No data for an assessment (biotech, loss-making company or not enough data).")
    elif score >= 7:
        lines.append(f"\n💡 *Conclusion:* A classic value company — strong fundamentals.")
    elif score >= 4:
        lines.append(f"\n💡 *Conclusion:* Mixed metrics — there are both strong and weak sides.")
    elif score >= 1:
        lines.append(f"\n💡 *Conclusion:* The company fails the classic value tests (probably a growth company or overvalued).")
    else:  # score == 0, but there was data
        lines.append(f"\n💡 *Conclusion:* The company fails every value test — overvalued on all classic metrics.")
    
    return "\n".join(lines)