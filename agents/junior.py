# agents/junior.py
from config import JUNIOR_MODEL, MAX_LINE_LOG
from utils.schemas import JUNIOR_SCHEMA
from utils.ollama import call_ollama_json
from utils.helpers import flatten_to_nested
from agents.prompts import build_junior_prompt


def run_junior_analyst(ticker, indicators, news, relative_context, 
                       intraday_levels=None, volume_profile=None, 
                       stumpy_text="", value_text="",
                       news_analysis=None):  # ← NEW PARAMETER
    """
    Runs the Junior analyst.
    
    Args:
        ticker: ticker symbol
        indicators: dict of indicators
        news: list of raw news items
        relative_context: context
        intraday_levels: intraday levels
        volume_profile: volume profile
        stumpy_text: text from STUMPY
        value_text: text from VALUE INVESTING
        news_analysis: ready-made news analysis (dict); if None, raw news is used
    """
    print(f"\n🧠 [4/7] JUNIOR ANALYST")
    junior_prompt = build_junior_prompt(
        ticker, indicators, news, relative_context, 
        intraday_levels, volume_profile, stumpy_text, value_text,
        news_analysis=news_analysis  # ← PASSED INTO THE PROMPT
    )
    print(f"   📝 PROMPT: {len(junior_prompt)} chars")
    for line in junior_prompt.split('\n'):
        print(f"   │ {line[:MAX_LINE_LOG]}")
    
    analysis = call_ollama_json(junior_prompt, JUNIOR_MODEL, json_schema=JUNIOR_SCHEMA, temperature=0)
    analysis = flatten_to_nested(analysis)
    
    print(f"\n   ✅ JUNIOR SUMMARY:")
    summary = analysis.get("executive_summary", "")
    if isinstance(summary, list):
        summary = " ".join(summary)
    if summary:
        for line in summary.split('\n'):
            print(f"   │ {line[:MAX_LINE_LOG]}")
    else:
        print(f"   ⚠️ The model did not fill executive_summary")
    
    return analysis