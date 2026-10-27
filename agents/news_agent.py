# agents/news_agent.py
import time
from typing import List, Dict, Any, Optional
from agents.prompts import build_news_prompt, format_news_report
from data.news_filter import filter_relevant_news
from utils.ollama import call_ollama_json
from utils.schemas import NEWS_SCHEMA
from config import NEWS_MODEL, MAX_NEWS

# Simple cache (optional)
_news_cache = {}

def analyze_news(ticker: str, news_items: List[Dict], model: str = NEWS_MODEL,
                 use_cache: bool = False, max_news: int = 7) -> Dict[str, Any]:
    """
    Analyses news for a ticker.
    Returns a dict with fields:
        overall_sentiment, summary, key_themes, detailed_analysis, catalysts, risk_flags
    """
    # Cache (optional)
    cache_key = f"{ticker}_{hash(str(news_items))}"
    if use_cache and cache_key in _news_cache:
        print(f"♻️ Using cached news for {ticker}")
        return _news_cache[cache_key]

    # Keep only relevant news
    relevant_news = filter_relevant_news(ticker, news_items, max_news=max_news)
    if not relevant_news:
        empty = {
            "overall_sentiment": "neutral",
            "summary": "No news",
            "key_themes": [],
            "detailed_analysis": "",
            "catalysts": [],
            "risk_flags": []
        }
        if use_cache:
            _news_cache[cache_key] = empty
        return empty

    # Build the prompt and call the model
    prompt = build_news_prompt(ticker, relevant_news)
    analysis = call_ollama_json(prompt, model, json_schema=NEWS_SCHEMA, temperature=0)

    # Make sure all fields are present
    for field in ["overall_sentiment", "summary", "key_themes", "detailed_analysis", "catalysts", "risk_flags"]:
        if field not in analysis:
            if field in ["key_themes", "catalysts", "risk_flags"]:
                analysis[field] = []
            else:
                analysis[field] = ""

    if use_cache:
        _news_cache[cache_key] = analysis

    return analysis

# For Telegram, the ready-made format_news_report from prompts.py can be used
# or wrapped here so existing code does not change
def format_news_for_telegram(ticker: str, analysis: Dict[str, Any], news_items: List[Dict]) -> str:
    """Formats the news analysis for Telegram (uses format_news_report)"""
    return format_news_report(ticker, analysis, news_items)