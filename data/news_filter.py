import re
from data.fetcher import get_company_name


def filter_relevant_news(ticker, news_list, max_news=5):
    """
    A universal news filter — works with any ticker.
    Uses the Yahoo Finance API to get the company name.
    """
    ticker_lower = ticker.lower()
    company_name = get_company_name(ticker)
    
    def relevance_score(news):
        title = news['title'].lower()
        text = news.get('text', '').lower()
        score = 0
        
        if ticker_lower in title:
            score += 15
        
        if f'({ticker_lower})' in title:
            score += 10
        
        if ticker_lower in text:
            score += 8
        
        if company_name and company_name in title:
            score += 15
        
        if company_name and company_name in text:
            score += 8
        
        ticker_or_company_mentioned = (
            ticker_lower in title or ticker_lower in text or
            (company_name and company_name in title) or
            (company_name and company_name in text)
        )
        
        if ticker_or_company_mentioned:
            financial_terms = [
                'earnings', 'revenue', 'profit', 'loss', 'guidance', 'outlook',
                'forecast', 'eps', 'ebitda', 'margin', 'dividend',
                'buyback', 'repurchase', 'ipo', 'offering',
                'upgrade', 'downgrade', 'outperform', 'underperform',
                'target', 'price target', 'initiate', 'coverage',
                'probe', 'investigation', 'settlement', 'fine', 'sec', 'ftc',
                'senate', 'lawsuit', 'litigation', 'fraud', 'compliance',
                'regulation', 'antitrust', 'penalty',
                'layoff', 'layoffs', 'acquisition', 'merger', 'partnership', 'deal',
                'spin-off', 'split', 'restructuring', 'bankruptcy',
                'stock', 'shares', 'market', 'trading', 'rally', 'drop', 'fell',
                'surge', 'jump', 'decline', 'plunge', 'soar', 'tumble',
                'bull', 'bear', 'correction', 'crash', 'volatility',
                'inflation', 'rate', 'fed', 'interest', 'gdp', 'recession',
                'economic', 'jobs', 'employment', 'consumer', 'spending',
                'ai', 'artificial intelligence', 'cloud', 'chip', 'semiconductor',
                'data center', 'gpu', 'model', 'openai', 'anthropic',
                'warning', 'alert', 'risk', 'concern', 'fear', 'optimism',
                'pessimism', 'sentiment', 'outlook', 'uncertainty'
            ]
            for term in financial_terms:
                if term in title:
                    score += 5
                elif term in text:
                    score += 2
        
        if ticker_lower in title or (company_name and company_name in title):
            urgent_words = ['warning', 'alert', 'probe', 'investigation', 'crash', 
                           'surge', 'jump', 'drop', 'fell', 'rally', 'record', 'down today']
            for word in urgent_words:
                if word in title:
                    score += 3
        
        other_tickers = re.findall(r'\b[A-Z]{2,5}\b', title)
        for t in other_tickers:
            if t.lower() != ticker_lower and t.lower() not in ['AI', 'IT', 'US', 'FTC', 'SEC', 'ETF']:
                score -= 3
        
        return score
    
    sorted_news = sorted(news_list, key=relevance_score, reverse=True)
    
    relevant = [n for n in sorted_news if relevance_score(n) > 0][:max_news]
    
    if len(relevant) < max_news:
        remaining = [n for n in sorted_news if n not in relevant]
        relevant.extend(remaining[:max_news - len(relevant)])
    
    print(f"   📰 Filtered: {len(relevant)}/{len(news_list)} news items (ticker: {ticker})")
    for i, n in enumerate(relevant):
        print(f"      [{i+1}] score={relevance_score(n)} | {n['title'][:100]}")
    
    return relevant