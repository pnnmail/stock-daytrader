import os as os_module
import requests
import numpy as np
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from config import OLLAMA_URL, EMBED_MODEL, CHROMA_DB_PATH, KNOWLEDGE_PATH

RAG_ENABLED = os_module.path.exists(KNOWLEDGE_PATH) and bool(os_module.listdir(KNOWLEDGE_PATH))

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)


class OllamaEmbedding(EmbeddingFunction):
    def __init__(self):
        pass
    
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            resp = requests.post(
                OLLAMA_URL.replace("generate", "embeddings"), 
                json={"model": EMBED_MODEL, "prompt": text}
            ).json()
            if "embeddings" in resp:
                embeddings.append(resp["embeddings"][0])
            elif "embedding" in resp:
                embeddings.append(resp["embedding"])
            else:
                print(f"   ⚠️ Unexpected response from Ollama: {list(resp.keys())}")
                embeddings.append([0.0] * 768)
        return embeddings


collection = None
if RAG_ENABLED:
    collection = chroma_client.get_or_create_collection("trading_knowledge", embedding_function=OllamaEmbedding())


def get_embedding(text):
    resp = requests.post(
        OLLAMA_URL.replace("generate", "embeddings"), 
        json={"model": EMBED_MODEL, "prompt": text[:1000]}
    ).json()
    # The Ollama API returns "embeddings" (plural) or "embedding"
    if "embeddings" in resp:
        return np.array(resp["embeddings"][0])
    elif "embedding" in resp:
        return np.array(resp["embedding"])
    else:
        print(f"   ⚠️ Unexpected response from Ollama: {list(resp.keys())}")
        return np.array([0.0] * 768)  # fallback: a zero vector


GOOD_CATEGORIES = {
    "candles": "candles doji hammer engulfing star harami marubozu reversal pattern candlestick chart japanese candlestick shadow body bullish bearish piercing dark cloud cover morning evening abandoned baby hanging man inverted hammer shooting star",
    "indicators": "RSI MACD stochastic oscillator indicator bollinger bands moving average SMA EMA momentum rate of change convergence divergence signal line overbought oversold divergence convergence histogram",
    "levels": "support resistance level trendline pivot volatility channel breakout pullback correction ATR range fibonacci retracement extension",
    "volume_analysis": "volume VWAP market profile delta order flow POC Value Area accumulation distribution climax blow-off spike exhaustion turnover",
    "patterns": "head shoulders triangle flag pennant double top bottom reversal trend ascending descending sideways wedge rectangle broadening diamond cup handle",
    "psychology": "fear greed psychology crowd panic euphoria discipline trader behaviour emotions market mood sentiment",
    "risk": "stop-loss risk management capital position hedging volatility money management lot size leverage margin loss profit take-profit",
    "orderflow": "order flow order book depth of market liquidity spread bid ask market maker large player institutional volume",
    "breakouts": "breakout break false breakout retest confirmation momentum acceleration deceleration consolidation range",
    "timing": "open close session intraday timeframe minute hourly daily weekly",
    "entry_exit": "entry exit signal confirmation condition trigger entry point scalping swing day trading",
    "market_structure": "market structure trend phase accumulation distribution market maker manipulation liquidity stop hunt",
    "trading_rules": "rule entry exit stop target signal reversal trend specific instruction",
    "trading_ideas": "idea concept approach method strategy observation important key",
    "trading_warnings": "warning danger trap mistake risk caution attention",
}

GOOD_EMBEDDINGS = None


def init_good_embeddings():
    global GOOD_EMBEDDINGS
    if GOOD_EMBEDDINGS is not None:
        return
    print("📚 Initialising reference embeddings for classification...")
    GOOD_EMBEDDINGS = {}
    for category, description in GOOD_CATEGORIES.items():
        GOOD_EMBEDDINGS[category] = get_embedding(description)
    print("   ✅ Reference embeddings ready")


def classify_chunk(text):
    init_good_embeddings()
    if len(text.strip()) < 100:
        return "garbage"
    first_line = text.strip().split('\n')[0].lower()
    obvious_junk = ['table of contents', 'contents', 'isbn', 'copyright', 'all rights reserved', 'printed in']
    if any(junk in first_line for junk in obvious_junk):
        return "garbage"
    chunk_embedding = get_embedding(text[:1000])
    best_similarity = 0
    best_category = "garbage"
    for category, good_emb in GOOD_EMBEDDINGS.items():
        similarity = np.dot(chunk_embedding, good_emb) / (np.linalg.norm(chunk_embedding) * np.linalg.norm(good_emb))
        if similarity > best_similarity:
            best_similarity = similarity
            best_category = category
    if best_similarity >= 0.40:
        return best_category
    return "garbage"