from config import EXTRACTOR_MODEL, MAX_LINE_LOG
from utils.schemas import EXTRACTOR_SCHEMA
from utils.ollama import call_ollama_json
from agents.prompts import build_extractor_prompt


def run_extractor(indicators, intraday_levels, relative_context, market_tags, orderflow, earnings_info, rag_knowledge):
    print(f"\n📝 [6/7] KNOWLEDGE EXTRACTOR")
    
    try:
        extractor_prompt = build_extractor_prompt(indicators, intraday_levels, relative_context, market_tags, orderflow, earnings_info, rag_knowledge)
        
        if extractor_prompt and len(extractor_prompt) > 0:
            print(f"   📝 EXTRACTOR PROMPT: {len(extractor_prompt)} chars")
            
            # ✅ PRINT THE PROMPT CONTENT
            print(f"   📄 PROMPT CONTENT:")
            for line in extractor_prompt.split('\n'):
                print(f"   │ {line[:MAX_LINE_LOG]}")
            print(f"   {'─'*60}")
            
            extracted_rules = call_ollama_json(extractor_prompt, EXTRACTOR_MODEL, json_schema=EXTRACTOR_SCHEMA, temperature=0)
            
            if extracted_rules:
                print(f"   ✅ Pattern: {extracted_rules.get('pattern', '')[:2000]}")
                print(f"   ✅ Rules: {', '.join(extracted_rules.get('rules', []))}")
                print(f"   ✅ Warnings: {', '.join(extracted_rules.get('warnings', []))}")
                return extracted_rules
            else:
                print(f"   ⚠️ EXTRACTOR returned an empty result")
                return {}
        else:
            print(f"   ⚠️ Could not build the extractor prompt")
            return {}
    except Exception as e:
        print(f"   ⚠️ Extractor error: {str(e)[:200]}")
        return {}