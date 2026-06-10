"""
Groq LLM Client — Fast extraction and structured data generation.

Design Decision: Groq is used for tasks requiring speed and structured output:
entity extraction, sentiment analysis, translation, and competitor detection.
These tasks are well-suited to Groq because:
1. They have clear, constrained output formats (JSON)
2. They don't require deep multi-hop reasoning
3. Speed matters when processing 120+ accounts
4. Llama 3.3 70B has excellent instruction following

Error Handling: We implement retry with backoff because Groq's free tier
has rate limits. Failures gracefully degrade to heuristic-based extraction.
"""

import json
import time
import logging
from typing import Dict

from renewal_intelligence.config.settings import GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE
from renewal_intelligence.llm.prompts import (
    CSM_NOTE_EXTRACTION_PROMPT,
    TRANSLATE_COMMENT_PROMPT,
    BATCH_EXTRACTION_PROMPT,
)

logger = logging.getLogger(__name__)


def _get_groq_client():
    """Lazy initialization of Groq client."""
    try:
        from groq import Groq
        if not GROQ_API_KEY:
            logger.warning("GROQ_API_KEY not set. LLM features will use fallback heuristics.")
            return None
        return Groq(api_key=GROQ_API_KEY)
    except ImportError:
        logger.warning("groq package not installed. Using fallback heuristics.")
        return None


def extract_csm_intelligence(
    note_text: str,
    account_name: str,
    account_id: int,
    arr: float,
    plan_tier: str,
    contract_end: str,
) -> Dict:
    """
    Extract structured intelligence from a CSM note using Groq.
    
    Falls back to heuristic extraction if Groq is unavailable.
    
    Args:
        note_text: Raw CSM note text
        account_name: Official account name
        account_id: Account ID
        arr: Annual recurring revenue
        plan_tier: Plan tier
        contract_end: Contract end date string
        
    Returns:
        Structured extraction result
    """
    client = _get_groq_client()
    
    if client is None:
        return _heuristic_csm_extraction(note_text)
    
    prompt = CSM_NOTE_EXTRACTION_PROMPT.format(
        note_text=note_text,
        account_name=account_name,
        account_id=account_id,
        arr=arr,
        plan_tier=plan_tier,
        contract_end=contract_end,
    )
    
    return _call_groq(client, prompt, fallback=lambda: _heuristic_csm_extraction(note_text))


def translate_comment(comment: str) -> Dict:
    """
    Translate a non-English NPS comment using Groq.
    
    Falls back to returning the original comment if translation fails.
    """
    client = _get_groq_client()
    
    if client is None:
        return {
            "translation": comment,
            "original_language": "unknown",
            "sentiment": "unknown",
            "risk_signals": [],
        }
    
    prompt = TRANSLATE_COMMENT_PROMPT.format(comment=comment)
    
    return _call_groq(client, prompt, fallback=lambda: {
        "translation": comment,
        "original_language": "unknown",
        "sentiment": "unknown",
        "risk_signals": [],
    })


def batch_extract_notes(notes_text: str) -> Dict:
    """Extract risk signals from multiple CSM notes at once."""
    client = _get_groq_client()
    
    if client is None:
        return {"high_risk_accounts": [], "portfolio_themes": [], "urgent_actions": []}
    
    prompt = BATCH_EXTRACTION_PROMPT.format(notes_text=notes_text)
    return _call_groq(client, prompt, fallback=lambda: {
        "high_risk_accounts": [],
        "portfolio_themes": [],
        "urgent_actions": [],
    })


def _call_groq(client, prompt: str, fallback, max_retries: int = 3) -> Dict:
    """
    Call Groq API with retry logic.
    
    Implements exponential backoff for rate limit handling.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise data extraction assistant. Always respond with valid JSON only. No markdown, no explanations, just JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=GROQ_TEMPERATURE,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            
            result = response.choices[0].message.content
            return json.loads(result)
            
        except json.JSONDecodeError as e:
            logger.warning(f"Groq returned invalid JSON (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return fallback()
                
        except Exception as e:
            logger.warning(f"Groq API error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
            else:
                return fallback()
    
    return fallback()


def _heuristic_csm_extraction(note_text: str) -> Dict:
    """
    Fallback heuristic extraction when Groq is unavailable.
    
    Uses keyword matching to approximate LLM extraction.
    This ensures the app works even without API keys.
    """
    text_lower = note_text.lower()
    
    # Competitor detection
    competitor_keywords = {
        "hygraph": "Hygraph",
        "contentful": "Contentful",
        "strapi": "Strapi",
        "sanity": "Sanity",
        "kontent.ai": "Kontent.ai",
        "builder.io": "builder.io",
        "wordpress": "WordPress",
    }
    competitors = [
        name for keyword, name in competitor_keywords.items()
        if keyword in text_lower
    ]
    
    # Sentiment
    negative_words = ["frustrated", "furious", "tense", "threatening", "walk", "done", 
                       "embarrassing", "lost faith", "poorly", "painful"]
    positive_words = ["happy", "love", "great", "smooth", "champagne", "excellent", 
                       "expanding", "good news"]
    
    neg_count = sum(1 for w in negative_words if w in text_lower)
    pos_count = sum(1 for w in positive_words if w in text_lower)
    
    if neg_count > pos_count:
        sentiment = "negative"
    elif pos_count > neg_count:
        sentiment = "positive"
    elif neg_count > 0 and pos_count > 0:
        sentiment = "mixed"
    else:
        sentiment = "neutral"
    
    # Executive detection
    exec_titles = ["vp", "cto", "cro", "ciso", "director", "c-suite", "chief"]
    executive_involved = any(title in text_lower for title in exec_titles)
    
    # Budget concern
    budget_words = ["budget", "price", "cost", "discount", "expensive", "cut"]
    budget_concern = any(w in text_lower for w in budget_words)
    
    # Risk factors
    risk_factors = []
    if competitors:
        risk_factors.append(f"Competitor evaluation: {', '.join(competitors)}")
    if executive_involved:
        risk_factors.append("Executive involvement detected")
    if budget_concern:
        risk_factors.append("Budget/pricing concerns raised")
    if "no show" in text_lower or "no-show" in text_lower:
        risk_factors.append("Missed meeting/QBR")
    if "churn" in text_lower or "cancel" in text_lower:
        risk_factors.append("Churn/cancellation language detected")
    if "migration" in text_lower or "moving" in text_lower:
        risk_factors.append("Migration activity detected")
    if "downgrade" in text_lower:
        risk_factors.append("Downgrade discussion")
    if "merger" in text_lower or "acquired" in text_lower:
        risk_factors.append("M&A activity")
    
    # Renewal signal
    if any(w in text_lower for w in ["walk", "cancel", "done", "explore options"]):
        renewal_signal = "negative"
    elif any(w in text_lower for w in ["expanding", "upgrade", "locked in", "formality"]):
        renewal_signal = "positive"
    elif competitors or budget_concern:
        renewal_signal = "mixed"
    else:
        renewal_signal = "neutral"
    
    # Champion status
    if "lost faith" in text_lower or "nervous" in text_lower or "champion" in text_lower:
        if "lost" in text_lower or "nervous" in text_lower:
            champion_status = "at_risk"
        else:
            champion_status = "active"
    else:
        champion_status = "unknown"
    
    # Missed QBRs
    missed_qbrs = text_lower.count("missed qbr") + text_lower.count("no show") + text_lower.count("no-show")
    
    return {
        "sentiment": sentiment,
        "competitors": competitors,
        "risk_factors": risk_factors,
        "executive_involved": executive_involved,
        "budget_concern": budget_concern,
        "renewal_signal": renewal_signal,
        "champion_status": champion_status,
        "missed_qbrs": missed_qbrs,
        "action_items": [],
        "key_quote": "",
    }
