"""
Gemini LLM Client — Deep reasoning, explanations, and multimodal analysis.

Design Decision: Gemini handles the "thinking" tasks that require:
1. Multi-hop reasoning across datasets
2. Nuanced natural language generation
3. Contradictory signal resolution
4. Vision-based chart analysis (multimodal)

Gemini 2.5 Flash is chosen over Pro because:
- Faster inference for interactive Streamlit UX
- Sufficient reasoning capability for CS analysis
- More cost-effective for a prototype
- Good vision capabilities for chart analysis

The multimodal pipeline:
1. Plotly charts are rendered as PNG images
2. Images + structured data are sent to Gemini Vision
3. Gemini identifies visual patterns (declining curves, anomalies)
4. Visual insights augment the structured analysis
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional

from renewal_intelligence.config.settings import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE
from renewal_intelligence.llm.prompts import (
    RENEWAL_RISK_EXPLANATION_PROMPT,
    SILENT_CHURN_DETECTION_PROMPT,
    PORTFOLIO_INSIGHTS_PROMPT,
    MULTIMODAL_ANALYSIS_PROMPT,
)

logger = logging.getLogger(__name__)


def _get_gemini_model():
    """Lazy initialization of Gemini client."""
    try:
        import google.generativeai as genai
        if not GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not set. Gemini features will use fallback text.")
            return None
        genai.configure(api_key=GEMINI_API_KEY)
        return genai.GenerativeModel(GEMINI_MODEL)
    except ImportError:
        logger.warning("google-generativeai package not installed. Using fallback.")
        return None


def generate_risk_explanation(
    account_name: str,
    account_id: int,
    plan_tier: str,
    arr: float,
    region: str,
    contract_end: str,
    days_to_renewal: int,
    csm_name: str,
    usage_summary: str,
    ticket_summary: str,
    nps_summary: str,
    csm_intelligence: str,
    product_risk_summary: str,
    risk_score: float,
    risk_tier: str,
) -> str:
    """
    Generate a detailed risk explanation using Gemini.
    
    This is the crown jewel of the system — it produces the plain-English
    explanations that make the tool actually useful for BizOps teams.
    
    Falls back to a template-based explanation if Gemini is unavailable.
    """
    model = _get_gemini_model()
    
    prompt = RENEWAL_RISK_EXPLANATION_PROMPT.format(
        account_name=account_name,
        account_id=account_id,
        plan_tier=plan_tier,
        arr=arr,
        region=region,
        contract_end=contract_end,
        days_to_renewal=days_to_renewal,
        csm_name=csm_name,
        usage_summary=usage_summary,
        ticket_summary=ticket_summary,
        nps_summary=nps_summary,
        csm_intelligence=csm_intelligence,
        product_risk_summary=product_risk_summary,
        risk_score=f"{risk_score:.2f}",
        risk_tier=risk_tier,
    )
    
    if model is None:
        return _fallback_explanation(
            account_name, risk_tier, risk_score, usage_summary,
            ticket_summary, nps_summary, csm_intelligence, product_risk_summary
        )
    
    return _call_gemini(model, prompt, fallback=lambda: _fallback_explanation(
        account_name, risk_tier, risk_score, usage_summary,
        ticket_summary, nps_summary, csm_intelligence, product_risk_summary
    ))


def detect_silent_churn(
    account_name: str,
    nps_score: Optional[int],
    nps_comment: str,
    api_trend: float,
    user_trend: float,
    content_trend: float,
    csm_intelligence: str,
) -> Dict:
    """
    Use Gemini to detect silent churn patterns.
    
    Silent churn is the most dangerous form of churn because it
    doesn't trigger traditional alert systems. The customer appears
    healthy on surface metrics but is quietly building alternatives.
    """
    model = _get_gemini_model()
    
    prompt = SILENT_CHURN_DETECTION_PROMPT.format(
        account_name=account_name,
        nps_score=nps_score if nps_score is not None else "N/A",
        nps_comment=nps_comment or "No comment provided",
        api_trend=f"{api_trend * 100:.1f}" if api_trend else "0",
        user_trend=f"{user_trend * 100:.1f}" if user_trend else "0",
        content_trend=f"{content_trend * 100:.1f}" if content_trend else "0",
        csm_intelligence=csm_intelligence or "No CSM notes available",
    )
    
    if model is None:
        return _fallback_silent_churn()
    
    result = _call_gemini(model, prompt, fallback=_fallback_silent_churn, expect_json=True)
    
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return _fallback_silent_churn()
    
    return result


def generate_portfolio_insights(
    total_accounts: int,
    high_risk_count: int,
    high_risk_arr: float,
    medium_risk_count: int,
    medium_risk_arr: float,
    low_risk_count: int,
    top_risk_accounts: str,
    portfolio_signals: str,
    product_risk_events: str,
) -> str:
    """Generate portfolio-level executive insights."""
    model = _get_gemini_model()
    
    prompt = PORTFOLIO_INSIGHTS_PROMPT.format(
        total_accounts=total_accounts,
        high_risk_count=high_risk_count,
        high_risk_arr=high_risk_arr,
        medium_risk_count=medium_risk_count,
        medium_risk_arr=medium_risk_arr,
        low_risk_count=low_risk_count,
        top_risk_accounts=top_risk_accounts,
        portfolio_signals=portfolio_signals,
        product_risk_events=product_risk_events,
    )
    
    if model is None:
        return _fallback_portfolio_insights(
            total_accounts, high_risk_count, high_risk_arr,
            medium_risk_count, medium_risk_arr, low_risk_count
        )
    
    return _call_gemini(model, prompt, fallback=lambda: _fallback_portfolio_insights(
        total_accounts, high_risk_count, high_risk_arr,
        medium_risk_count, medium_risk_arr, low_risk_count
    ))


def analyze_charts_multimodal(
    account_name: str,
    risk_tier: str,
    structured_data: str,
    chart_images: List[Path],
) -> str:
    """
    Perform multimodal analysis using Gemini Vision.
    
    This is how we implement the multimodal requirement:
    1. Charts are pre-rendered as PNG images by the visualization module
    2. Images are base64-encoded and sent alongside structured data
    3. Gemini Vision analyzes visual patterns + numbers together
    4. Output identifies patterns that structured data alone misses
    
    Why multimodal matters:
    - Humans process trends visually (a declining curve is more intuitive)
    - Visual analysis catches acceleration/deceleration patterns
    - Combined analysis resolves ambiguous numerical signals
    """
    model = _get_gemini_model()
    
    if model is None:
        return "Multimodal analysis requires Gemini API access. Set GEMINI_API_KEY to enable."
    
    prompt = MULTIMODAL_ANALYSIS_PROMPT.format(
        account_name=account_name,
        risk_tier=risk_tier,
        structured_data=structured_data,
    )
    
    # Build multimodal content
    content_parts = [prompt]
    
    for img_path in chart_images:
        if img_path.exists():
            try:
                import PIL.Image
                img = PIL.Image.open(str(img_path))
                content_parts.append(img)
            except Exception as e:
                logger.warning(f"Failed to load chart image {img_path}: {e}")
    
    try:
        response = model.generate_content(
            content_parts,
            generation_config={
                "temperature": GEMINI_TEMPERATURE,
                "max_output_tokens": 1024,
            }
        )
        return response.text
    except Exception as e:
        logger.error(f"Gemini multimodal analysis failed: {e}")
        return f"Visual analysis unavailable: {e}"


def _call_gemini(model, prompt: str, fallback, max_retries: int = 3, expect_json: bool = False):
    """Call Gemini API with retry logic."""
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": GEMINI_TEMPERATURE,
                    "max_output_tokens": 2048,
                }
            )
            
            result = response.text
            
            if expect_json:
                # Try to parse JSON from the response
                # Gemini sometimes wraps JSON in markdown code blocks
                cleaned = result.strip()
                if cleaned.startswith("```"):
                    # Remove markdown code block
                    lines = cleaned.split("\n")
                    cleaned = "\n".join(lines[1:-1])
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    if attempt == max_retries - 1:
                        return fallback()
                    continue
            
            return result
            
        except Exception as e:
            logger.warning(f"Gemini API error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return fallback()
    
    return fallback()


def _fallback_explanation(
    account_name: str,
    risk_tier: str,
    risk_score: float,
    usage_summary: str,
    ticket_summary: str,
    nps_summary: str,
    csm_intelligence: str,
    product_risk_summary: str,
) -> str:
    """Generate a template-based explanation when Gemini is unavailable."""
    return f"""## Risk Assessment for {account_name}

**Risk Tier: {risk_tier}** (Score: {risk_score:.2f}/1.0)

### Usage Analysis
{usage_summary}

### Support Ticket Analysis
{ticket_summary}

### NPS Analysis
{nps_summary}

### CSM Intelligence
{csm_intelligence}

### Product Risk Impact
{product_risk_summary}

---
*Note: Set GEMINI_API_KEY for AI-generated strategic analysis and recommendations.*"""


def _fallback_silent_churn() -> Dict:
    """Fallback for silent churn detection."""
    return {
        "is_silent_churn": False,
        "confidence": 0.0,
        "pattern_type": "analysis_unavailable",
        "evidence": ["Gemini API not configured — manual review recommended"],
        "counter_evidence": [],
        "recommended_investigation": "Set GEMINI_API_KEY for automated silent churn detection",
    }


def _fallback_portfolio_insights(
    total: int, high: int, high_arr: float,
    medium: int, medium_arr: float, low: int
) -> str:
    """Fallback portfolio insights."""
    return f"""## Portfolio Summary

- **{total}** accounts in the renewal window
- **{high} High Risk** accounts with **${high_arr:,.0f}** ARR at risk
- **{medium} Medium Risk** accounts with **${medium_arr:,.0f}** ARR at risk  
- **{low} Low Risk** accounts on track for renewal

*Set GEMINI_API_KEY for AI-generated strategic portfolio insights.*"""
