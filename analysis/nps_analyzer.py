"""
NPS Analyzer — Extracts risk signals from Net Promoter Score data.

Design Decision: NPS alone is a weak predictor (Meridian Health proves this).
We combine the numerical score with comment sentiment analysis and cross-
reference against usage data to detect contradictions. A high NPS with
declining usage is MORE concerning than a low NPS with stable usage,
because the former suggests hidden churn intent.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def analyze_nps(nps_df: pd.DataFrame, account_id: int) -> Dict:
    """
    Analyze NPS data for a single account.
    
    Args:
        nps_df: Full NPS responses DataFrame
        account_id: The account to analyze
        
    Returns:
        Dictionary of NPS analysis results
    """
    acct = nps_df[nps_df["account_id"] == account_id]
    
    if len(acct) == 0:
        return _empty_nps_result()
    
    # Get the most recent NPS response (assuming one per account in this dataset)
    latest = acct.iloc[-1]
    
    score = int(latest["score"])
    comment = latest.get("verbatim_comment", "")
    category = str(latest.get("nps_category", "Unknown"))
    is_non_english = bool(latest.get("is_non_english", False))
    
    # Detect score-comment contradiction
    # A high score with negative language, or low score with positive language
    contradiction = _detect_contradiction(score, comment)
    
    return {
        "score": score,
        "category": category,
        "comment": comment,
        "is_non_english": is_non_english,
        "has_comment": bool(comment),
        "contradiction_detected": contradiction["is_contradiction"],
        "contradiction_type": contradiction["type"],
        "contradiction_detail": contradiction["detail"],
    }


def _detect_contradiction(score: int, comment: str) -> Dict:
    """
    Detect mismatches between NPS score and comment sentiment.
    
    This is a key non-obvious insight: automated NPS analysis often
    looks only at the score. But a customer who gives a 9 and says
    "watching competitors closely" is more at risk than a customer
    who gives a 5 and says nothing.
    
    Note: Full sentiment analysis is deferred to the LLM layer.
    This is a fast heuristic for initial flagging.
    """
    if not comment:
        return {"is_contradiction": False, "type": None, "detail": None}
    
    comment_lower = comment.lower()
    
    # Negative signals in comment
    negative_signals = [
        "downgrade", "competitor", "done", "frustrated", "disappointed",
        "expensive", "steep", "evaluating", "watching competitors",
        "fallen off", "wasted", "embarrassing", "lost faith",
        "no communication", "forever", "cliff"
    ]
    
    # Positive signals in comment
    positive_signals = [
        "love", "great", "best", "transformed", "phenomenal",
        "recommend", "easily", "won"
    ]
    
    has_negative = any(sig in comment_lower for sig in negative_signals)
    has_positive = any(sig in comment_lower for sig in positive_signals)
    
    # High score + negative comment = dangerous silent churn
    if score >= 7 and has_negative:
        return {
            "is_contradiction": True,
            "type": "positive_score_negative_comment",
            "detail": f"NPS {score} (Passive/Promoter) but comment contains risk signals",
        }
    
    # Low score + positive comment = potential survey fatigue or misclick
    if score <= 4 and has_positive:
        return {
            "is_contradiction": True,
            "type": "negative_score_positive_comment",
            "detail": f"NPS {score} (Detractor) but comment is positive — possible misclick or survey fatigue",
        }
    
    return {"is_contradiction": False, "type": None, "detail": None}


def _empty_nps_result() -> Dict:
    """Return empty result for accounts with no NPS data."""
    return {
        "score": None,
        "category": "No Response",
        "comment": "",
        "is_non_english": False,
        "has_comment": False,
        "contradiction_detected": False,
        "contradiction_type": None,
        "contradiction_detail": None,
    }


def compute_nps_risk_score(nps_result: Dict) -> Dict[str, float]:
    """
    Convert NPS analysis into normalized risk scores [0, 1].
    
    Returns:
        nps_detractor_risk: Based on raw score
        nps_deterioration_risk: Based on contradiction/context
    """
    score = nps_result.get("score")
    
    if score is None:
        # No NPS data is moderately concerning (disengaged customer)
        return {"nps_detractor_risk": 0.3, "nps_deterioration_risk": 0.2}
    
    # Detractor risk based on score
    if score <= 4:
        detractor_risk = 1.0
    elif score <= 6:
        detractor_risk = 0.7
    elif score <= 7:
        detractor_risk = 0.3
    elif score <= 8:
        detractor_risk = 0.1
    else:
        detractor_risk = 0.0
    
    # Deterioration risk based on contradiction
    deterioration_risk = 0.0
    if nps_result.get("contradiction_detected"):
        if nps_result["contradiction_type"] == "positive_score_negative_comment":
            # This is the most dangerous: looks healthy on the surface
            deterioration_risk = 0.8
        elif nps_result["contradiction_type"] == "negative_score_positive_comment":
            # Less dangerous: might just be a data quality issue
            deterioration_risk = 0.3
    
    return {
        "nps_detractor_risk": round(detractor_risk, 3),
        "nps_deterioration_risk": round(deterioration_risk, 3),
    }
