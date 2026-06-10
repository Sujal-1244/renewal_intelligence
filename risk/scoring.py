"""
Risk Scoring Engine — Computes final risk scores and classifications.

Design Decision: The scoring uses a weighted linear combination of normalized
signals. We chose this over a more complex model (e.g., gradient boosting)
because:
1. Interpretability: Every weight is justified and auditable
2. No training data: We don't have labeled churn outcomes to train on
3. Domain expertise: BizOps teams need to understand AND trust the scores
4. Tuning: Weights can be adjusted based on CS team feedback

Alternative considered: Bayesian scoring with signal independence assumptions.
Rejected because signals are clearly NOT independent (e.g., high tickets
often correlate with usage decline). The weighted approach is more honest
about this limitation.
"""

import pandas as pd
from typing import Dict, List, Tuple

from renewal_intelligence.config.settings import (
    RISK_WEIGHTS,
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
    REFERENCE_DATE,
    RENEWAL_CUTOFF,
)
from renewal_intelligence.risk.signals import extract_all_signals


def compute_risk_score(signals: Dict) -> Tuple[float, str, Dict]:
    """
    Compute the final risk score from normalized signals.
    
    The score is a weighted sum of individual signal scores,
    each multiplied by its configured weight.
    
    Args:
        signals: Dictionary with 'scores' key containing normalized [0,1] values
        
    Returns:
        Tuple of (risk_score, risk_tier, score_breakdown)
    """
    scores = signals.get("scores", {})
    
    # Compute weighted sum
    total_score = 0.0
    breakdown = {}
    
    for signal_name, weight in RISK_WEIGHTS.items():
        signal_value = scores.get(signal_name, 0.0)
        contribution = signal_value * weight
        total_score += contribution
        
        breakdown[signal_name] = {
            "raw_score": round(signal_value, 3),
            "weight": weight,
            "contribution": round(contribution, 4),
        }
    
    # Clamp to [0, 1]
    total_score = max(0.0, min(1.0, total_score))
    
    # Classify risk tier
    if total_score >= HIGH_RISK_THRESHOLD:
        risk_tier = "High"
    elif total_score >= MEDIUM_RISK_THRESHOLD:
        risk_tier = "Medium"
    else:
        risk_tier = "Low"
    
    return (round(total_score, 4), risk_tier, breakdown)


def score_all_accounts(
    accounts_df: pd.DataFrame,
    usage_df: pd.DataFrame,
    tickets_df: pd.DataFrame,
    nps_df: pd.DataFrame,
    csm_notes: List[Dict],
    renewal_window_only: bool = True,
) -> pd.DataFrame:
    """
    Score all accounts and return a ranked DataFrame.
    
    Args:
        accounts_df: Accounts DataFrame
        usage_df: Usage metrics DataFrame
        tickets_df: Support tickets DataFrame
        nps_df: NPS responses DataFrame
        csm_notes: Reconciled CSM notes
        renewal_window_only: If True, only score accounts in the renewal window
        
    Returns:
        DataFrame with risk scores, tiers, and breakdowns
    """
    if renewal_window_only:
        target_accounts = accounts_df[accounts_df["in_renewal_window"] == True]
    else:
        target_accounts = accounts_df
    
    results = []
    
    for _, account in target_accounts.iterrows():
        account_id = account["account_id"]
        
        # Extract all signals
        signals = extract_all_signals(
            account_id=account_id,
            accounts_df=accounts_df,
            usage_df=usage_df,
            tickets_df=tickets_df,
            nps_df=nps_df,
            csm_notes=csm_notes,
        )
        
        # Compute risk score
        risk_score, risk_tier, breakdown = compute_risk_score(signals)
        
        # Get top risk drivers
        top_drivers = _get_top_drivers(breakdown)
        
        results.append({
            "account_id": account_id,
            "account_name": account["account_name"],
            "arr": account["arr"],
            "plan_tier": account["plan_tier"],
            "industry": account["industry"],
            "region": account["region"],
            "csm_name": account["csm_name"],
            "contract_end_date": account["contract_end_date"],
            "days_to_renewal": account["days_to_renewal"],
            "risk_score": risk_score,
            "risk_tier": risk_tier,
            "top_risk_drivers": top_drivers,
            "score_breakdown": breakdown,
            "signals": signals,
        })
    
    # Create DataFrame and sort by risk score descending
    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.sort_values("risk_score", ascending=False).reset_index(drop=True)
    
    return results_df


def _get_top_drivers(breakdown: Dict, top_n: int = 5) -> List[Dict]:
    """
    Identify the top N risk drivers by contribution.
    
    This is what makes the scores explainable — users can see
    exactly which factors are driving the risk assessment.
    """
    drivers = []
    for signal_name, details in breakdown.items():
        if details["contribution"] > 0:
            drivers.append({
                "signal": signal_name,
                "contribution": details["contribution"],
                "raw_score": details["raw_score"],
                "weight": details["weight"],
                "label": _signal_label(signal_name),
            })
    
    # Sort by contribution descending
    drivers.sort(key=lambda x: x["contribution"], reverse=True)
    
    return drivers[:top_n]


def _signal_label(signal_name: str) -> str:
    """Human-readable label for a signal name."""
    labels = {
        "usage_decline": "📉 Usage Decline",
        "competitor_mention": "🏢 Competitor Evaluation",
        "p1_tickets": "🔴 P1 Support Tickets",
        "nps_detractor": "😤 NPS Detractor Score",
        "open_tickets": "📋 Unresolved Tickets",
        "executive_escalation": "👔 Executive Involvement",
        "budget_concern": "💰 Budget Constraints",
        "champion_loss": "🏆 Champion at Risk",
        "nps_deterioration": "📊 NPS Deterioration",
        "sdk_deprecation_risk": "⚠️ SDK Deprecation",
        "product_risk_impact": "🔧 Product Risk Impact",
        "missed_qbrs": "📅 Missed QBRs",
    }
    return labels.get(signal_name, signal_name)
