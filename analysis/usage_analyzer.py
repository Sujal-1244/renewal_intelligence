"""
Usage Trend Analyzer — Detects usage decline patterns and health signals.

Design Decision: We compute both absolute and relative decline metrics
because a 50% decline from 100 API calls is noise, but a 50% decline from
100,000 API calls is a five-alarm fire. We use the last 3 months vs first
3 months comparison (half-window) to capture sustained trends rather than
one-off dips.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

from renewal_intelligence.config.settings import (
    USAGE_DECLINE_SEVERE,
    USAGE_DECLINE_MODERATE,
    USAGE_DECLINE_MILD,
)


def analyze_usage_trends(usage_df: pd.DataFrame, account_id: int) -> Dict:
    """
    Analyze usage trends for a single account over the 6-month window.
    
    Metrics computed:
    - api_calls_trend: % change first 3 months avg vs last 3 months avg
    - active_users_trend: Same for active users
    - content_creation_trend: Same for content entries
    - workflows_trend: Same for workflow triggers
    - overall_decline_pct: Weighted average of all metrics
    - decline_severity: 'severe' | 'moderate' | 'mild' | 'stable' | 'growing'
    
    Args:
        usage_df: Full usage metrics DataFrame
        account_id: The account to analyze
        
    Returns:
        Dictionary of trend analysis results
    """
    acct = usage_df[usage_df["account_id"] == account_id].sort_values("month_dt")
    
    if len(acct) < 4:
        return _empty_trend_result()
    
    # Split into first half (Oct-Dec 2025) and second half (Jan-Mar 2026)
    midpoint = len(acct) // 2
    first_half = acct.iloc[:midpoint]
    second_half = acct.iloc[midpoint:]
    
    # Compute trend for each metric
    metrics = {
        "api_calls": _compute_trend(first_half["api_calls"], second_half["api_calls"]),
        "active_users": _compute_trend(first_half["active_users"], second_half["active_users"]),
        "content_entries": _compute_trend(
            first_half["content_entries_created"],
            second_half["content_entries_created"]
        ),
        "workflows": _compute_trend(
            first_half["workflows_triggered"],
            second_half["workflows_triggered"]
        ),
    }
    
    # Weighted overall decline (API calls matter most for SaaS usage)
    weights = {"api_calls": 0.40, "active_users": 0.30, "content_entries": 0.20, "workflows": 0.10}
    overall_decline = sum(
        metrics[k]["pct_change"] * weights[k] for k in weights
    )
    
    # Month-over-month trend for the most recent 3 months
    recent_mom = _compute_mom_trend(acct)
    
    # Determine severity
    abs_decline = abs(min(overall_decline, 0))
    if abs_decline >= USAGE_DECLINE_SEVERE:
        severity = "severe"
    elif abs_decline >= USAGE_DECLINE_MODERATE:
        severity = "moderate"
    elif abs_decline >= USAGE_DECLINE_MILD:
        severity = "mild"
    elif overall_decline >= 0:
        severity = "growing"
    else:
        severity = "stable"
    
    # SDK version (from latest month)
    sdk_version = acct.iloc[-1]["sdk_version"] if "sdk_version" in acct.columns else "unknown"
    
    return {
        "metrics": metrics,
        "overall_decline_pct": round(overall_decline, 4),
        "decline_severity": severity,
        "mom_trends": recent_mom,
        "sdk_version": sdk_version,
        "latest_month_data": acct.iloc[-1].to_dict() if len(acct) > 0 else {},
        "first_month_data": acct.iloc[0].to_dict() if len(acct) > 0 else {},
        "monthly_data": acct.to_dict("records"),
    }


def _compute_trend(first_half: pd.Series, second_half: pd.Series) -> Dict:
    """Compute percentage change between two halves of a time series."""
    first_avg = first_half.mean()
    second_avg = second_half.mean()
    
    if first_avg == 0:
        pct_change = 0.0 if second_avg == 0 else 1.0
    else:
        pct_change = (second_avg - first_avg) / first_avg
    
    return {
        "first_half_avg": round(first_avg, 1),
        "second_half_avg": round(second_avg, 1),
        "pct_change": round(pct_change, 4),
        "direction": "up" if pct_change > 0.05 else "down" if pct_change < -0.05 else "flat",
    }


def _compute_mom_trend(acct_df: pd.DataFrame) -> List[Dict]:
    """Compute month-over-month changes for all months."""
    trends = []
    for i in range(1, len(acct_df)):
        prev = acct_df.iloc[i - 1]
        curr = acct_df.iloc[i]
        
        api_change = (
            (curr["api_calls"] - prev["api_calls"]) / prev["api_calls"]
            if prev["api_calls"] > 0 else 0
        )
        
        trends.append({
            "month": curr["month"],
            "api_calls_change": round(api_change, 4),
            "active_users_change": curr["active_users"] - prev["active_users"],
        })
    
    return trends


def _empty_trend_result() -> Dict:
    """Return empty result for accounts with insufficient data."""
    return {
        "metrics": {},
        "overall_decline_pct": 0.0,
        "decline_severity": "insufficient_data",
        "mom_trends": [],
        "sdk_version": "unknown",
        "latest_month_data": {},
        "first_month_data": {},
        "monthly_data": [],
    }


def compute_usage_risk_score(trend_result: Dict) -> float:
    """
    Convert usage trend analysis into a normalized risk score [0, 1].
    
    Score interpretation:
    - 0.0: Usage is growing or stable → no risk
    - 0.5: Moderate decline → watch closely
    - 1.0: Severe, sustained decline → critical risk
    
    We use a sigmoid-like mapping to avoid extreme sensitivity to small changes
    while still catching dramatic declines.
    """
    decline = trend_result.get("overall_decline_pct", 0)
    severity = trend_result.get("decline_severity", "stable")
    
    if severity in ("growing", "stable"):
        return 0.0
    elif severity == "mild":
        return 0.3
    elif severity == "moderate":
        return 0.6
    elif severity == "severe":
        # Scale based on how severe: 40% decline = 0.7, 60%+ = 1.0
        return min(1.0, 0.7 + abs(decline) * 0.5)
    
    return 0.0
