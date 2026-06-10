"""
Risk Signal Extraction — Normalizes signals from all data sources.

Design Decision: Each signal source produces a normalized [0, 1] score
that represents the risk contribution from that dimension. Normalization
is critical because raw metrics have wildly different scales (NPS is 0-10,
tickets are counts, usage is percentage change). Without normalization,
signals with larger absolute values would dominate regardless of their
actual risk contribution.
"""

from typing import Dict, List, Optional
import pandas as pd

from renewal_intelligence.analysis.usage_analyzer import analyze_usage_trends, compute_usage_risk_score
from renewal_intelligence.analysis.ticket_analyzer import analyze_tickets, compute_ticket_risk_score
from renewal_intelligence.analysis.nps_analyzer import analyze_nps, compute_nps_risk_score
from renewal_intelligence.analysis.changelog_analyzer import analyze_product_risk
from renewal_intelligence.llm.groq_client import extract_csm_intelligence
from renewal_intelligence.config.settings import DEPRECATED_SDK_VERSIONS, RISKY_SDK_VERSIONS


def extract_all_signals(
    account_id: int,
    accounts_df: pd.DataFrame,
    usage_df: pd.DataFrame,
    tickets_df: pd.DataFrame,
    nps_df: pd.DataFrame,
    csm_notes: List[Dict],
) -> Dict:
    """
    Extract and normalize all risk signals for a single account.
    
    This is the signal aggregation layer — it pulls from every data source
    and produces a standardized signal dictionary that the scoring engine
    can consume.
    
    Args:
        account_id: The account to analyze
        accounts_df: Accounts DataFrame
        usage_df: Usage metrics DataFrame
        tickets_df: Support tickets DataFrame
        nps_df: NPS responses DataFrame
        csm_notes: Reconciled CSM notes (list of dicts)
        
    Returns:
        Dictionary with normalized signal scores and raw analysis
    """
    # Get account info
    account = accounts_df[accounts_df["account_id"] == account_id]
    if len(account) == 0:
        return _empty_signals()
    account_info = account.iloc[0]
    
    # ─── Usage Signals ────────────────────────────────────────────────────────
    usage_analysis = analyze_usage_trends(usage_df, account_id)
    usage_risk = compute_usage_risk_score(usage_analysis)
    
    # ─── Ticket Signals ───────────────────────────────────────────────────────
    ticket_analysis = analyze_tickets(tickets_df, account_id)
    ticket_risk = compute_ticket_risk_score(ticket_analysis)
    
    # ─── NPS Signals ──────────────────────────────────────────────────────────
    nps_analysis = analyze_nps(nps_df, account_id)
    nps_risk = compute_nps_risk_score(nps_analysis)
    
    # ─── Product Risk Signals ─────────────────────────────────────────────────
    product_risk = analyze_product_risk(usage_df, tickets_df, account_id)
    
    # ─── CSM Intelligence Signals ─────────────────────────────────────────────
    csm_signals = _extract_csm_signals(csm_notes, account_id, account_info)
    
    # ─── SDK Risk Signal ──────────────────────────────────────────────────────
    sdk_version = usage_analysis.get("sdk_version", "unknown")
    if sdk_version in DEPRECATED_SDK_VERSIONS:
        sdk_risk = 1.0
    elif sdk_version in RISKY_SDK_VERSIONS:
        sdk_risk = 0.6
    else:
        sdk_risk = 0.0
    
    return {
        # Normalized scores (0-1 scale) for the scoring engine
        "scores": {
            "usage_decline": usage_risk,
            "p1_tickets": ticket_risk["p1_risk"],
            "open_tickets": ticket_risk["open_ticket_risk"],
            "nps_detractor": nps_risk["nps_detractor_risk"],
            "nps_deterioration": nps_risk["nps_deterioration_risk"],
            "competitor_mention": csm_signals["competitor_risk"],
            "executive_escalation": csm_signals["executive_risk"],
            "budget_concern": csm_signals["budget_risk"],
            "champion_loss": csm_signals["champion_risk"],
            "missed_qbrs": csm_signals["qbr_risk"],
            "sdk_deprecation_risk": sdk_risk,
            "product_risk_impact": product_risk["product_risk_score"],
        },
        # Raw analysis for explanations
        "raw": {
            "usage": usage_analysis,
            "tickets": ticket_analysis,
            "nps": nps_analysis,
            "product_risk": product_risk,
            "csm": csm_signals,
        },
        # Account info
        "account": account_info.to_dict(),
    }


def _extract_csm_signals(
    csm_notes: List[Dict],
    account_id: int,
    account_info: pd.Series,
) -> Dict:
    """
    Extract risk signals from CSM notes for a specific account.
    
    Uses Groq for extraction when available, falls back to heuristics.
    """
    # Filter notes for this account
    acct_notes = [n for n in csm_notes if n.get("matched_account_id") == account_id]
    
    if not acct_notes:
        return _empty_csm_signals()
    
    # Combine all notes for this account
    all_text = "\n\n".join(n.get("raw_text", "") for n in acct_notes)
    
    # Extract intelligence using Groq (or heuristic fallback)
    intelligence = extract_csm_intelligence(
        note_text=all_text,
        account_name=account_info.get("account_name", "Unknown"),
        account_id=account_id,
        arr=account_info.get("arr", 0),
        plan_tier=account_info.get("plan_tier", "Unknown"),
        contract_end=str(account_info.get("contract_end_date", "")),
    )
    
    # Normalize signals to [0, 1]
    competitor_risk = min(1.0, len(intelligence.get("competitors", [])) * 0.5)
    executive_risk = 1.0 if intelligence.get("executive_involved", False) else 0.0
    budget_risk = 1.0 if intelligence.get("budget_concern", False) else 0.0
    
    # Champion risk
    champion_status = intelligence.get("champion_status", "unknown")
    if champion_status == "lost":
        champion_risk = 1.0
    elif champion_status == "at_risk":
        champion_risk = 0.7
    else:
        champion_risk = 0.0
    
    # QBR risk
    missed_qbrs = intelligence.get("missed_qbrs", 0)
    qbr_risk = min(1.0, missed_qbrs * 0.4)
    
    # Sentiment-based adjustment
    sentiment = intelligence.get("sentiment", "neutral")
    sentiment_modifier = {
        "positive": -0.1,
        "neutral": 0.0,
        "mixed": 0.2,
        "negative": 0.4,
    }.get(sentiment, 0.0)
    
    return {
        "competitor_risk": competitor_risk,
        "executive_risk": executive_risk,
        "budget_risk": budget_risk,
        "champion_risk": champion_risk,
        "qbr_risk": qbr_risk,
        "sentiment": sentiment,
        "sentiment_modifier": sentiment_modifier,
        "competitors": intelligence.get("competitors", []),
        "risk_factors": intelligence.get("risk_factors", []),
        "renewal_signal": intelligence.get("renewal_signal", "neutral"),
        "raw_intelligence": intelligence,
        "note_count": len(acct_notes),
    }


def _empty_csm_signals() -> Dict:
    """Return empty CSM signals for accounts with no notes."""
    return {
        "competitor_risk": 0.0,
        "executive_risk": 0.0,
        "budget_risk": 0.0,
        "champion_risk": 0.0,
        "qbr_risk": 0.0,
        "sentiment": "unknown",
        "sentiment_modifier": 0.0,
        "competitors": [],
        "risk_factors": [],
        "renewal_signal": "unknown",
        "raw_intelligence": {},
        "note_count": 0,
    }


def _empty_signals() -> Dict:
    """Return empty signals structure."""
    return {
        "scores": {key: 0.0 for key in [
            "usage_decline", "p1_tickets", "open_tickets",
            "nps_detractor", "nps_deterioration", "competitor_mention",
            "executive_escalation", "budget_concern", "champion_loss",
            "missed_qbrs", "sdk_deprecation_risk", "product_risk_impact",
        ]},
        "raw": {},
        "account": {},
    }
