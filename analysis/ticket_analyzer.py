"""
Support Ticket Analyzer — Extracts risk signals from ticket history.

Design Decision: We look at multiple dimensions of ticket health:
1. Volume: Are tickets increasing? (frustration building)
2. Severity: Are there P1s? (production impact)
3. Resolution: Are tickets sitting unresolved? (trust erosion)
4. Patterns: Recurring issues? (systemic problems)

Each dimension contributes independently to risk because they represent
different failure modes. High volume P4s ≠ one unresolved P1.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import timedelta

from renewal_intelligence.config.settings import REFERENCE_DATE


def analyze_tickets(tickets_df: pd.DataFrame, account_id: int) -> Dict:
    """
    Analyze support ticket patterns for a single account.
    
    Args:
        tickets_df: Full support tickets DataFrame
        account_id: The account to analyze
        
    Returns:
        Dictionary of ticket analysis metrics and risk signals
    """
    acct = tickets_df[tickets_df["account_id"] == account_id]
    
    if len(acct) == 0:
        return _empty_ticket_result()
    
    # ─── Volume Analysis ─────────────────────────────────────────────────────
    total_tickets = len(acct)
    
    # Recent vs older ticket comparison (last 3 months vs prior)
    three_months_ago = REFERENCE_DATE - timedelta(days=90)
    recent = acct[acct["created_date"] >= three_months_ago]
    older = acct[acct["created_date"] < three_months_ago]
    
    recent_count = len(recent)
    older_count = len(older)
    
    # Volume trend
    if older_count > 0:
        volume_change = (recent_count - older_count) / older_count
    else:
        volume_change = 0.0 if recent_count == 0 else 1.0
    
    # ─── Severity Analysis ────────────────────────────────────────────────────
    p1_count = len(acct[acct["priority"] == "P1"])
    p2_count = len(acct[acct["priority"] == "P2"])
    recent_p1 = len(recent[recent["priority"] == "P1"]) if len(recent) > 0 else 0
    
    # ─── Resolution Analysis ─────────────────────────────────────────────────
    open_tickets = acct[acct["status"].isin(["Open", "Escalated"])]
    open_count = len(open_tickets)
    escalated_count = len(acct[acct["status"] == "Escalated"])
    
    # Average resolution time (for resolved tickets)
    resolved = acct[acct["resolution_time_hours"].notna()]
    avg_resolution_hours = resolved["resolution_time_hours"].mean() if len(resolved) > 0 else None
    
    # ─── Pattern Analysis ─────────────────────────────────────────────────────
    # Identify recurring issues (tickets with "recurring" in description)
    recurring = acct[acct["description"].str.contains("recurring", case=False, na=False)]
    recurring_count = len(recurring)
    
    # Identify blocking issues
    blocking = acct[acct["description"].str.contains("blocking", case=False, na=False)]
    blocking_count = len(blocking)
    
    # ─── Topic Analysis ──────────────────────────────────────────────────────
    subject_counts = acct["subject"].value_counts().to_dict()
    
    # Identify tickets related to changelog events
    changelog_related = _identify_changelog_tickets(acct)
    
    return {
        "total_tickets": total_tickets,
        "recent_tickets": recent_count,
        "older_tickets": older_count,
        "volume_change": round(volume_change, 3),
        "p1_count": p1_count,
        "p2_count": p2_count,
        "recent_p1_count": recent_p1,
        "open_count": open_count,
        "escalated_count": escalated_count,
        "avg_resolution_hours": round(avg_resolution_hours, 1) if avg_resolution_hours else None,
        "recurring_count": recurring_count,
        "blocking_count": blocking_count,
        "subject_breakdown": subject_counts,
        "changelog_related_tickets": changelog_related,
        "ticket_records": acct.to_dict("records"),
    }


def _identify_changelog_tickets(tickets: pd.DataFrame) -> List[Dict]:
    """
    Link tickets to specific changelog events.
    
    This is a key insight: product changes cause support tickets,
    and those tickets affect renewal risk. Connecting the dots
    reveals product-caused churn patterns.
    """
    changelog_keywords = {
        "sdk_deprecation": ["SDK upgrade", "REST API deprecation", "sdk upgrade guidance"],
        "legacy_editor": ["Migration from legacy editor", "new editor crash"],
        "locale_fallback": ["Locale fallback"],
        "workflow_broken": ["Workflow automation broken"],
        "breaking_change": ["REST API deprecation migration"],
    }
    
    related = []
    for _, ticket in tickets.iterrows():
        subject = ticket.get("subject", "")
        description = ticket.get("description", "")
        combined = f"{subject} {description}".lower()
        
        for event_type, keywords in changelog_keywords.items():
            for keyword in keywords:
                if keyword.lower() in combined:
                    related.append({
                        "ticket_id": ticket.get("ticket_id", ""),
                        "event_type": event_type,
                        "subject": subject,
                        "priority": ticket.get("priority", ""),
                        "status": ticket.get("status", ""),
                    })
                    break
    
    return related


def _empty_ticket_result() -> Dict:
    """Return empty result for accounts with no tickets."""
    return {
        "total_tickets": 0,
        "recent_tickets": 0,
        "older_tickets": 0,
        "volume_change": 0.0,
        "p1_count": 0,
        "p2_count": 0,
        "recent_p1_count": 0,
        "open_count": 0,
        "escalated_count": 0,
        "avg_resolution_hours": None,
        "recurring_count": 0,
        "blocking_count": 0,
        "subject_breakdown": {},
        "changelog_related_tickets": [],
        "ticket_records": [],
    }


def compute_ticket_risk_score(ticket_result: Dict) -> Dict[str, float]:
    """
    Convert ticket analysis into normalized risk scores [0, 1].
    
    Returns separate scores for P1 impact and open ticket burden
    because they represent different risk dimensions.
    """
    # P1 risk: Any P1 is concerning, multiple P1s are alarming
    p1_score = min(1.0, ticket_result["p1_count"] * 0.3)
    
    # Recent P1s are worse (recency bias is appropriate here)
    if ticket_result["recent_p1_count"] > 0:
        p1_score = min(1.0, p1_score + 0.2 * ticket_result["recent_p1_count"])
    
    # Open ticket risk: Unresolved issues compound
    open_score = min(1.0, ticket_result["open_count"] * 0.15)
    
    # Escalations amplify risk
    if ticket_result["escalated_count"] > 0:
        open_score = min(1.0, open_score + 0.1 * ticket_result["escalated_count"])
    
    return {
        "p1_risk": round(p1_score, 3),
        "open_ticket_risk": round(open_score, 3),
    }
