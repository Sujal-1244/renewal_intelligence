"""
Changelog Intelligence Analyzer — Links product changes to customer risk.

Design Decision: The changelog is included in the dataset for a reason that
the assignment doesn't explain. The insight is: product changes (deprecations,
breaking changes, migrations) create involuntary churn risk. A customer who
is perfectly happy can be forced to churn if a product change breaks their
implementation and they can't or won't migrate.

This module:
1. Parses changelog events into structured data
2. Maps SDK versions to affected accounts
3. Identifies accounts hit by deprecations, breaking changes, and bugs
4. Creates a "Product Risk Impact" score per account

This is a key differentiator — most churn models only look at customer
behavior, not product-caused risk.
"""

import pandas as pd
from typing import Dict, List, Set
from datetime import datetime

from renewal_intelligence.config.settings import (
    CHANGELOG_EVENTS,
    DEPRECATED_SDK_VERSIONS,
    RISKY_SDK_VERSIONS,
    REFERENCE_DATE,
)


def analyze_product_risk(
    usage_df: pd.DataFrame,
    tickets_df: pd.DataFrame,
    account_id: int
) -> Dict:
    """
    Determine how much product-side risk affects a specific account.
    
    Strategy:
    1. Check their SDK version against deprecation timelines
    2. Check if their tickets reference changelog events
    3. Compute a Product Risk Impact score
    
    Args:
        usage_df: Usage metrics (contains sdk_version)
        tickets_df: Support tickets (may reference changelog issues)
        account_id: The account to analyze
        
    Returns:
        Dictionary with product risk assessment
    """
    # Get the account's current SDK version (latest month)
    acct_usage = usage_df[usage_df["account_id"] == account_id].sort_values("month_dt")
    
    if len(acct_usage) == 0:
        return _empty_product_risk()
    
    current_sdk = acct_usage.iloc[-1]["sdk_version"]
    
    # ─── SDK Deprecation Risk ────────────────────────────────────────────────
    sdk_risks = []
    is_deprecated = current_sdk in DEPRECATED_SDK_VERSIONS
    is_risky = current_sdk in RISKY_SDK_VERSIONS
    
    if is_deprecated:
        sdk_risks.append({
            "type": "sdk_deprecated",
            "severity": "critical",
            "detail": (
                f"Running {current_sdk} — SDK v3.x is deprecated. "
                f"Security patches ended April 30, 2026. "
                f"REST API v2 endpoints sunset. "
                f"Must migrate to v4.2.3+ immediately."
            ),
        })
    
    if is_risky:
        sdk_risks.append({
            "type": "sdk_risky_version",
            "severity": "high",
            "detail": (
                f"Running {current_sdk} — hit by locale fallback bug "
                f"(fixed in v4.2.3) and affected by v4.2.0 breaking change "
                f"(response.entry → response.data). Upgrade recommended."
            ),
        })
    
    # Check for specific version-related changelog impacts
    version_impacts = _check_version_impacts(current_sdk)
    sdk_risks.extend(version_impacts)
    
    # ─── Ticket-Changelog Correlation ─────────────────────────────────────────
    acct_tickets = tickets_df[tickets_df["account_id"] == account_id]
    changelog_ticket_links = _link_tickets_to_changelog(acct_tickets)
    
    # ─── Compute Product Risk Score ──────────────────────────────────────────
    product_risk_score = _compute_product_risk_score(
        is_deprecated=is_deprecated,
        is_risky=is_risky,
        sdk_risks=sdk_risks,
        changelog_ticket_links=changelog_ticket_links,
    )
    
    return {
        "current_sdk": current_sdk,
        "is_deprecated_sdk": is_deprecated,
        "is_risky_sdk": is_risky,
        "sdk_risks": sdk_risks,
        "changelog_ticket_links": changelog_ticket_links,
        "product_risk_score": round(product_risk_score, 3),
        "affected_events": [e for e in CHANGELOG_EVENTS if _event_affects_version(e, current_sdk)],
    }


def _check_version_impacts(sdk_version: str) -> List[Dict]:
    """Check which changelog events affect this SDK version."""
    impacts = []
    
    # v3.x users: affected by REST API v2 sunset + security patch end
    if sdk_version.startswith("v3"):
        impacts.append({
            "type": "rest_api_sunset",
            "severity": "critical",
            "detail": "REST Content Delivery API v2 sunset April 30, 2026",
        })
    
    # v4.0.0 and v4.1.0: affected by breaking change in v4.2.0
    if sdk_version in ("v4.0.0", "v4.1.0"):
        impacts.append({
            "type": "breaking_change_impact",
            "severity": "medium",
            "detail": "SDK v4.2.0 changed response envelope (entry → data). Must update API integrations.",
        })
    
    # Users below v4.2.3: missing locale fallback fix
    major_minor = sdk_version.replace("v", "").split(".")
    try:
        version_tuple = tuple(int(x) for x in major_minor)
        if version_tuple < (4, 2, 3):
            impacts.append({
                "type": "locale_fallback_bug",
                "severity": "medium",
                "detail": "Missing locale fallback null fix (available in v4.2.3)",
            })
    except ValueError:
        pass
    
    return impacts


def _link_tickets_to_changelog(tickets: pd.DataFrame) -> List[Dict]:
    """
    Identify tickets that are directly caused by changelog events.
    
    This reveals the causal chain:
    Product Change → Support Ticket → Customer Frustration → Churn Risk
    """
    links = []
    
    changelog_patterns = {
        "SDK upgrade guidance": "sdk_deprecation",
        "REST API deprecation migration": "rest_api_sunset",
        "Migration from legacy editor": "legacy_editor_removal",
        "Locale fallback not working": "locale_fallback_bug",
        "Workflow automation broken": "workflow_deprecation",
        "New editor crash": "new_editor_issues",
        "Custom field rendering issue": "custom_field_xss_patch",
    }
    
    for _, ticket in tickets.iterrows():
        subject = ticket.get("subject", "")
        for pattern, event in changelog_patterns.items():
            if pattern.lower() in subject.lower():
                links.append({
                    "ticket_id": ticket.get("ticket_id", ""),
                    "subject": subject,
                    "changelog_event": event,
                    "priority": ticket.get("priority", ""),
                    "status": ticket.get("status", ""),
                })
                break
    
    return links


def _event_affects_version(event: Dict, sdk_version: str) -> bool:
    """Check if a changelog event affects the given SDK version."""
    affects = event.get("affects", "")
    
    if affects == "all_v3_sdk" and sdk_version.startswith("v3"):
        return True
    if affects == "sdk_below_4.2.0" and sdk_version in ("v3.1.2", "v3.2.0", "v3.4.1", "v4.0.0", "v4.1.0"):
        return True
    if affects == "v4.0.0_v4.1.0" and sdk_version in ("v4.0.0", "v4.1.0"):
        return True
    if affects == "legacy_editor_users":
        return True  # Can't determine from SDK version alone
    if affects == "workflow_users":
        return True  # Can't determine from SDK version alone
    
    return False


def _compute_product_risk_score(
    is_deprecated: bool,
    is_risky: bool,
    sdk_risks: List[Dict],
    changelog_ticket_links: List[Dict],
) -> float:
    """
    Compute a normalized product risk score [0, 1].
    
    This captures risk that originates from product decisions,
    not customer behavior. Even a perfectly happy customer is
    at risk if their SDK is about to be sunset.
    """
    score = 0.0
    
    # Deprecated SDK: major risk
    if is_deprecated:
        score += 0.5
    
    # Risky SDK version: moderate risk
    if is_risky:
        score += 0.3
    
    # Each critical SDK risk adds to the score
    critical_risks = [r for r in sdk_risks if r.get("severity") == "critical"]
    score += len(critical_risks) * 0.1
    
    # Tickets linked to changelog events = evidence of impact
    if changelog_ticket_links:
        # More linked tickets = more evidence of product-caused pain
        score += min(0.3, len(changelog_ticket_links) * 0.05)
    
    return min(1.0, score)


def _empty_product_risk() -> Dict:
    """Return empty product risk for accounts with no usage data."""
    return {
        "current_sdk": "unknown",
        "is_deprecated_sdk": False,
        "is_risky_sdk": False,
        "sdk_risks": [],
        "changelog_ticket_links": [],
        "product_risk_score": 0.0,
        "affected_events": [],
    }
