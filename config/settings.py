"""
Configuration settings for the Renewal Risk Intelligence Engine.

Design Decision: All tunable parameters are centralized here to avoid
magic numbers scattered across the codebase. Each weight and threshold
is documented with its business justification.
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict
from dotenv import load_dotenv

load_dotenv()

# ─── Project Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT  # CSV files are in the project root
CHART_OUTPUT_DIR = PROJECT_ROOT / "renewal_intelligence" / "output" / "charts"
CHART_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Date Configuration ─────────────────────────────────────────────────────
# "Today" is June 9, 2026, per the assignment context.
# In production, this would be datetime.now()
REFERENCE_DATE = datetime(2026, 6, 9)
RENEWAL_WINDOW_DAYS = 90  # Accounts renewing within this window are flagged
RENEWAL_CUTOFF = REFERENCE_DATE + timedelta(days=RENEWAL_WINDOW_DAYS)

# ─── Risk Scoring Weights ───────────────────────────────────────────────────
# Each weight reflects the relative importance of the signal in predicting
# renewal risk. Total weights sum to 1.0 for interpretability.
#
# Justifications:
# - usage_decline (0.20): Strongest behavioral signal. Usage is what customers
#   DO vs what they SAY. A >30% decline in API calls or active users over 3
#   months is the most reliable leading indicator. Research from Gainsight
#   and Totango consistently shows usage as #1 predictor.
#
# - competitor_mention (0.12): Active evaluation signals high intent to leave.
#   By the time a customer names competitors in a CSM call, they've already
#   done research. This is a late-stage signal but highly actionable.
#
# - p1_tickets (0.10): P1 tickets indicate production-impacting issues. Even
#   one unresolved P1 can destroy trust built over months. Weighted heavily
#   because these are escalation-worthy by definition.
#
# - nps_detractor (0.10): NPS ≤6 is an industry-standard risk signal. However,
#   it's attitudinal (not behavioral) and can be misleading (see Meridian
#   Health case). Moderate weight to avoid over-indexing on surveys.
#
# - open_tickets (0.08): Unresolved tickets compound frustration. Even P3/P4
#   tickets, when left open for weeks, signal organizational neglect.
#
# - executive_escalation (0.08): When VPs/CXOs join calls, it means the
#   decision has moved from operational to strategic. Weight reflects the
#   qualitative severity of this signal.
#
# - budget_concern (0.07): Financial constraints are a hard ceiling on
#   renewal. Even a satisfied customer can't renew if budget is cut.
#
# - champion_loss (0.06): Internal champions are often the only reason a
#   product survives vendor reviews. Losing them removes political cover.
#
# - nps_deterioration (0.05): A declining NPS trend is more concerning than
#   a static low score. It suggests worsening experience trajectory.
#
# - sdk_deprecation_risk (0.05): Accounts on deprecated SDKs face forced
#   migration pain. This is solvable with intervention, so weight is
#   moderate — it's a risk amplifier, not a standalone churn driver.
#
# - product_risk_impact (0.05): Changelog events (breaking changes,
#   deprecations) directly affecting the customer's implementation.
#
# - missed_qbrs (0.04): Disengagement signal. Not attending QBRs means
#   the customer isn't investing in the relationship. Lower weight because
#   it could also be simple scheduling.

RISK_WEIGHTS: Dict[str, float] = {
    "usage_decline": 0.20,
    "competitor_mention": 0.12,
    "p1_tickets": 0.10,
    "nps_detractor": 0.10,
    "open_tickets": 0.08,
    "executive_escalation": 0.08,
    "budget_concern": 0.07,
    "champion_loss": 0.06,
    "nps_deterioration": 0.05,
    "sdk_deprecation_risk": 0.05,
    "product_risk_impact": 0.05,
    "missed_qbrs": 0.04,
}

# Verify weights sum to 1.0 (within floating point tolerance)
assert abs(sum(RISK_WEIGHTS.values()) - 1.0) < 0.001, \
    f"Risk weights must sum to 1.0, got {sum(RISK_WEIGHTS.values())}"

# ─── Risk Tier Thresholds ───────────────────────────────────────────────────
# These thresholds determine risk classification.
# High: Immediate intervention needed (executive sponsor, discount, SA)
# Medium: Proactive outreach within 2 weeks
# Low: Routine monitoring, no special action
HIGH_RISK_THRESHOLD = 0.65
MEDIUM_RISK_THRESHOLD = 0.40

# ─── Usage Analysis Thresholds ───────────────────────────────────────────────
# Percentage decline thresholds for flagging usage concerns
USAGE_DECLINE_SEVERE = 0.40   # 40%+ decline = severe risk
USAGE_DECLINE_MODERATE = 0.20  # 20-40% decline = moderate concern
USAGE_DECLINE_MILD = 0.10     # 10-20% decline = early warning

# ─── SDK Version Risk ────────────────────────────────────────────────────────
# SDK versions mapped to risk levels based on changelog analysis
# v3.x: Deprecated, security patches end April 30, 2026 → HIGH RISK
# v4.0.0: Missed locale fallback fix, hit by v4.2.0 breaking change → MEDIUM
# v4.1.0: Missed v4.2.0 breaking change → LOW-MEDIUM
# v4.2.0+: Current, no version risk → NONE
DEPRECATED_SDK_VERSIONS = ["v3.1.2", "v3.2.0", "v3.4.1"]
RISKY_SDK_VERSIONS = ["v4.0.0"]  # Hit by locale fallback bug + breaking change

# ─── Fuzzy Match Configuration ───────────────────────────────────────────────
FUZZY_MATCH_THRESHOLD = 75  # Minimum score for account name matching

# ─── LLM API Configuration ──────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_TEMPERATURE = 0.1  # Low temp for consistent extraction

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_TEMPERATURE = 0.3  # Slightly higher for nuanced reasoning

# ─── Changelog Events ────────────────────────────────────────────────────────
# Structured representation of key changelog events for analysis
CHANGELOG_EVENTS = [
    {
        "version": "v4.3.2",
        "date": "2026-03-01",
        "type": "deprecation",
        "title": "REST API v2 sunset extended to April 30, 2026",
        "affects": "all_v3_sdk",
        "severity": "critical",
    },
    {
        "version": "v4.3.2",
        "date": "2026-03-01",
        "type": "deprecation",
        "title": "Legacy editor removal in v4.4.0 (May 2026)",
        "affects": "legacy_editor_users",
        "severity": "high",
    },
    {
        "version": "v4.3.2",
        "date": "2026-03-01",
        "type": "deprecation",
        "title": "SDK v3.x security patches end April 30, 2026",
        "affects": "all_v3_sdk",
        "severity": "critical",
    },
    {
        "version": "v4.2.0",
        "date": "2025-10-15",
        "type": "breaking_change",
        "title": "Response envelope change: entry → data",
        "affects": "sdk_below_4.2.0",
        "severity": "high",
    },
    {
        "version": "v4.2.0",
        "date": "2025-10-15",
        "type": "breaking_change",
        "title": "Webhook payload v2 default",
        "affects": "webhook_users",
        "severity": "medium",
    },
    {
        "version": "v4.3.0",
        "date": "2025-12-15",
        "type": "deprecation",
        "title": "Legacy Workflow Engine deprecated",
        "affects": "workflow_users",
        "severity": "medium",
    },
    {
        "version": "v4.2.3",
        "date": "2025-11-01",
        "type": "bugfix",
        "title": "Locale fallback null fix (v4.0.0 and v4.1.0)",
        "affects": "v4.0.0_v4.1.0",
        "severity": "medium",
    },
    {
        "version": "v4.3.2",
        "date": "2026-03-01",
        "type": "security",
        "title": "Privilege escalation patch (CVE-2026-1102)",
        "affects": "all_below_4.3.2",
        "severity": "critical",
    },
]
