"""
Prompt Templates for LLM Integration.

Design Decision: All prompts are centralized here for:
1. Easy iteration and A/B testing
2. Consistent formatting across the pipeline
3. Clear separation of concerns (prompts ≠ business logic)

Each prompt is designed with:
- Clear role definition
- Structured output format
- Guard rails against hallucination
- Token efficiency (avoid verbose instructions)
"""


# ─── GROQ PROMPTS (Fast Extraction) ─────────────────────────────────────────

CSM_NOTE_EXTRACTION_PROMPT = """You are an expert at extracting structured intelligence from messy customer success manager (CSM) call notes.

Analyze the following CSM note and extract structured data. Be precise and conservative — only flag signals you are confident about.

CSM NOTE:
{note_text}

ACCOUNT CONTEXT:
- Account Name: {account_name}
- Account ID: {account_id}
- ARR: ${arr:,}
- Plan: {plan_tier}
- Contract End: {contract_end}

Respond with ONLY valid JSON in this exact schema:
{{
  "sentiment": "positive|neutral|negative|mixed",
  "competitors": ["list of competitor names mentioned, e.g. Hygraph, Contentful, Strapi, Sanity, Kontent.ai, builder.io, WordPress"],
  "risk_factors": ["list of specific risk factors identified"],
  "executive_involved": false,
  "budget_concern": false,
  "renewal_signal": "positive|negative|neutral|mixed",
  "champion_status": "active|at_risk|lost|unknown",
  "missed_qbrs": 0,
  "action_items": ["list of urgent action items"],
  "key_quote": "most important quote from the note"
}}

Rules:
- For competitors, only include explicitly named alternatives (not general mentions of "evaluation")
- For executive_involved, return true if any VP, CXO, CTO, CRO, CISO, or Director-level person is mentioned
- For budget_concern, return true if budget cuts, cost reduction, or pricing complaints are mentioned
- For champion_status, assess if the internal champion is active, at risk of leaving, or has left
- For missed_qbrs, count QBRs missed or marked as no-shows
- For key_quote, extract the most revealing direct or indirect quote"""


TRANSLATE_COMMENT_PROMPT = """Translate the following customer feedback comment to English. 
Preserve the original tone and sentiment. Also analyze the sentiment.

Comment: {comment}

Respond with ONLY valid JSON:
{{
  "translation": "English translation here",
  "original_language": "language name",
  "sentiment": "positive|neutral|negative|mixed",
  "risk_signals": ["any risk factors detected in the comment"]
}}"""


BATCH_EXTRACTION_PROMPT = """You are an expert analyst extracting risk signals from multiple CSM notes.

Analyze these notes and provide a consolidated view of risk signals across the portfolio.

NOTES:
{notes_text}

Respond with ONLY valid JSON:
{{
  "high_risk_accounts": [
    {{
      "account_name": "name",
      "risk_level": "high|medium|low",
      "key_risks": ["list"],
      "competitors_mentioned": ["list"],
      "executive_involved": true/false
    }}
  ],
  "portfolio_themes": ["recurring themes across notes"],
  "urgent_actions": ["most time-sensitive items"]
}}"""


# ─── GEMINI PROMPTS (Deep Reasoning) ────────────────────────────────────────

RENEWAL_RISK_EXPLANATION_PROMPT = """You are a senior Customer Success strategist at a B2B SaaS company. 
Your job is to analyze renewal risk for a specific account and provide actionable intelligence.

ACCOUNT: {account_name} (ID: {account_id})
Plan: {plan_tier} | ARR: ${arr:,} | Region: {region}
Contract End: {contract_end} | Days to Renewal: {days_to_renewal}
CSM: {csm_name}

─── USAGE TRENDS ───
{usage_summary}

─── SUPPORT TICKETS ───
{ticket_summary}

─── NPS DATA ───
{nps_summary}

─── CSM NOTES (AI-Extracted Intelligence) ───
{csm_intelligence}

─── PRODUCT RISK (Changelog Impact) ───
{product_risk_summary}

─── COMPUTED RISK SCORE ───
Overall Risk Score: {risk_score}/1.0
Risk Tier: {risk_tier}

INSTRUCTIONS:
As a senior CS strategist, provide:

1. **Risk Assessment** (2-3 sentences): What is the real story with this account? Go beyond the numbers. What's the underlying dynamic?

2. **Key Risk Drivers** (bullet points): List the top 3-5 risk factors, ordered by severity. For each, explain WHY it matters, not just WHAT it is.

3. **Non-Obvious Insight** (1-2 sentences): What signal would a junior analyst miss? What contradiction or hidden pattern do you see?

4. **Recommended Actions** (numbered list): Concrete steps the account team should take in the next 2 weeks. Be specific — "schedule a call" is useless. Say "schedule a call with [role] to discuss [topic] because [reason]."

5. **Confidence Level**: How confident are you in this assessment? What data would change your mind?

Keep your response concise but insightful. Write for a VP of Customer Success who needs to triage 40 accounts in 30 minutes.
"""


SILENT_CHURN_DETECTION_PROMPT = """You are an expert at detecting "silent churn" — customers who appear healthy on surface metrics but are quietly disengaging or building alternatives.

Analyze this account for silent churn indicators:

ACCOUNT: {account_name}
NPS Score: {nps_score} | NPS Comment: "{nps_comment}"

USAGE TREND:
- API calls change: {api_trend}%
- Active users change: {user_trend}%
- Content creation change: {content_trend}%

CSM INTELLIGENCE:
{csm_intelligence}

SIGNALS TO LOOK FOR:
- Positive NPS but declining usage (the "friendly goodbye" pattern)
- Stable usage but building middleware/alternatives
- Champion still engaged but org is moving on
- High usage concentrated in fewer users (dependency risk)
- Data export API requests (migration signal)
- Questions about contract terms or exit clauses

Respond with ONLY valid JSON:
{{
  "is_silent_churn": true/false,
  "confidence": 0.0-1.0,
  "pattern_type": "name of the pattern if detected",
  "evidence": ["specific evidence points"],
  "counter_evidence": ["reasons this might NOT be silent churn"],
  "recommended_investigation": "what to look into next"
}}"""


PORTFOLIO_INSIGHTS_PROMPT = """You are a VP of Customer Success reviewing the quarterly renewal portfolio.

PORTFOLIO SUMMARY:
- Total accounts in renewal window: {total_accounts}
- High risk: {high_risk_count} (${high_risk_arr:,} ARR at risk)
- Medium risk: {medium_risk_count} (${medium_risk_arr:,} ARR at risk)
- Low risk: {low_risk_count}

TOP RISK ACCOUNTS:
{top_risk_accounts}

PORTFOLIO-WIDE SIGNALS:
{portfolio_signals}

PRODUCT RISK EVENTS:
{product_risk_events}

Provide:
1. **Executive Summary** (3-4 sentences for the board): What's the overall health of this quarter's renewals?
2. **Top 3 Portfolio Themes**: What systemic issues are driving risk across multiple accounts?
3. **SDK Deprecation Impact**: How many accounts are affected and what's the ARR exposure?
4. **Recommended Portfolio Actions**: What should the CS leadership team prioritize?
5. **Prediction**: Based on these signals, what's your expected renewal rate for this quarter?"""


MULTIMODAL_ANALYSIS_PROMPT = """You are analyzing visual representations of customer health metrics alongside structured data.

ACCOUNT: {account_name}
Current Risk Tier: {risk_tier}

STRUCTURED METRICS:
{structured_data}

You are viewing charts showing:
1. Usage trend over 6 months (API calls, active users, content creation)
2. Support ticket timeline and severity distribution
3. Account health scorecard

Based on BOTH the visual patterns and structured data, provide:

1. **Visual Pattern Analysis**: What trends do you see in the charts that the numbers alone might not reveal? (e.g., acceleration of decline, seasonal patterns, cliff drops vs gradual erosion)

2. **Trajectory Assessment**: Where is this account headed in the next 90 days based on the visual trajectory?

3. **Anomaly Detection**: Any visual outliers or pattern breaks that warrant investigation?

Respond concisely — focus on insights that visual analysis adds BEYOND what the numbers already tell us."""
