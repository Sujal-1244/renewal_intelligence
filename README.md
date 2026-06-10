# 🔮 Renewal Risk Intelligence Engine

A production-quality AI-powered prototype that identifies at-risk customer renewals, explains why they're at risk, and recommends actions — before the account team has to ask.

## Detailed Project Explanation
The Renewal Risk Intelligence Engine is designed to solve a critical problem for B2B SaaS BizOps teams: identifying true churn risk hidden within conflicting signals. Standard rule-based systems often fail because they treat metrics in isolation. This engine uses a multi-modal, dual-LLM approach to synthesize structured usage data, unstructured CSM notes, support tickets, and NPS responses into a single cohesive narrative. 

By analyzing the "why" behind the data—such as detecting "silent churn" where NPS is high but product usage has cratered, or "product-induced churn" caused by forced SDK deprecations—the engine empowers the account team with actionable insights rather than just raw dashboards.

## Architecture

### Clean Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         PRESENTATION LAYER                                   │
│                                                                              │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐  ┌───────┐   │
│  │ Risk Summary    │  │ Account Drill-   │  │ Data Quality  │  │Export │   │
│  │ (Table + KPIs)  │  │ Down (Signals)   │  │ Report        │  │ (CSV) │   │
│  └────────┬────────┘  └────────┬─────────┘  └───────┬───────┘  └───┬───┘   │
│           │                    │                     │              │        │
│           └────────────────────┴─────────────────────┴──────────────┘        │
│                                  │                                           │
│                    ┌─────────────▼───────────────┐                          │
│                    │   pages/               │                          │
│                    │   app.py (Streamlit)  │                          │
│                    └──────────┬──────────────┘                          │
└──────────────────────────────┼────────────────────────────────────────┘
                               │
┌──────────────────────────────┼────────────────────────────────────────┐
│                    ORCHESTRATION LAYER                              │
│                                                                      │
│    ┌─────────────────────────────────────────────────────────────┐  │
│    │  main.py / runner.py (Pipeline Orchestration)            │  │
│    │  ├─ Load data                                             │  │
│    │  ├─ Reconcile accounts                                   │  │
│    │  ├─ Run 4 parallel analyzers                            │  │
│    │  ├─ Extract LLM signals                                 │  │
│    │  ├─ Compute risk scores                                 │  │
│    │  └─ Render dashboard                                    │  │
│    └──────────────────────────────────────────────────────────┘  │
│                                                                      │
└──────────────────────────────┬────────────────────────────────────────┘
                               │
┌──────────────────────────────┼────────────────────────────────────────┐
│               BUSINESS LOGIC / DOMAIN LAYER                           │
│                                                                        │
│  ┌─────────────────────────┐  ┌─────────────────────────────────┐   │
│  │     ANALYSIS LAYER      │  │      RISK SCORING LAYER         │   │
│  │                         │  │                                 │   │
│  │ • usage_analyzer        │  │ • signals.py (normalize)       │   │
│  │   (regression trends)   │  │   (all signals → 0-1 range)    │   │
│  │                         │  │                                 │   │
│  │ • ticket_analyzer       │  │ • scoring.py (weighted sum)    │   │
│  │   (P1, open, SLA)      │  │   (12 weights × signals)        │   │
│  │                         │  │                                 │   │
│  │ • nps_analyzer          │  │ • explanations (signal          │   │
│  │   (score + validation)  │  │   breakdown, recommendations)   │   │
│  │                         │  │                                 │   │
│  │ • changelog_analyzer    │  │ WEIGHTS JUSTIFY EACH SIGNAL:   │   │
│  │   (deprecation risk)    │  │ ├─ Usage Decline: 20%         │   │
│  │                         │  │ ├─ Competitor: 12%            │   │
│  │                         │  │ ├─ P1 Tickets: 10%            │   │
│  │                         │  │ ├─ NPS Detractor: 10%         │   │
│  │                         │  │ ├─ Open Tickets: 8%           │   │
│  │                         │  │ ├─ Exec Escalation: 8%        │   │
│  │                         │  │ ├─ Budget Concern: 7%         │   │
│  │                         │  │ ├─ Champion Loss: 6%          │   │
│  │                         │  │ ├─ NPS Deterioration: 5%      │   │
│  │                         │  │ ├─ SDK Deprecation: 5%        │   │
│  │                         │  │ ├─ Product Risk: 5%           │   │
│  │                         │  │ └─ Missed QBRs: 4%            │   │
│  │                         │  │                                 │   │
│  └─────────────────────────┘  └─────────────────────────────────┘   │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │            LLM INTELLIGENCE LAYER (Dual-LLM)              │   │
│  │                                                             │   │
│  │  GROQ (Fast Extraction):          GEMINI (Reasoning):     │   │
│  │  ├─ Competitor mentions          ├─ Signal narratives    │   │
│  │  ├─ Budget concerns              ├─ Risk explanations    │   │
│  │  ├─ Org changes                  ├─ Multimodal charts    │   │
│  │  └─ Sentiment extraction         └─ Action recommendations
│  │                                                             │   │
│  │  FALLBACK: Heuristic extraction (no API keys)             │   │
│  │                                                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                        │
└──────────────────────────────┬────────────────────────────────────────┘
                               │
┌──────────────────────────────┼────────────────────────────────────────┐
│                 DATA ACCESS & TRANSFORMATION LAYER                    │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  data/loader.py                                             │   │
│  │  ├─ Parse 6 CSV/TXT/MD files                               │   │
│  │  ├─ Type coercion (dates, numbers)                        │   │
│  │  ├─ Normalize account names                               │   │
│  │  └─ Handle missing values                                 │   │
│  └──────────────────┬───────────────────────────────────────┘   │
│                     │                                             │
│  ┌──────────────────▼───────────────────────────────────────┐   │
│  │  data/reconciler.py                                     │   │
│  │  ├─ Fuzzy match CSM notes to account IDs (75% threshold)│   │
│  │  ├─ Link garbled names to actual accounts              │   │
│  │  └─ Output: reconciliation confidence scores            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                        │
└──────────────────────────────┬────────────────────────────────────────┘
                               │
┌──────────────────────────────┼────────────────────────────────────────┐
│                      DATA LAYER (Immutable)                           │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  data/raw/ (6 Data Sources - Read Only)                    │   │
│  │                                                             │   │
│  │  Structured Data:                  Unstructured Data:      │   │
│  │  ├─ accounts.csv (120 rows)        ├─ csm_notes.txt       │   │
│  │  ├─ usage_metrics.csv (6 months)  │  (120+ notes)        │   │
│  │  ├─ support_tickets.csv (~500)    └─ changelog.md        │   │
│  │  └─ nps_responses.csv (120+)         (product releases)   │   │
│  │                                                             │   │
│  │  Data Quality: ~94% reconciliation success rate           │   │
│  │                                                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
renewal_intelligence/
├── app.py                      # Streamlit main app (4-tab dashboard)
├── config/settings.py          # Centralized config with justified weights
├── data/
│   ├── loader.py               # Data ingestion & cleaning (all 6 sources)
│   └── reconciler.py           # Fuzzy matching with RapidFuzz
├── analysis/
│   ├── usage_analyzer.py       # 6-month trend analysis with severity scoring
│   ├── ticket_analyzer.py      # Multi-dimensional ticket risk assessment
│   ├── nps_analyzer.py         # NPS with contradiction detection
│   └── changelog_analyzer.py   # Product Risk Impact (changelog ↔ tickets)
├── llm/
│   ├── groq_client.py          # Groq/Llama 3.3 70B — extraction tasks
│   ├── gemini_client.py        # Gemini 2.5 Flash — reasoning + multimodal
│   └── prompts.py              # All prompt templates (centralized)
├── risk/
│   ├── signals.py              # Signal normalization [0,1] from all sources
│   └── scoring.py              # Weighted scoring engine with explainability
├── visualization/
│   └── charts.py               # Plotly charts (dark theme, multimodal-ready)
├── pages/                      # Streamlit sub-pages (drill-down views)
└── data/
    ├── raw/                    # 6 data sources (read-only)
    └── processed/              # Intermediate outputs (cached)
```

### Layer Responsibilities

| Layer | Responsibility | Key Files |
|-------|-----------------|-----------|
| **Presentation** | User interface, dashboards, visualizations | `app.py`, `pages/` |
| **Orchestration** | Pipeline coordination, data flow | `main.py` (runner) |
| **Business Logic** | Risk analysis, signal extraction, scoring | `analysis/`, `risk/` |
| **LLM Intelligence** | AI-powered extraction & reasoning | `llm/` |
| **Data Access** | Ingestion, cleaning, reconciliation | `data/loader.py`, `data/reconciler.py` |
| **Data Storage** | Raw sources (immutable) | `data/raw/` |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API keys (optional — app works without them via fallback heuristics)
cp .env.example .env
# Edit .env with your Groq and Gemini API keys

# 3. Run
export GROQ_API_KEY="your_key"    # Optional
export GEMINI_API_KEY="your_key"  # Optional
streamlit run renewal_intelligence/app.py
```

## Approach & Key Decisions

### 1. Risk Scoring Methodology
Weighted linear combination of 12 normalized signals. Each weight is justified:

| Signal | Weight | Rationale |
|--------|--------|-----------|
| Usage Decline | 0.20 | Behavioral signal > attitudinal. What customers DO matters most. |
| Competitor Mention | 0.12 | Active evaluation = late-stage intent to leave |
| P1 Tickets | 0.10 | Production impact erodes trust fastest |
| NPS Detractor | 0.10 | Industry-validated but can mislead (see insight #1) |
| Open Tickets | 0.08 | Unresolved issues compound frustration |
| Executive Escalation | 0.08 | C-suite involvement = strategic decision imminent |
| Budget Concern | 0.07 | Financial constraint is a hard ceiling |
| Champion Loss | 0.06 | Internal advocates are renewal insurance |
| NPS Deterioration | 0.05 | Trend matters more than absolute score |
| SDK Deprecation | 0.05 | Solvable with intervention, but amplifies other risks |
| Product Risk | 0.05 | Changelog events affecting the customer's stack |
| Missed QBRs | 0.04 | Disengagement signal |

### 2. LLM Architecture (Dual-LLM)
- **Groq (Llama 3.3 70B)**: Fast extraction — sentiment, competitors, entities, translation. Low temperature (0.1) for consistency.
- **Gemini 2.5 Flash**: Deep reasoning — risk explanations, silent churn detection, portfolio insights, multimodal chart analysis.

Both gracefully degrade to heuristic fallbacks when API keys aren't set.

### 3. Non-Obvious Insights
1. **Silent Churn**: Meridian Health (NPS 8) building homegrown replacement. Score reflects relationship warmth, not renewal intent.
2. **SDK Deprecation Cluster**: Product changelog creates simultaneous forced-migration pain across 8+ accounts.
3. **NPS Contradiction / Survey Fatigue**: Summit Analytics (NPS 3) with comment "Great developer experience" — likely misclick.
4. **M&A-Driven Risk**: Orion Education merging with WordPress company. No metric captures this — only CSM notes.
5. **Champion-at-Risk**: Vanguard Retail's biggest champion "lost faith" after 6-week unresolved bug.

### 4. Changelog Intelligence
The `changelog.md` is parsed to identify deprecations, breaking changes, and migrations. These are linked to:
- Customer SDK versions (from `usage_metrics.csv`)
- Support ticket subjects (identifying product-caused tickets)
- CSM notes (connecting complaints to root causes)

This reveals **product-caused churn risk** — a dimension most churn models miss entirely.

### 5. Multimodal Architecture
Charts rendered by Plotly are saved as PNGs and passed to Gemini Vision alongside structured data. Gemini analyzes visual patterns (acceleration of decline, cliff drops vs gradual erosion) that numbers alone don't capture.

## Tradeoffs Made

| Decision | Tradeoff | Alternative |
|----------|----------|-------------|
| Weighted linear scoring | Interpretable but can't capture non-linear interactions | Gradient boosting (needs labeled data we don't have) |
| Heuristic fallbacks | Works without API keys but less accurate | Hard requirement on API keys |
| Single-page Streamlit | Simpler deployment but all tabs load at once | Multi-page app with lazy loading |
| Fuzzy match at 75% threshold | May miss very garbled names | Lower threshold risks false matches |

## What I'd Do With More Time
1. **Feedback loop**: Let CSMs flag incorrect risk assessments to calibrate weights
2. **Slack/email alerts**: Push notifications for risk tier changes
3. **Historical backtesting**: Score past quarters against actual churn outcomes
4. **RAG pipeline**: Index all CSM notes + tickets for semantic search
5. **Real-time scoring**: Webhook-triggered re-scoring when new data arrives
6. **A/B testing prompts**: Track which Gemini explanations CSMs find most useful

## Production Roadmap
1. **Data pipeline**: Replace CSV files with Salesforce/Gainsight API integrations
2. **Auth**: SSO integration for BizOps team access control
3. **Database**: PostgreSQL for scored results + audit trail
4. **Monitoring**: LLM cost tracking, latency dashboards, prompt drift detection
5. **CI/CD**: Automated testing of risk scoring logic + prompt regression tests

## Managing the Application (Start & Stop)

### How to Start the Project
To run the Streamlit dashboard locally, use the following command in your terminal from the root directory of the project. This will start the server and typically open the application at `http://localhost:8501`.

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/streamlit run renewal_intelligence/app.py
```

### How to Stop the Project
To stop the running Streamlit server, go to the terminal window where the server is actively running and press the following keyboard shortcut:

```bash
Ctrl + C
```
This will safely terminate the process and shut down the dashboard.
