"""
Renewal Risk Intelligence Engine — Streamlit Application
Main entry point for the interactive dashboard.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from renewal_intelligence.data.loader import load_all_data
from renewal_intelligence.data.reconciler import reconcile_csm_notes
from renewal_intelligence.risk.scoring import score_all_accounts
from renewal_intelligence.visualization.charts import (
    create_risk_distribution_chart,
    create_arr_at_risk_chart,
    create_usage_trend_chart,
    create_ticket_timeline_chart,
    create_risk_score_breakdown_chart,
    create_portfolio_heatmap,
    create_nps_distribution_chart,
    create_sdk_version_chart,
)
from renewal_intelligence.llm.gemini_client import (
    generate_risk_explanation,
    generate_portfolio_insights,
)
from renewal_intelligence.config.settings import REFERENCE_DATE

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Renewal Risk Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp { font-family: 'Inter', sans-serif; }
    
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; color: #94A3B8; margin-top: 4px; }
    
    .risk-high { color: #EF4444; }
    .risk-medium { color: #F59E0B; }
    .risk-low { color: #10B981; }
    
    .account-card {
        background: #1E293B;
        border-left: 4px solid;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .account-card.high { border-color: #EF4444; }
    .account-card.medium { border-color: #F59E0B; }
    .account-card.low { border-color: #10B981; }
    
    .insight-box {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 100%);
        border: 1px solid #4338CA;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
    
    div[data-testid="stSidebar"] { background: #0F172A; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #1E293B;
        border-radius: 8px;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=600)
def load_and_process_data():
    """Load all data, reconcile entities, and compute risk scores."""
    data = load_all_data()
    reconciled_notes = reconcile_csm_notes(data["csm_notes"], data["accounts"])
    results = score_all_accounts(
        accounts_df=data["accounts"],
        usage_df=data["usage"],
        tickets_df=data["tickets"],
        nps_df=data["nps"],
        csm_notes=reconciled_notes,
        renewal_window_only=False,
    )
    return data, reconciled_notes, results


def main():
    # ─── Header ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px 0;">
        <h1 style="font-size:2.5rem; background: linear-gradient(90deg, #6366F1, #EC4899);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        🔮 Renewal Risk Intelligence Engine</h1>
        <p style="color:#94A3B8; font-size:1.1rem;">
        AI-powered renewal risk scoring • Powered by Groq + Gemini</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    with st.spinner("🔄 Loading data & computing risk scores..."):
        data, reconciled_notes, results_df = load_and_process_data()

    # Filter to renewal window
    renewal_df = results_df[results_df["days_to_renewal"].between(-30, 90)].copy()
    
    # ─── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Filters")
        
        show_all = st.toggle("Show all accounts", value=False,
                            help="Toggle to see all 120 accounts vs only renewal window")
        
        display_df = results_df if show_all else renewal_df
        
        tier_filter = st.multiselect(
            "Risk Tier", ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"]
        )
        display_df = display_df[display_df["risk_tier"].isin(tier_filter)]
        
        plan_filter = st.multiselect(
            "Plan Tier",
            sorted(display_df["plan_tier"].unique()),
            default=sorted(display_df["plan_tier"].unique()),
        )
        display_df = display_df[display_df["plan_tier"].isin(plan_filter)]
        
        st.markdown("---")
        st.markdown(f"**Reference Date:** {REFERENCE_DATE.strftime('%B %d, %Y')}")
        st.markdown(f"**Accounts in view:** {len(display_df)}")
        st.markdown(f"**Renewal window:** {len(renewal_df)} accounts")

    # ─── Tabs ────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Portfolio Dashboard",
        "🔍 Account Deep Dive",
        "🧠 AI Insights",
        "📋 Changelog Impact",
    ])

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1: PORTFOLIO DASHBOARD
    # ═══════════════════════════════════════════════════════════════════════════
    with tab1:
        _render_portfolio_dashboard(display_df, renewal_df, data)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2: ACCOUNT DEEP DIVE
    # ═══════════════════════════════════════════════════════════════════════════
    with tab2:
        _render_account_deep_dive(display_df, data, reconciled_notes)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 3: AI INSIGHTS
    # ═══════════════════════════════════════════════════════════════════════════
    with tab3:
        _render_ai_insights(display_df, renewal_df, data, reconciled_notes)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 4: CHANGELOG IMPACT
    # ═══════════════════════════════════════════════════════════════════════════
    with tab4:
        _render_changelog_impact(display_df, data)


def _render_portfolio_dashboard(display_df, renewal_df, data):
    """Render the portfolio overview dashboard."""
    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    high_risk = display_df[display_df["risk_tier"] == "High"]
    med_risk = display_df[display_df["risk_tier"] == "Medium"]
    
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value risk-high">{len(high_risk)}</div>
            <div class="metric-label">High Risk Accounts</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value risk-medium">{len(med_risk)}</div>
            <div class="metric-label">Medium Risk Accounts</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        arr_at_risk = high_risk["arr"].sum() + med_risk["arr"].sum()
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#EC4899;">${arr_at_risk:,.0f}</div>
            <div class="metric-label">Total ARR at Risk</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        avg_score = display_df["risk_score"].mean() if len(display_df) > 0 else 0
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#6366F1;">{avg_score:.2f}</div>
            <div class="metric-label">Avg Risk Score</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts row
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(create_risk_distribution_chart(display_df), use_container_width=True)
    with c2:
        st.plotly_chart(create_arr_at_risk_chart(display_df), use_container_width=True)
    
    # Heatmap
    st.plotly_chart(create_portfolio_heatmap(display_df), use_container_width=True)
    
    # Risk table
    st.markdown("### 📋 Risk-Scored Account List")
    table_df = display_df[[
        "account_name", "risk_tier", "risk_score", "arr",
        "plan_tier", "days_to_renewal", "csm_name", "region"
    ]].copy()
    table_df["arr"] = table_df["arr"].apply(lambda x: f"${x:,.0f}")
    table_df["risk_score"] = table_df["risk_score"].apply(lambda x: f"{x:.3f}")
    table_df.columns = ["Account", "Risk", "Score", "ARR", "Plan", "Days to Renewal", "CSM", "Region"]
    
    st.dataframe(
        table_df,
        use_container_width=True,
        height=400,
        column_config={
            "Risk": st.column_config.TextColumn(width="small"),
            "Score": st.column_config.TextColumn(width="small"),
        },
    )


def _render_account_deep_dive(display_df, data, reconciled_notes):
    """Render the individual account analysis view."""
    if len(display_df) == 0:
        st.info("No accounts match current filters.")
        return
    
    # Account selector
    account_options = {
        f"{row['account_name']} ({row['risk_tier']}) — Score: {row['risk_score']:.3f}": row['account_id']
        for _, row in display_df.iterrows()
    }
    
    selected_label = st.selectbox("Select Account", list(account_options.keys()))
    selected_id = account_options[selected_label]
    
    account_row = display_df[display_df["account_id"] == selected_id].iloc[0]
    signals = account_row.get("signals", {})
    raw = signals.get("raw", {})
    
    # Account header
    tier = account_row["risk_tier"]
    tier_class = tier.lower()
    st.markdown(f"""
    <div class="account-card {tier_class}">
        <h2>{account_row['account_name']}</h2>
        <p><strong>Risk: <span class="risk-{tier_class}">{tier}</span></strong> | 
        Score: {account_row['risk_score']:.3f} | 
        ARR: ${account_row['arr']:,.0f} | 
        Plan: {account_row['plan_tier']} | 
        Renewal: {account_row['days_to_renewal']} days |
        CSM: {account_row['csm_name']} | Region: {account_row['region']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Score breakdown chart
    breakdown = account_row.get("score_breakdown", {})
    st.plotly_chart(create_risk_score_breakdown_chart(breakdown), use_container_width=True)
    
    # Usage and Ticket charts
    c1, c2 = st.columns(2)
    with c1:
        usage_data = raw.get("usage", {}).get("monthly_data", [])
        st.plotly_chart(
            create_usage_trend_chart(usage_data, account_row["account_name"]),
            use_container_width=True
        )
    with c2:
        ticket_records = raw.get("tickets", {}).get("ticket_records", [])
        st.plotly_chart(
            create_ticket_timeline_chart(ticket_records, account_row["account_name"]),
            use_container_width=True
        )
    
    # Details columns
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("#### 📈 Usage Summary")
        usage = raw.get("usage", {})
        severity = usage.get("decline_severity", "N/A")
        decline = usage.get("overall_decline_pct", 0)
        sdk = usage.get("sdk_version", "N/A")
        st.markdown(f"- **Decline Severity:** {severity}")
        st.markdown(f"- **Overall Change:** {decline*100:.1f}%")
        st.markdown(f"- **SDK Version:** {sdk}")
    
    with c2:
        st.markdown("#### 🎫 Ticket Summary")
        tickets = raw.get("tickets", {})
        st.markdown(f"- **Total Tickets:** {tickets.get('total_tickets', 0)}")
        st.markdown(f"- **P1 Tickets:** {tickets.get('p1_count', 0)}")
        st.markdown(f"- **Open/Escalated:** {tickets.get('open_count', 0)}")
        st.markdown(f"- **Recurring Issues:** {tickets.get('recurring_count', 0)}")
    
    with c3:
        st.markdown("#### 📊 NPS & CSM Intel")
        nps = raw.get("nps", {})
        csm = raw.get("csm", {})
        st.markdown(f"- **NPS Score:** {nps.get('score', 'N/A')}")
        st.markdown(f"- **Category:** {nps.get('category', 'N/A')}")
        if nps.get("contradiction_detected"):
            st.warning(f"⚠️ {nps.get('contradiction_detail', '')}")
        if csm.get("competitors"):
            st.error(f"🏢 Competitors: {', '.join(csm['competitors'])}")
    
    # CSM Notes
    acct_notes = [n for n in reconciled_notes if n.get("matched_account_id") == selected_id]
    if acct_notes:
        st.markdown("#### 📝 CSM Notes")
        for note in acct_notes:
            st.markdown(f"""<div class="insight-box">
                <small>Match: {note.get('match_method', 'N/A')} 
                (confidence: {note.get('match_confidence', 0)}%)</small>
                <p>{note.get('raw_text', '')}</p>
            </div>""", unsafe_allow_html=True)
    
    # AI Explanation
    st.markdown("#### 🤖 AI Risk Explanation")
    if st.button("Generate AI Explanation", key=f"explain_{selected_id}"):
        with st.spinner("Generating analysis with Gemini..."):
            explanation = generate_risk_explanation(
                account_name=account_row["account_name"],
                account_id=selected_id,
                plan_tier=account_row["plan_tier"],
                arr=account_row["arr"],
                region=account_row["region"],
                contract_end=str(account_row["contract_end_date"]),
                days_to_renewal=account_row["days_to_renewal"],
                csm_name=account_row["csm_name"],
                usage_summary=f"Decline: {usage.get('decline_severity','N/A')}, Change: {usage.get('overall_decline_pct',0)*100:.1f}%",
                ticket_summary=f"Total: {tickets.get('total_tickets',0)}, P1: {tickets.get('p1_count',0)}, Open: {tickets.get('open_count',0)}",
                nps_summary=f"Score: {nps.get('score','N/A')}, Category: {nps.get('category','N/A')}",
                csm_intelligence=str(csm.get("risk_factors", [])),
                product_risk_summary=f"SDK: {sdk}, Deprecated: {raw.get('product_risk',{}).get('is_deprecated_sdk',False)}",
                risk_score=account_row["risk_score"],
                risk_tier=tier,
            )
            st.markdown(explanation)


def _render_ai_insights(display_df, renewal_df, data, reconciled_notes):
    """Render the AI-powered insights page."""
    st.markdown("### 🧠 Non-Obvious Insights")
    st.markdown("*These insights surface patterns that simple rule-based systems would miss.*")
    
    # Insight 1: Silent Churn
    st.markdown("""<div class="insight-box">
        <h4>🔇 Insight 1: Silent Churn — The Friendly Goodbye</h4>
        <p><strong>Meridian Health (1003)</strong> — NPS 8 ("Great support team") but usage 
        has cratered ~35%. CSM notes reveal they're building a homegrown replacement middleware. 
        The NPS score reflects relationship warmth with the support team, NOT intent to renew.</p>
        <p><em>Why rules miss this:</em> Any NPS ≥ 7 passes standard health checks. 
        Only cross-referencing NPS comment sentiment + usage trend + CSM intelligence reveals the disconnect.</p>
    </div>""", unsafe_allow_html=True)
    
    # Insight 2: Product-Caused Churn Cluster
    st.markdown("""<div class="insight-box">
        <h4>⚡ Insight 2: SDK Deprecation Churn Cluster</h4>
        <p>Accounts on SDK v3.x (1000-1007, others) face forced migration after the April 30 deadline. 
        NovaTech (1002) spent 200+ hours on workarounds and mentioned Strapi & Sanity. 
        This isn't customer dissatisfaction — it's <strong>product-caused friction</strong> 
        creating a simultaneous churn cluster.</p>
        <p><em>Why rules miss this:</em> Without linking the changelog deprecation → SDK version → 
        ticket subjects → CSM notes, each account looks like isolated frustration.</p>
    </div>""", unsafe_allow_html=True)
    
    # Insight 3: NPS-Score Contradiction
    st.markdown("""<div class="insight-box">
        <h4>🎭 Insight 3: Contradictory Signals — Survey Fatigue</h4>
        <p><strong>Summit Analytics (1019)</strong> — NPS 3 (Detractor) but comment says 
        "Great developer experience and the support team is phenomenal." 
        Similarly, <strong>Amber Trail (1041)</strong> has NPS 2 but says "Best headless CMS on the market."</p>
        <p>These are likely survey misclicks or fatigue, NOT genuine detractors. 
        Treating them as high risk wastes CS bandwidth on happy customers.</p>
    </div>""", unsafe_allow_html=True)
    
    # Insight 4: M&A Driven Churn
    st.markdown("""<div class="insight-box">
        <h4>🏢 Insight 4: External Event Risk — M&A and Procurement Reviews</h4>
        <p><strong>Orion Education (1009)</strong> is merging with a WordPress shop. 
        <strong>Evergreen Media (1015)</strong>'s parent was acquired. Both have healthy usage 
        but face existential platform decisions beyond their control.</p>
        <p><em>Why rules miss this:</em> No metric captures M&A risk. Only CSM notes 
        contain this intelligence, and only if correctly matched to accounts.</p>
    </div>""", unsafe_allow_html=True)

    # Insight 5: Champion-at-risk
    st.markdown("""<div class="insight-box">
        <h4>🏆 Insight 5: Champion Loss Despite Healthy Metrics</h4>
        <p><strong>Vanguard Retail (1005)</strong> — their ops manager, previously 
        "our biggest champion," has "lost faith in the roadmap" after a workflow 
        automation bug sat open for 6 weeks. Usage decline is severe (73%), 
        but the real risk is political: the internal advocate has turned hostile.</p>
    </div>""", unsafe_allow_html=True)
    
    # Portfolio Insights
    st.markdown("---")
    st.markdown("### 📊 Portfolio-Level AI Analysis")
    if st.button("Generate Portfolio Insights (Gemini)"):
        with st.spinner("Analyzing portfolio with Gemini..."):
            high = display_df[display_df["risk_tier"] == "High"]
            med = display_df[display_df["risk_tier"] == "Medium"]
            low = display_df[display_df["risk_tier"] == "Low"]
            
            top_accounts = "\n".join([
                f"- {r['account_name']}: Score {r['risk_score']:.3f}, ARR ${r['arr']:,}, {r['plan_tier']}"
                for _, r in high.head(10).iterrows()
            ])
            
            insights = generate_portfolio_insights(
                total_accounts=len(display_df),
                high_risk_count=len(high),
                high_risk_arr=high["arr"].sum(),
                medium_risk_count=len(med),
                medium_risk_arr=med["arr"].sum(),
                low_risk_count=len(low),
                top_risk_accounts=top_accounts,
                portfolio_signals="SDK deprecation cluster, champion losses, competitor evaluations",
                product_risk_events="SDK v3.x sunset April 30, Legacy editor removal May 2026",
            )
            st.markdown(insights)

    # NPS and SDK charts
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(create_nps_distribution_chart(data["nps"]), use_container_width=True)
    with c2:
        st.plotly_chart(create_sdk_version_chart(data["usage"]), use_container_width=True)


def _render_changelog_impact(display_df, data):
    """Render the changelog impact analysis page."""
    st.markdown("### 📋 Product Risk Impact — Changelog Intelligence")
    st.markdown("""
    This analysis links product changelog events to customer risk. 
    Product changes (deprecations, breaking changes, forced migrations) create 
    **involuntary churn risk** that traditional churn models miss.
    """)
    
    from renewal_intelligence.config.settings import CHANGELOG_EVENTS
    
    # Display changelog events
    st.markdown("#### ⚡ Key Changelog Events")
    for event in CHANGELOG_EVENTS:
        severity_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡"}.get(event["severity"], "⚪")
        st.markdown(f"""
        {severity_icon} **{event['title']}** ({event['version']}, {event['date']})  
        Type: `{event['type']}` | Affects: `{event['affects']}` | Severity: `{event['severity']}`
        """)
    
    st.markdown("---")
    
    # SDK Version Risk Table
    st.markdown("#### 🔧 Accounts on Deprecated/Risky SDK Versions")
    
    at_risk_accounts = []
    for _, row in display_df.iterrows():
        signals = row.get("signals", {})
        raw = signals.get("raw", {})
        product = raw.get("product_risk", {})
        
        if product.get("is_deprecated_sdk") or product.get("is_risky_sdk"):
            at_risk_accounts.append({
                "Account": row["account_name"],
                "SDK": product.get("current_sdk", "?"),
                "Risk": "🔴 Deprecated" if product.get("is_deprecated_sdk") else "🟠 Risky",
                "ARR": f"${row['arr']:,.0f}",
                "Renewal": f"{row['days_to_renewal']} days",
                "Linked Tickets": len(product.get("changelog_ticket_links", [])),
            })
    
    if at_risk_accounts:
        st.dataframe(pd.DataFrame(at_risk_accounts), use_container_width=True)
        
        total_arr = sum(row["arr"] for _, row in display_df.iterrows() 
                       if row.get("signals",{}).get("raw",{}).get("product_risk",{}).get("is_deprecated_sdk"))
        st.error(f"💰 **${total_arr:,.0f} ARR** on deprecated SDK versions requiring immediate migration support")
    else:
        st.success("✅ No accounts on deprecated SDKs in current view")


if __name__ == "__main__":
    main()
