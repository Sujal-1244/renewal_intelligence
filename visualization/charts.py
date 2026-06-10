"""
Visualization Module — Plotly charts for the Streamlit dashboard.

Design Decision: All charts use a consistent dark theme with Contentstack's
brand-adjacent color palette. Charts are designed to be both informative
AND suitable for multimodal analysis (Gemini Vision inspection).

For multimodal: Charts are saved as PNG when needed for Gemini analysis.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List
from pathlib import Path

from renewal_intelligence.config.settings import CHART_OUTPUT_DIR

# ─── Color Palette ───────────────────────────────────────────────────────────
COLORS = {
    "high_risk": "#EF4444",      # Red
    "medium_risk": "#F59E0B",    # Amber
    "low_risk": "#10B981",       # Green
    "primary": "#6366F1",        # Indigo
    "secondary": "#8B5CF6",      # Violet
    "accent": "#EC4899",         # Pink
    "bg_dark": "#0F172A",        # Slate 900
    "bg_card": "#1E293B",        # Slate 800
    "text": "#F1F5F9",           # Slate 100
    "text_muted": "#94A3B8",     # Slate 400
    "grid": "#334155",           # Slate 700
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "info": "#3B82F6",
}

RISK_COLORS = {
    "High": COLORS["high_risk"],
    "Medium": COLORS["medium_risk"],
    "Low": COLORS["low_risk"],
}

# Common layout settings for all charts
BASE_LAYOUT = dict(
    plot_bgcolor=COLORS["bg_dark"],
    paper_bgcolor=COLORS["bg_dark"],
    font=dict(color=COLORS["text"], family="Inter, sans-serif"),
    margin=dict(l=40, r=40, t=60, b=40),
    xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
    yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
)


def create_risk_distribution_chart(results_df: pd.DataFrame) -> go.Figure:
    """
    Create a donut chart showing risk tier distribution.
    
    This is the hero visualization on the dashboard — it gives
    an instant portfolio health snapshot.
    """
    tier_counts = results_df["risk_tier"].value_counts()
    
    # Ensure all tiers are represented
    for tier in ["High", "Medium", "Low"]:
        if tier not in tier_counts:
            tier_counts[tier] = 0
    
    fig = go.Figure(data=[go.Pie(
        labels=tier_counts.index.tolist(),
        values=tier_counts.values.tolist(),
        hole=0.55,
        marker=dict(colors=[RISK_COLORS.get(t, COLORS["text_muted"]) for t in tier_counts.index]),
        textinfo="label+value",
        textfont=dict(size=14, color="white"),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    )])
    
    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="Renewal Risk Distribution", font=dict(size=20)),
        showlegend=True,
        legend=dict(font=dict(size=12)),
        height=400,
    )
    
    # Add center text
    total = tier_counts.sum()
    fig.add_annotation(
        text=f"<b>{total}</b><br>Accounts",
        x=0.5, y=0.5,
        font=dict(size=20, color=COLORS["text"]),
        showarrow=False,
    )
    
    return fig


def create_arr_at_risk_chart(results_df: pd.DataFrame) -> go.Figure:
    """
    Create a horizontal bar chart showing ARR at risk by tier.
    
    This is the money chart — literally. It shows the dollar impact
    of each risk tier, which is what the CFO cares about.
    """
    tier_arr = results_df.groupby("risk_tier")["arr"].sum().reindex(["High", "Medium", "Low"], fill_value=0)
    
    fig = go.Figure(data=[go.Bar(
        y=tier_arr.index.tolist(),
        x=tier_arr.values.tolist(),
        orientation="h",
        marker=dict(
            color=[RISK_COLORS.get(t, COLORS["text_muted"]) for t in tier_arr.index],
            line=dict(width=0),
        ),
        text=[f"${v:,.0f}" for v in tier_arr.values],
        textposition="outside",
        textfont=dict(size=14, color=COLORS["text"]),
    )])
    
    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="ARR at Risk by Tier", font=dict(size=20)),
        xaxis_title="Annual Recurring Revenue ($)",
        height=300,
    )
    
    return fig


def create_usage_trend_chart(usage_data: List[Dict], account_name: str) -> go.Figure:
    """
    Create a multi-line chart showing usage trends over 6 months.
    
    Uses dual y-axis: API calls (left) and Active Users (right).
    This chart is designed to be visually clear for both humans AND
    Gemini Vision analysis.
    """
    if not usage_data:
        return _empty_chart("No usage data available")
    
    df = pd.DataFrame(usage_data)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # API Calls (primary y-axis)
    fig.add_trace(
        go.Scatter(
            x=df["month"],
            y=df["api_calls"],
            name="API Calls",
            line=dict(color=COLORS["primary"], width=3),
            fill="tozeroy",
            fillcolor="rgba(99, 102, 241, 0.1)",
            mode="lines+markers",
        ),
        secondary_y=False,
    )
    
    # Active Users (secondary y-axis)
    fig.add_trace(
        go.Scatter(
            x=df["month"],
            y=df["active_users"],
            name="Active Users",
            line=dict(color=COLORS["accent"], width=3, dash="dot"),
            mode="lines+markers",
        ),
        secondary_y=True,
    )
    
    # Content Entries
    fig.add_trace(
        go.Bar(
            x=df["month"],
            y=df["content_entries_created"],
            name="Content Created",
            marker=dict(color=COLORS["secondary"], opacity=0.3),
        ),
        secondary_y=False,
    )
    
    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text=f"Usage Trends — {account_name}", font=dict(size=18)),
        height=400,
        legend=dict(x=0, y=-0.2, orientation="h"),
    )
    
    fig.update_yaxes(title_text="API Calls / Content", secondary_y=False)
    fig.update_yaxes(title_text="Active Users", secondary_y=True)
    
    return fig


def create_ticket_timeline_chart(ticket_records: List[Dict], account_name: str) -> go.Figure:
    """
    Create a timeline scatter plot of support tickets.
    
    Color = priority, size = resolution time (larger = slower),
    shape = status (open vs resolved).
    """
    if not ticket_records:
        return _empty_chart("No support tickets")
    
    df = pd.DataFrame(ticket_records)
    
    priority_colors = {
        "P1": COLORS["danger"],
        "P2": COLORS["warning"],
        "P3": COLORS["info"],
        "P4": COLORS["text_muted"],
    }
    
    fig = go.Figure()
    
    for priority in ["P1", "P2", "P3", "P4"]:
        pdata = df[df["priority"] == priority]
        if len(pdata) == 0:
            continue
        
        fig.add_trace(go.Scatter(
            x=pdata["created_date"],
            y=[priority] * len(pdata),
            mode="markers",
            name=priority,
            marker=dict(
                size=12,
                color=priority_colors.get(priority, COLORS["text_muted"]),
                symbol=["circle" if s in ("Resolved",) else "x" for s in pdata["status"]],
                line=dict(width=1, color="white"),
            ),
            text=pdata["subject"],
            hovertemplate="<b>%{text}</b><br>Date: %{x}<br>Priority: " + priority + "<extra></extra>",
        ))
    
    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text=f"Support Ticket Timeline — {account_name}", font=dict(size=18)),
        height=300,
    )
    fig.update_yaxes(categoryorder="array", categoryarray=["P4", "P3", "P2", "P1"])
    
    return fig


def create_risk_score_breakdown_chart(breakdown: Dict) -> go.Figure:
    """
    Create a horizontal bar chart showing risk score breakdown.
    
    This is the explainability chart — it shows exactly which
    signals are contributing to the risk score and by how much.
    """
    signals = []
    contributions = []
    colors = []
    
    for signal, details in sorted(breakdown.items(), key=lambda x: x[1]["contribution"], reverse=True):
        if details["contribution"] > 0:
            from renewal_intelligence.risk.scoring import _signal_label
            signals.append(_signal_label(signal))
            contributions.append(details["contribution"])
            
            # Color based on contribution magnitude
            if details["contribution"] > 0.1:
                colors.append(COLORS["danger"])
            elif details["contribution"] > 0.05:
                colors.append(COLORS["warning"])
            else:
                colors.append(COLORS["info"])
    
    if not signals:
        return _empty_chart("No risk signals detected")
    
    fig = go.Figure(data=[go.Bar(
        y=signals,
        x=contributions,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{c:.3f}" for c in contributions],
        textposition="outside",
        textfont=dict(size=12, color=COLORS["text"]),
    )])
    
    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="Risk Score Breakdown", font=dict(size=18)),
        xaxis_title="Contribution to Risk Score",
        height=max(250, len(signals) * 40 + 100),
    )
    
    return fig


def create_portfolio_heatmap(results_df: pd.DataFrame) -> go.Figure:
    """
    Create a heatmap showing risk signals across all accounts.
    
    Rows = accounts (sorted by risk score), Columns = signal types.
    This gives a portfolio-wide view of where risk is concentrated.
    """
    if len(results_df) == 0:
        return _empty_chart("No data for heatmap")
    
    # Extract signal scores for each account
    signal_names = [
        "usage_decline", "competitor_mention", "p1_tickets",
        "nps_detractor", "open_tickets", "executive_escalation",
        "budget_concern", "sdk_deprecation_risk", "product_risk_impact",
    ]
    
    labels = {
        "usage_decline": "Usage ↓",
        "competitor_mention": "Competitors",
        "p1_tickets": "P1 Tickets",
        "nps_detractor": "NPS Low",
        "open_tickets": "Open Tickets",
        "executive_escalation": "Exec Involved",
        "budget_concern": "Budget Risk",
        "sdk_deprecation_risk": "SDK Risk",
        "product_risk_impact": "Product Risk",
    }
    
    matrix = []
    account_names = []
    
    for _, row in results_df.head(20).iterrows():  # Top 20 by risk
        account_names.append(f"{row['account_name']} ({row['risk_tier'][0]})")
        signals = row.get("signals", {}).get("scores", {})
        matrix.append([signals.get(s, 0) for s in signal_names])
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[labels.get(s, s) for s in signal_names],
        y=account_names,
        colorscale=[
            [0, COLORS["bg_card"]],
            [0.3, COLORS["warning"]],
            [0.7, "#F97316"],
            [1, COLORS["danger"]],
        ],
        text=[[f"{v:.2f}" for v in row] for row in matrix],
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.2f}<extra></extra>",
    ))
    
    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="Risk Signal Heatmap (Top 20)", font=dict(size=20)),
        height=max(400, len(account_names) * 30 + 100),
    )
    fig.update_xaxes(side="top")
    
    return fig


def create_nps_distribution_chart(nps_df: pd.DataFrame) -> go.Figure:
    """Create a histogram of NPS score distribution."""
    fig = go.Figure(data=[go.Histogram(
        x=nps_df["score"],
        nbinsx=11,
        marker=dict(
            color=[
                COLORS["danger"] if i <= 6 
                else COLORS["warning"] if i <= 8 
                else COLORS["success"]
                for i in range(11)
            ],
            line=dict(width=1, color=COLORS["bg_dark"]),
        ),
    )])
    
    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="NPS Score Distribution", font=dict(size=18)),
        xaxis_title="NPS Score",
        yaxis_title="Count",
        height=350,
        bargap=0.1,
    )
    
    # Add vertical lines for NPS boundaries
    fig.add_vline(x=6.5, line_dash="dash", line_color=COLORS["danger"], 
                  annotation_text="Detractor / Passive", annotation_position="top")
    fig.add_vline(x=8.5, line_dash="dash", line_color=COLORS["success"],
                  annotation_text="Passive / Promoter", annotation_position="top")
    
    return fig


def create_sdk_version_chart(usage_df: pd.DataFrame) -> go.Figure:
    """Create a chart showing SDK version distribution across accounts."""
    # Get latest SDK version per account
    latest = usage_df.sort_values("month_dt").groupby("account_id").last()["sdk_version"]
    version_counts = latest.value_counts().sort_index()
    
    colors = []
    for ver in version_counts.index:
        if ver.startswith("v3"):
            colors.append(COLORS["danger"])
        elif ver in ("v4.0.0",):
            colors.append(COLORS["warning"])
        elif ver in ("v4.1.0",):
            colors.append(COLORS["info"])
        else:
            colors.append(COLORS["success"])
    
    fig = go.Figure(data=[go.Bar(
        x=version_counts.index.tolist(),
        y=version_counts.values.tolist(),
        marker=dict(color=colors, line=dict(width=0)),
        text=version_counts.values.tolist(),
        textposition="outside",
        textfont=dict(size=14, color=COLORS["text"]),
    )])
    
    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="SDK Version Distribution", font=dict(size=18)),
        xaxis_title="SDK Version",
        yaxis_title="Number of Accounts",
        height=350,
    )
    
    return fig


def save_chart_as_image(fig: go.Figure, filename: str) -> Path:
    """
    Save a Plotly figure as a PNG for multimodal analysis.
    
    These images are what Gemini Vision will analyze to detect
    visual patterns in the data.
    """
    filepath = CHART_OUTPUT_DIR / f"{filename}.png"
    try:
        fig.write_image(str(filepath), width=1200, height=600, scale=2)
    except Exception:
        # Kaleido might not be installed; skip image saving
        pass
    return filepath


def _empty_chart(message: str) -> go.Figure:
    """Create a placeholder chart with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        font=dict(size=16, color=COLORS["text_muted"]),
        showarrow=False,
    )
    fig.update_layout(**BASE_LAYOUT, height=300)
    return fig
