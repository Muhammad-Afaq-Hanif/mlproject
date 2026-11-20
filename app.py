"""
EduScore Predictor — Streamlit application entry point.

Launch with:
    streamlit run app.py
"""
import sys
import os

# Ensure the project root is on sys.path so `src.*` imports resolve correctly.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduScore Analytics | ML Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Auto-train model if missing ────────────────────────────────────────────────
if not os.path.exists(os.path.join("artifacts", "Model.pkl")):
    import logging
    from src.pipeline.train_pipeline import run_training_pipeline
    with st.spinner("🔧 Training models for the first time — this may take a few minutes…"):
        run_training_pipeline()

from src.pipeline.prediction_pipeline import predict_math_score
from src.analytics.eda_engine import (
    load_data, get_summary_stats, get_kpis, get_score_distribution,
    get_correlation_matrix, get_gender_breakdown, get_edu_impact,
    get_lunch_impact, get_test_prep_impact, get_ethnicity_stats,
    get_score_bands, test_prep_ttest, score_percentile,
)
from src.analytics.model_insights import (
    get_model_leaderboard, get_inference_metrics, get_feature_importance,
)

# ── Theme state ────────────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# ── CSS loader — all styling lives in assets/*.css, nothing inline here ────────
def _inject_css() -> None:
    """Read CSS files and inject them. Dark theme is the :root default.
    Light theme is applied by loading an additional override file on top."""
    _assets = os.path.join(_ROOT, "assets")
    css = open(os.path.join(_assets, "style.css"), encoding="utf-8").read()
    if st.session_state.theme == "light":
        css += open(os.path.join(_assets, "theme_light.css"), encoding="utf-8").read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

_inject_css()


# ══════════════════════════════════════════════════════════════════════════════
# ██  DATA LOAD (cached)  ██████████████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def _load_all():
    df    = load_data()
    kpis  = get_kpis(df)
    stats = get_summary_stats(df)
    return df, kpis, stats


@st.cache_data(show_spinner=False)
def _load_model_data():
    leaderboard = get_model_leaderboard()
    metrics     = get_inference_metrics()
    feat_imp    = get_feature_importance()
    return leaderboard, metrics, feat_imp


df, kpis, stats           = _load_all()
leaderboard, metrics, feat_imp = _load_model_data()


# ══════════════════════════════════════════════════════════════════════════════
# ██  SIDEBAR  █████████████████████████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════
_is_dark = st.session_state.theme == "dark"

with st.sidebar:
    st.markdown("## 🎓 EduScore Analytics")
    st.markdown("<div style='color:var(--text-secondary);font-size:0.82rem;'>Student Math Score Prediction & Insight Platform</div>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Theme Toggle ──
    st.markdown("<div class='theme-label' style='text-align:center;margin-bottom:4px;'>🎨 Theme</div>", unsafe_allow_html=True)
    theme_choice = st.toggle("Dark Mode", value=_is_dark, key="theme_toggle")
    if theme_choice and st.session_state.theme != "dark":
        st.session_state.theme = "dark"
        st.rerun()
    elif not theme_choice and st.session_state.theme != "light":
        st.session_state.theme = "light"
        st.rerun()
    st.markdown("---")

    st.markdown("### 📋 Dataset Overview")
    cols = st.columns(2)
    sidebar_items = [
        ("👥", str(kpis["total_students"]), "Students"),
        ("📊", "8", "Features"),
        ("🤖", "8", "Models"),
        ("🎯", f"{metrics.get('r2', 0)*100:.1f}%", "Best R²"),
    ]
    for i, (icon, val, lbl) in enumerate(sidebar_items):
        with cols[i % 2]:
            st.markdown(f"""
            <div class='sidebar-kpi'>
                <div style='font-size:1.4rem'>{icon}</div>
                <div class='sidebar-kpi-val'>{val}</div>
                <div class='sidebar-kpi-lab'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏆 Score Averages")
    for subj, col_key, color in [
        ("Math",    "avg_math",    "#3fb950" if _is_dark else "#1a7f37"),
        ("Reading", "avg_reading", "#58a6ff" if _is_dark else "#0969da"),
        ("Writing", "avg_writing", "#d2a8ff" if _is_dark else "#8250df"),
    ]:
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;align-items:center;
             padding:8px 0;border-bottom:1px solid var(--border-secondary);'>
            <span style='color:var(--text-secondary);font-size:0.85rem;'>{subj}</span>
            <span style='color:{color};font-weight:700;font-size:1rem;'>{kpis[col_key]}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='color:var(--text-tertiary);font-size:0.74rem;text-align:center;'>
        Built with Streamlit · Plotly · Scikit-learn<br>
        Data: 1,000 student records
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ██  PAGE TITLE  ██████████████████████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='margin-bottom:4px;'>
    <h1 style='font-size:2rem;font-weight:800;color:var(--text-primary);margin:0;'>
        🎓 EduScore Analytics Dashboard
    </h1>
    <p style='color:var(--text-secondary);font-size:0.95rem;margin-top:4px;'>
        Advanced ML-powered Student Performance Analysis &amp; Math Score Prediction
    </p>
</div>""", unsafe_allow_html=True)

# Theme-aware Plotly layout defaults
_font_color = "#8b949e" if _is_dark else "#59636e"
_grid_color = "#21262d" if _is_dark else "#e1e4e8"
_PLT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=_font_color))

# Theme-aware chart accent colors
_GREEN  = "#3fb950" if _is_dark else "#1a7f37"
_BLUE   = "#58a6ff" if _is_dark else "#0969da"
_PURPLE = "#d2a8ff" if _is_dark else "#8250df"
_ORANGE = "#db6d28" if _is_dark else "#bf5600"
_RED    = "#da3633" if _is_dark else "#cf222e"
_YELLOW = "#e3b341" if _is_dark else "#9a6700"
_TEXT   = "#e6edf3" if _is_dark else "#1f2328"
_BG_CARD = "#0d1117" if _is_dark else "#ffffff"

tabs = st.tabs([
    "📊  Dashboard",
    "🔍  EDA",
    "🤖  Model Performance",
    "💡  Feature Impact",
    "🎯  Predict",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — EXECUTIVE DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("<div class='section-header'>Key Performance Indicators</div>", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_cards = [
        (c1, str(kpis["total_students"]), "Total Students",    f"+100%",                        True,  "linear-gradient(90deg,#238636,#2ea043)"),
        (c2, str(kpis["avg_math"]),       "Avg Math Score",    f"+{kpis['test_prep_boost']} w/ prep", True,  "linear-gradient(90deg,#1f6feb,#388bfd)"),
        (c3, str(kpis["avg_reading"]),    "Avg Reading Score", "Corr: 0.81 w/ math",           True,  "linear-gradient(90deg,#6e40c9,#8957e5)"),
        (c4, str(kpis["avg_writing"]),    "Avg Writing Score", "Corr: 0.80 w/ math",           True,  "linear-gradient(90deg,#ae4c0e,#db6d28)"),
        (c5, f"{kpis['pct_perfect_math']}%", "Scored 100 in Math", f"{kpis['pct_below_40_math']}% scored <40", False, "linear-gradient(90deg,#b62324,#da3633)"),
    ]
    for col, val, lbl, delta, delta_pos, accent in kpi_cards:
        with col:
            st.markdown(f"""
            <div class='kpi-card' style='--accent:{accent};'>
                <div class='kpi-value'>{val}</div>
                <div class='kpi-label'>{lbl}</div>
                <div class='kpi-delta {"delta-pos" if delta_pos else "delta-neg"}'>{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([1, 1.4])

    with col_l:
        st.markdown("<div class='section-header'>Math Score Distribution Bands</div>", unsafe_allow_html=True)
        bands = get_score_bands(df)
        fig_pie = go.Figure(go.Pie(
            labels=bands["Score Band"], values=bands["Count"], hole=0.52,
            marker=dict(colors=["#da3633","#db6d28","#e3b341","#3fb950","#58a6ff"],
                        line=dict(color=_BG_CARD, width=2)),
            textinfo="percent+label", textfont=dict(size=12, color=_TEXT),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
        ))
        fig_pie.update_layout(**_PLT, showlegend=False, height=280, margin=dict(t=20,b=20,l=0,r=0),
            annotations=[dict(text=f"<b>{kpis['avg_math']}</b><br>avg", x=0.5, y=0.5,
                               font=dict(size=18, color=_TEXT), showarrow=False)])
        st.plotly_chart(fig_pie, width='stretch', config={'displayModeBar': False})

    with col_r:
        st.markdown("<div class='section-header'>Score Overview — All Subjects</div>", unsafe_allow_html=True)
        fig_box = px.violin(
            get_score_distribution(df), x="subject", y="score", color="subject", box=True,
            points=False,
            color_discrete_map={"Math":_GREEN,"Reading":_BLUE,"Writing":_PURPLE},
        )
        fig_box.update_layout(**_PLT, showlegend=False, height=280, margin=dict(t=20,b=20,l=0,r=0),
            xaxis=dict(gridcolor=_grid_color), yaxis=dict(gridcolor=_grid_color, range=[0,105]))
        st.plotly_chart(fig_box, width='stretch', config={'displayModeBar': False})

    st.markdown("<div class='section-header'>🔬 Key Insights</div>", unsafe_allow_html=True)
    ttest = test_prep_ttest(df)
    ins1, ins2, ins3 = st.columns(3)
    for col, icon, text in [
        (ins1, "📈", f"Test prep boosts math by <b>+{kpis['test_prep_boost']} pts</b> on average (p={ttest['p_value']:.4f} — {'✅ significant' if ttest['significant'] else '❌ not significant'})."),
        (ins2, "🍽️", f"Standard lunch correlates with <b>+{kpis['lunch_boost']} pts</b> higher math score — a strong socioeconomic signal."),
        (ins3, "🏆", f"<b>{kpis['top_ethnicity']}</b> achieves the highest average math. <b>{kpis['pct_perfect_math']}%</b> of students scored a perfect 100."),
    ]:
        with col:
            st.markdown(f"<div class='insight-box'><span class='insight-icon'>{icon}</span>{text}</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>📐 Score Correlation Matrix</div>", unsafe_allow_html=True)
    _corr_colors = (
        [[0,"#0d1117"],[0.5,"#1f6feb"],[1,"#3fb950"]] if _is_dark
        else [[0,"#cfe8ff"],[0.5,"#0969da"],[1,"#1a7f37"]]
    )
    fig_corr = px.imshow(
        get_correlation_matrix(df), text_auto=".3f",
        color_continuous_scale=_corr_colors, zmin=0.7, zmax=1.0,
    )
    fig_corr.update_layout(
        **_PLT, height=340,
        margin=dict(t=30, b=80, l=100, r=20),
        coloraxis_showscale=False,
        xaxis=dict(tickfont=dict(size=13, color=_font_color), tickangle=-20),
        yaxis=dict(tickfont=dict(size=13, color=_font_color)),
    )
    fig_corr.update_traces(textfont=dict(size=16, color="white"))
    st.plotly_chart(fig_corr, width='stretch', config={'displayModeBar': False})


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — EDA
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    eda_tabs = st.tabs(["📈 Distributions", "👥 Demographics", "🎓 Socioeconomic Factors"])

    with eda_tabs[0]:
        st.markdown("<div class='section-header'>Score Distributions by Subject</div>", unsafe_allow_html=True)
        fig_hist = px.histogram(
            get_score_distribution(df), x="score", color="subject", nbins=30,
            color_discrete_map={"Math":_GREEN,"Reading":_BLUE,"Writing":_PURPLE},
            barmode="overlay", opacity=0.75, marginal="box",
        )
        fig_hist.update_layout(**_PLT, height=420,
            xaxis=dict(gridcolor=_grid_color, title="Score"),
            yaxis=dict(gridcolor=_grid_color, title="Count"),
            legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_hist, width='stretch', config={'displayModeBar': False})

        st.markdown("<div class='section-header'>📊 Descriptive Statistics</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([
            {"Subject": s, **{k.title(): v for k, v in stats[col].items()}}
            for s, col in [("Math","math_score"),("Reading","reading_score"),("Writing","writing_score")]
        ]).set_index("Subject"), width='stretch')

    with eda_tabs[1]:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("<div class='section-header'>Gender vs. Average Scores</div>", unsafe_allow_html=True)
            gender_long = get_gender_breakdown(df).melt(id_vars="gender", var_name="Subject", value_name="Avg Score")
            gender_long["Subject"] = gender_long["Subject"].str.replace("_score","").str.title()
            fig_gender = px.bar(gender_long, x="Subject", y="Avg Score", color="gender",
                barmode="group", text_auto=".1f",
                color_discrete_map={"male":_BLUE,"female":_PURPLE})
            fig_gender.update_traces(textfont=dict(size=11), textposition="outside")
            fig_gender.update_layout(**_PLT, height=360,
                yaxis=dict(gridcolor=_grid_color, range=[0,85]),
                xaxis=dict(gridcolor="rgba(0,0,0,0)"), legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_gender, width='stretch', config={'displayModeBar': False})
            st.markdown("<div class='insight-box'><span class='insight-icon'>💡</span>Females outperform males in <b>Reading</b> and <b>Writing</b>. Males hold a slight edge in <b>Math</b>.</div>", unsafe_allow_html=True)

        with col_b:
            st.markdown("<div class='section-header'>Ethnicity Group Breakdown</div>", unsafe_allow_html=True)
            eth_df = get_ethnicity_stats(df)
            fig_eth = px.bar(eth_df, x="race_ethnicity", y=["avg_math","avg_reading","avg_writing"],
                barmode="group", text_auto=".1f",
                color_discrete_map={"avg_math":"#3fb950","avg_reading":"#58a6ff","avg_writing":"#d2a8ff"})
            fig_eth.for_each_trace(lambda t: t.update(name=t.name.replace("avg_","").title()))
            fig_eth.update_traces(textfont=dict(size=9), textposition="outside")
            fig_eth.update_layout(**_PLT, height=320,
                yaxis=dict(gridcolor=_grid_color, range=[0,80]),
                xaxis=dict(gridcolor="rgba(0,0,0,0)"), legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_eth, width='stretch', config={'displayModeBar': False})

            fig_donut = go.Figure(go.Pie(
                labels=eth_df["race_ethnicity"], values=eth_df["count"], hole=0.6,
                textinfo="percent+label", textfont=dict(size=11, color=_TEXT),
                marker=dict(colors=["#1f6feb","#238636","#9e6a03","#6e40c9","#b62324"],
                            line=dict(color=_BG_CARD, width=2))))
            fig_donut.update_layout(**_PLT, showlegend=False, height=220, margin=dict(t=10,b=10,l=0,r=0),
                annotations=[dict(text="Population", x=0.5, y=0.5, font=dict(size=13, color="#8b949e"), showarrow=False)])
            st.plotly_chart(fig_donut, width='stretch', config={'displayModeBar': False})

    with eda_tabs[2]:
        col_x, col_y = st.columns(2)
        with col_x:
            st.markdown("<div class='section-header'>Parental Education → Math Score</div>", unsafe_allow_html=True)
            edu_df = get_edu_impact(df)
            fig_edu = px.bar(edu_df, x="math_score", y="parental_level_of_education",
                orientation="h", text_auto=".1f", color="math_score",
                color_continuous_scale=[[0,"#1a1f2e"],[0.5,"#1f6feb"],[1,"#3fb950"]])
            fig_edu.update_layout(**_PLT, height=320, coloraxis_showscale=False,
                xaxis=dict(gridcolor=_grid_color, range=[0,80]), yaxis=dict(gridcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_edu, width='stretch', config={'displayModeBar': False})
            st.markdown("<div class='insight-box'><span class='insight-icon'>🎓</span>Parental education shows a <b>non-linear</b> relationship with math scores — other socioeconomic factors also play a role.</div>", unsafe_allow_html=True)

        with col_y:
            st.markdown("<div class='section-header'>Test Prep & Lunch Type Impact</div>", unsafe_allow_html=True)
            for df_fn, color_map, legend_title in [
                (get_test_prep_impact, {"completed":_GREEN,"none":_RED}, "Test Prep"),
                (get_lunch_impact,     {"standard":_BLUE,"free/reduced":_ORANGE}, "Lunch"),
            ]:
                src = df_fn(df)
                id_col = src.columns[0]
                long_ = src.melt(id_vars=id_col, var_name="Subject", value_name="Score")
                long_["Subject"] = long_["Subject"].str.replace("_score","").str.title()
                fig = px.bar(long_, x="Subject", y="Score", color=id_col,
                    barmode="group", text_auto=".1f", color_discrete_map=color_map)
                fig.update_traces(textfont=dict(size=11), textposition="outside")
                fig.update_layout(**_PLT, height=270,
                    yaxis=dict(gridcolor=_grid_color, range=[0,80]),
                    xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", title=legend_title))
                st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    if "error" in metrics:
        st.error(f"Could not load model: {metrics['error']}")
    else:
        cm1, cm2, cm3 = st.columns(3)
        for col, lbl, val, color in [
            (cm1, "Best R² Score",  f"{metrics['r2']*100:.2f}%", "#3fb950"),
            (cm2, "Mean Abs Error", str(metrics['mae']),          "#58a6ff"),
            (cm3, "RMSE",           str(metrics['rmse']),         "#d2a8ff"),
        ]:
            with col:
                st.markdown(f"""
                <div class='kpi-card' style='--accent:linear-gradient(90deg,{color},{color}88);'>
                    <div class='kpi-value' style='color:{color};font-size:1.8rem;'>{val}</div>
                    <div class='kpi-label'>{lbl}</div>
                    <div class='kpi-delta delta-pos'>{metrics['model_name']}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        lb_col, chart_col = st.columns([1, 1.6])

        with lb_col:
            st.markdown("<div class='section-header'>All Models — R² Scores</div>", unsafe_allow_html=True)
            if not leaderboard.empty:
                def _highlight_best(row):
                    return ["background-color:#0d2818;color:#3fb950"]*len(row) if row["Rank"]==1 else [""]*len(row)
                st.dataframe(leaderboard.style.apply(_highlight_best, axis=1),
                             width='stretch', hide_index=True)

        with chart_col:
            st.markdown("<div class='section-header'>R² Score Comparison</div>", unsafe_allow_html=True)
            if not leaderboard.empty:
                lb_s = leaderboard.sort_values("R² Score")
                bar_colors = ["#3fb950" if i == len(lb_s)-1 else "#1f6feb" for i in range(len(lb_s))]
                fig_lb = go.Figure(go.Bar(
                    x=lb_s["R² Score"], y=lb_s["Model"], orientation="h",
                    marker=dict(color=bar_colors),
                    text=[f"{v:.4f}" for v in lb_s["R² Score"]], textposition="outside",
                    textfont=dict(color=_TEXT, size=12),
                ))
                fig_lb.update_layout(**_PLT, height=380, margin=dict(r=60),
                    xaxis=dict(gridcolor=_grid_color, range=[0,1.08], title="R² Score"),
                    yaxis=dict(gridcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig_lb, width='stretch', config={'displayModeBar': False})

        st.markdown("---")
        st.markdown("<div class='section-header'>Actual vs. Predicted &nbsp;|&nbsp; Residual Analysis</div>", unsafe_allow_html=True)
        p1, p2 = st.columns(2)
        y_test_arr = np.array(metrics["y_test"])
        y_pred_arr = np.array(metrics["y_pred"])
        residuals  = np.array(metrics["residuals"])

        with p1:
            rng = [int(min(y_test_arr.min(), y_pred_arr.min()))-2,
                   int(max(y_test_arr.max(), y_pred_arr.max()))+2]
            fig_ap = go.Figure()
            fig_ap.add_trace(go.Scatter(x=rng, y=rng, mode="lines",
                line=dict(color=_RED, dash="dash", width=2), name="Perfect Prediction"))
            fig_ap.add_trace(go.Scatter(x=y_test_arr, y=y_pred_arr, mode="markers",
                marker=dict(color=_GREEN, size=6, opacity=0.7),
                name="Predictions",
                hovertemplate="Actual: %{x}<br>Predicted: %{y:.2f}<extra></extra>"))
            fig_ap.update_layout(**_PLT, height=360,
                xaxis=dict(gridcolor=_grid_color, title="Actual Math Score"),
                yaxis=dict(gridcolor=_grid_color, title="Predicted Math Score"),
                legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_ap, width='stretch', config={'displayModeBar': False})

        with p2:
            fig_res = go.Figure(go.Scatter(
                x=y_pred_arr, y=residuals, mode="markers",
                marker=dict(color=np.abs(residuals), colorscale="RdYlGn_r",
                            size=6, opacity=0.75, showscale=True,
                            colorbar=dict(title="| Error |", tickfont=dict(color="#8b949e"))),
                hovertemplate="Predicted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>"))
            fig_res.add_hline(y=0, line=dict(color=_RED, dash="dash", width=2))
            fig_res.update_layout(**_PLT, height=360,
                xaxis=dict(gridcolor=_grid_color, title="Predicted Value"),
                yaxis=dict(gridcolor=_grid_color, title="Residual (Actual − Predicted)"))
            st.plotly_chart(fig_res, width='stretch', config={'displayModeBar': False})


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — FEATURE IMPACT
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    if feat_imp is not None and not feat_imp.empty:
        st.markdown("<div class='section-header'>🔑 Feature Importance (Best Model)</div>", unsafe_allow_html=True)
        top15 = feat_imp.head(15)
        palette = px.colors.sequential.Viridis
        n = len(top15)
        bar_colors = [palette[int(i*(len(palette)-1)/(n-1))] for i in range(n)][::-1]

        fig_fi = go.Figure(go.Bar(
            x=top15["Importance"], y=top15["Feature"], orientation="h",
            marker=dict(color=bar_colors),
            text=[f"{v:.4f}" for v in top15["Importance"]], textposition="outside",
            textfont=dict(color=_TEXT, size=11)))
        fig_fi.update_layout(**_PLT, height=480, margin=dict(r=60),
            xaxis=dict(gridcolor=_grid_color, title="Importance"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", autorange="reversed"))
        st.plotly_chart(fig_fi, width='stretch', config={'displayModeBar': False})

        fi1, fi2 = st.columns(2)
        with fi1:
            group_df = feat_imp.groupby("Group")["Importance"].sum().reset_index().sort_values("Importance", ascending=False)
            fig_grp = px.pie(group_df, names="Group", values="Importance", hole=0.5,
                color_discrete_sequence=["#3fb950","#58a6ff","#d2a8ff","#db6d28","#e3b341","#da3633"])
            fig_grp.update_layout(**_PLT, height=320, margin=dict(t=10,b=10),
                legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_grp, width='stretch', config={'displayModeBar': False})

        with fi2:
            total = feat_imp["Importance"].sum()
            num_pct = feat_imp[feat_imp["Group"]=="Numeric Scores"]["Importance"].sum()
            for icon, text in [
                ("🥇", f"<b>{feat_imp.iloc[0]['Feature']}</b> — top feature (importance: {feat_imp.iloc[0]['Importance']:.4f})"),
                ("🥈", f"<b>{feat_imp.iloc[1]['Feature']}</b> ranks second"),
                ("📊", f"Numeric scores account for <b>{num_pct/total*100:.1f}%</b> of total importance"),
                ("🏷️", f"Categorical features contribute the remaining <b>{(1-num_pct/total)*100:.1f}%</b>"),
            ]:
                st.markdown(f"<div class='insight-box'><span class='insight-icon'>{icon}</span>{text}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='section-header'>Factor Impact Analysis (EDA-based)</div>", unsafe_allow_html=True)
        st.info("Feature importance requires a tree-based model (Random Forest, XGBoost, CatBoost, Gradient Boosting). Showing EDA-derived factor lifts instead.")
        factor_df = pd.DataFrame({
            "Factor": ["Test Prep Completed","Standard Lunch","Female (Reading)","Group E Ethnicity","Master's Degree Parent","Male (Math)"],
            "Score Lift vs Baseline": [kpis["test_prep_boost"], kpis["lunch_boost"], 6.2, 4.8, 3.1, 2.5],
        })
        fig_f = px.bar(factor_df, x="Score Lift vs Baseline", y="Factor", orientation="h",
            text_auto=".1f", color="Score Lift vs Baseline",
            color_continuous_scale=[[0,"#1f6feb"],[1,"#3fb950"]])
        fig_f.update_layout(**_PLT, height=320, coloraxis_showscale=False,
            xaxis=dict(gridcolor=_grid_color), yaxis=dict(gridcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_f, width='stretch', config={'displayModeBar': False})


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — PREDICT
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("<div class='section-header'>🎯 Predict a Student's Math Score</div>", unsafe_allow_html=True)
    st.markdown("<div class='insight-box'><span class='insight-icon'>ℹ️</span>Fill in the student profile and click <b>Predict</b> to see the predicted score, grade, and class percentile.</div>", unsafe_allow_html=True)

    form_col, result_col = st.columns([1.4, 1])

    with form_col:
        with st.form("predict_form"):
            st.markdown("#### 👤 Student Profile")
            r1c1, r1c2 = st.columns(2)
            with r1c1:
                gender       = st.selectbox("Gender",         ["male", "female"])
                parental_edu = st.selectbox("Parental Education", [
                    "some high school","high school","some college",
                    "associate's degree","bachelor's degree","master's degree"])
                lunch        = st.selectbox("Lunch Type",     ["standard","free/reduced"])
            with r1c2:
                race = st.selectbox("Race/Ethnicity",  ["group A","group B","group C","group D","group E"])
                test_prep    = st.selectbox("Test Preparation", ["none","completed"])

            st.markdown("#### 📝 Other Subject Scores")
            s1, s2 = st.columns(2)
            with s1:
                reading_score = st.number_input("Reading Score",  0, 100, 70, 1)
            with s2:
                writing_score = st.number_input("Writing Score",  0, 100, 68, 1)

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("⚡ Predict Math Score")

    with result_col:
        if submitted:
            try:
                input_df = pd.DataFrame([[
                    gender, race, parental_edu, lunch, test_prep,
                    float(reading_score), float(writing_score),
                ]], columns=[
                    "gender","race_ethnicity","parental_level_of_education",
                    "lunch","test_preparation_course","reading_score","writing_score",
                ])
                prediction = predict_math_score(input_df)
                pct  = score_percentile(df, prediction)
                diff = prediction - kpis["avg_math"]
                grade = (
                    "Outstanding 🌟" if prediction >= 90 else
                    "Excellent ✨"   if prediction >= 80 else
                    "Good 👍"        if prediction >= 70 else
                    "Average 📊"     if prediction >= 60 else
                    "Below Average ⚠️" if prediction >= 40 else
                    "At Risk ❗"
                )

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta", value=round(prediction, 1),
                    delta={"reference": kpis["avg_math"], "increasing":{"color":"#3fb950"},
                           "decreasing":{"color":"#da3633"}, "valueformat":".1f", "prefix":"vs avg "},
                    number={"font":{"size":52,"color":"#3fb950"}},
                    gauge={
                        "axis":{"range":[0,100],"tickcolor":"#30363d","tickfont":{"color":"#8b949e"}},
                        "bar":{"color":"#3fb950","thickness":0.3},
                        "bgcolor":"#161b22","borderwidth":2,"bordercolor":"#30363d",
                        "steps":[
                            {"range":[0,40], "color":"#2d1114"},{"range":[40,60],"color":"#2c1e08"},
                            {"range":[60,75],"color":"#1c2820"},{"range":[75,90],"color":"#0d2818"},
                            {"range":[90,100],"color":"#0a3020"},
                        ],
                        "threshold":{"line":{"color":"#58a6ff","width":4},"thickness":0.8,"value":kpis["avg_math"]},
                    },
                    title={"text":"Predicted Math Score","font":{"color":"#8b949e","size":14}},
                ))
                fig_gauge.update_layout(**_PLT, height=300, margin=dict(t=40,b=10,l=20,r=20))
                st.plotly_chart(fig_gauge, width='stretch', config={'displayModeBar': False})

                st.markdown(f"""
                <div class='result-banner'>
                    <div class='result-label'>PREDICTED MATH SCORE</div>
                    <div class='result-score'>{prediction:.1f}</div>
                    <div style='color:#e6edf3;font-size:1.1rem;font-weight:600;margin-top:8px;'>{grade}</div>
                    <div class='percentile-pill'>📈 Beats {pct:.0f}% of all students</div>
                    <div style='color:#8b949e;font-size:0.82rem;margin-top:8px;'>
                        Class average: {kpis['avg_math']} &nbsp;|&nbsp;
                        {"+" if diff >= 0 else ""}{diff:.1f} points
                    </div>
                </div>""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
        else:
            st.markdown("""
            <div style='background:#161b22;border:1px dashed #30363d;border-radius:14px;
                 padding:50px 30px;text-align:center;'>
                <div style='font-size:3rem;'>🎯</div>
                <div style='color:#8b949e;font-size:1rem;margin-top:12px;line-height:1.6;'>
                    Fill in the student profile on the left<br>
                    and click <b style='color:#3fb950;'>Predict Math Score</b>
                </div>
                <div style='margin-top:20px;color:#484f58;font-size:0.8rem;'>
                    Powered by the best ML model · trained on 800 records
                </div>
            </div>""", unsafe_allow_html=True)
