# Student Performance Dashboard (Deployable Version)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# ---- Page and Theme ----
st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
sns.set_theme(style="whitegrid")
VIVID = px.colors.qualitative.Bold

# ---- Load Data ----
@st.cache_data
def load_data():
    data_path = Path(__file__).parent / "student_performance_cleaned.csv"
    df = pd.read_csv(data_path)
    df["passed_num"] = df["passed"].replace({"Yes": 1, "No": 0})
    score_cols = [
        "math_score", "reading_score", "writing_score",
        "science_score", "history_score", "geography_score",
        "attendance_rate", "performance_score"
    ]
    for c in score_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

df = load_data()

# ---- Sidebar Filters ----
students = st.sidebar.multiselect("Select Students", sorted(df["student_name"].unique()))
genders = st.sidebar.multiselect("Select Gender", sorted(df["gender"].dropna().unique()))
remove_outliers = st.sidebar.checkbox("Remove Outliers (IQR)", value=False)

filtered = df.copy()
if students:
    filtered = filtered[filtered["student_name"].isin(students)]
if genders:
    filtered = filtered[filtered["gender"].isin(genders)]
if remove_outliers:
    q1, q3 = np.nanpercentile(filtered["performance_score"], [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    filtered = filtered[(filtered["performance_score"] >= lower) & (filtered["performance_score"] <= upper)]

st.sidebar.write(f"Total Students: {len(filtered)}")

subject_cols = [
    "math_score", "reading_score", "writing_score",
    "science_score", "history_score", "geography_score"
]

# ---- KPIs ----
st.title("Student Performance Dashboard")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg Performance", f"{filtered['performance_score'].mean():.1f}%")
c2.metric("Avg Attendance", f"{filtered['attendance_rate'].mean():.1f}%")
c3.metric("Pass Rate", f"{filtered['passed_num'].mean()*100:.1f}%")
c4.metric("Total Students", len(filtered))
st.markdown("---")

# ---- Visualization Toggles ----
st.sidebar.header("Select Visualizations")
show_avg = st.sidebar.checkbox("Average Scores", True)
show_attendance = st.sidebar.checkbox("Attendance vs Performance", True)
show_gender = st.sidebar.checkbox("Performance by Gender", True)
show_corr = st.sidebar.checkbox("Correlation Heatmap", True)
show_grade = st.sidebar.checkbox("Overall Grade Distribution", True)
show_compare = st.sidebar.checkbox("Subject Comparison (X vs Y)", True)
show_radar = st.sidebar.checkbox("Subject Radar", True)
show_leaderboard = st.sidebar.checkbox("Leaderboard", True)
show_pies = st.sidebar.checkbox("Pie Charts", True)

# ---- Average Scores ----
if show_avg:
    st.subheader("Average Scores by Subject")
    avg_df = (
        filtered[subject_cols]
        .mean()
        .reset_index()
        .rename(columns={"index": "Subject", 0: "Average"})
        .sort_values("Average", ascending=True)
    )
    fig = px.bar(avg_df, x="Average", y="Subject", orientation="h",
                 color="Average", color_continuous_scale="Viridis", height=420)
    fig.update_traces(marker_line_color="rgba(0,0,0,0.35)", marker_line_width=0.8)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

# ---- Attendance vs Performance ----
if show_attendance:
    st.subheader("Attendance Rate vs Performance")
    try:
        import statsmodels.api as _sm
        trend = "ols"
    except Exception:
        trend = None
    fig2 = px.scatter(filtered, x="attendance_rate", y="performance_score",
                      color="passed", color_discrete_map={"Yes": "#2A9D8F", "No": "#E63946"},
                      trendline=trend, height=460)
    fig2.update_traces(marker_size=9, marker_opacity=0.85,
                       marker_line_width=0.8, marker_line_color="rgba(0,0,0,0.3)")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("---")

# ---- Performance by Gender ----
if show_gender:
    st.subheader("Performance by Gender")
    fig3 = px.box(filtered, x="gender", y="performance_score",
                  color="gender", color_discrete_sequence=VIVID, height=450)
    strip = px.strip(filtered, x="gender", y="performance_score", color="gender",
                     color_discrete_sequence=VIVID).update_traces(
                     jitter=0.25, marker_opacity=0.5, marker_size=5, showlegend=False)
    for tr in strip.data:
        fig3.add_trace(tr)
    fig3.update_layout(showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("---")

# ---- Correlation Heatmap ----
if show_corr:
    st.subheader("Correlation Heatmap")
    corr_vars = st.multiselect("Select Variables",
                               subject_cols + ["attendance_rate", "performance_score"],
                               default=subject_cols + ["performance_score"])
    if len(corr_vars) >= 2:
        corr = filtered[corr_vars].corr()
        fig4, ax4 = plt.subplots(figsize=(9, 6))
        sns.heatmap(corr, cmap="RdBu_r", center=0, annot=True,
                    fmt=".2f", linewidths=0.6, cbar_kws={"shrink": 0.8}, ax=ax4)
        ax4.set_title("Correlation Matrix", pad=10)
        st.pyplot(fig4, clear_figure=True)
    st.markdown("---")

# ---- Overall Grade Distribution ----
if show_grade:
    st.subheader("Overall Grade Distribution")
    fig5 = px.histogram(filtered, x="overall_grade", nbins=20,
                        color_discrete_sequence=["#277DA1"], height=420)
    fig5.update_traces(marker_line_color="rgba(0,0,0,0.35)", marker_line_width=0.8)
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("---")

# ---- Subject Comparison ----
if show_compare:
    st.subheader("Subject Comparison (X vs Y)")
    colx, coly = st.columns(2)
    x_subj = colx.selectbox("X Axis", subject_cols, index=0)
    y_subj = coly.selectbox("Y Axis", subject_cols, index=1)
    fig6 = px.scatter(filtered, x=x_subj, y=y_subj, color="passed",
                      color_discrete_map={"Yes": "#2B8A3E", "No": "#D00000"}, height=460)
    minv = float(np.nanmin([filtered[x_subj], filtered[y_subj]]))
    maxv = float(np.nanmax([filtered[x_subj], filtered[y_subj]]))
    fig6.add_trace(go.Scatter(x=[minv, maxv], y=[minv, maxv],
                              mode="lines", line=dict(color="gray", dash="dash"), showlegend=False))
    st.plotly_chart(fig6, use_container_width=True)
    st.markdown("---")

# ---- Radar Chart ----
if show_radar:
    st.subheader("Subject Radar (Group Profile)")
    group_by = st.selectbox("Group by", ["gender", "passed"], index=0)
    agg = filtered.groupby(group_by)[subject_cols].mean().reset_index()
    categories = [c.replace("_", " ").title() for c in subject_cols]
    figR = go.Figure()
    for _, row in agg.iterrows():
        figR.add_trace(go.Scatterpolar(r=[row[c] for c in subject_cols],
                                       theta=categories, fill="toself",
                                       name=str(row[group_by])))
    figR.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, height=480)
    st.plotly_chart(figR, use_container_width=True)
    st.markdown("---")

# ---- Leaderboard ----
if show_leaderboard:
    st.subheader("Leaderboard")
    n = st.slider("Rows per table", 5, 30, 10)
    cols = ["student_id", "student_name", "attendance_rate", "performance_score", "overall_grade", "passed"]
    top_tbl = filtered[cols].sort_values("performance_score", ascending=False).head(n)
    bottom_tbl = filtered[cols].sort_values("performance_score", ascending=True).head(n)
    t1, t2 = st.columns(2)
    t1.write("Top Performers")
    t1.dataframe(top_tbl, use_container_width=True)
    t2.write("Needs Attention")
    t2.dataframe(bottom_tbl, use_container_width=True)
    st.markdown("---")

# ---- Pie Charts ----
if show_pies:
    st.subheader("Summary Pie Charts")
    col1, col2, col3 = st.columns(3)

    # Pass vs Fail
    with col1:
        pass_counts = (
            filtered["passed"]
            .value_counts(dropna=False)
            .rename_axis("passed")
            .reset_index(name="count")
        )
        figP1 = px.pie(pass_counts, values="count", names="passed",
                       color="passed", color_discrete_map={"Yes": "#4CAF50", "No": "#FF5252"},
                       height=350, title="Pass vs Fail")
        st.plotly_chart(figP1, use_container_width=True)

    # Gender ratio
    with col2:
        gender_counts = (
            filtered["gender"]
            .value_counts(dropna=False)
            .rename_axis("gender")
            .reset_index(name="count")
        )
        figP2 = px.pie(gender_counts, values="count", names="gender",
                       color_discrete_sequence=VIVID, height=350, title="Gender Ratio")
        st.plotly_chart(figP2, use_container_width=True)

    # Subject average share
    with col3:
        avg_subjects = (
            filtered[subject_cols]
            .mean()
            .reset_index()
            .rename(columns={"index": "Subject", 0: "Average"})
        )
        if "Average" not in avg_subjects.columns:
            avg_subjects.columns = ["Subject", "Average"]
        figP3 = px.pie(avg_subjects, values="Average", names="Subject",
                       color_discrete_sequence=px.colors.sequential.Viridis,
                       height=350, title="Subject Average Share")
        st.plotly_chart(figP3, use_container_width=True)

    st.markdown("---")

# ---- Download ----
st.subheader("Download Filtered Data")
csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv,
                   file_name="filtered_student_performance.csv", mime="text/csv")
