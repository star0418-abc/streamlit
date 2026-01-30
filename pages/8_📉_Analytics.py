"""
Analytics Page - Trend analysis and dashboards.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Check plotly availability before importing
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db import list_measurements, get_measurement, list_recipes
from utils.i18n import t, init_language, language_selector

# Initialize language
init_language()

st.set_page_config(page_title=t("analytics.page_title"), page_icon="📉", layout="wide")

# Sidebar language selector
with st.sidebar:
    language_selector()

# Check plotly availability
if not PLOTLY_AVAILABLE:
    st.error("❌ **plotly 未安装 / plotly is not installed**")
    st.code("pip install plotly", language="bash")
    st.info(
        "💡 建议运行以下命令安装所有依赖：\n\n"
        "Recommended: install all dependencies:\n\n"
        "```\npip install -r requirements.txt\n```"
    )
    st.stop()

st.title(t("analytics.title"))
st.markdown(t("analytics.subtitle"))

# Load all measurements
measurements = list_measurements()

if measurements:
    meas_df = pd.DataFrame(measurements)
    
    st.subheader(t("analytics.overview"))
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(t("analytics.total"), len(meas_df))
    with col2:
        st.metric("EIS", len(meas_df[meas_df["measurement_type"] == "EIS"]))
    with col3:
        st.metric("LSV", len(meas_df[meas_df["measurement_type"] == "LSV"]))
    with col4:
        st.metric("Smart Window", len(meas_df[meas_df["measurement_type"] == "Transmittance"]))
    
    # Type distribution
    st.subheader(t("analytics.measurement_types"))
    type_counts = meas_df["measurement_type"].value_counts()
    
    fig_types = px.pie(
        values=type_counts.values,
        names=type_counts.index,
        title=t("analytics.by_type")
    )
    st.plotly_chart(fig_types, use_container_width=True)
    
    # Timeline
    st.subheader(t("analytics.over_time"))
    
    meas_df["created_date"] = pd.to_datetime(meas_df["created_at"]).dt.date
    timeline = meas_df.groupby("created_date").size().reset_index(name="count")
    
    fig_timeline = px.line(
        timeline,
        x="created_date",
        y="count",
        title=t("analytics.per_day"),
        markers=True
    )
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Results analysis (if any measurements have results)
    st.subheader(t("analytics.results_compare"))
    st.info(t("analytics.load_results"))
    
    # Demo: Show how to plot conductivity trends
    with st.expander(t("analytics.example_trends")):
        st.markdown(t("analytics.example_desc"))
        
        # Demo data
        demo_df = pd.DataFrame({
            "Salt Content (wt%)": [10, 15, 20, 25, 30],
            "σ (S/cm)": [1e-5, 5e-5, 2e-4, 1e-4, 8e-5]
        })
        
        fig_demo = px.scatter(
            demo_df,
            x="Salt Content (wt%)",
            y="σ (S/cm)",
            title="Example: Conductivity vs Salt Content",
            log_y=True
        )
        fig_demo.update_traces(mode="markers+lines")
        st.plotly_chart(fig_demo, use_container_width=True)

else:
    st.info(t("analytics.no_measurements"))
    
    st.markdown(t("analytics.getting_started"))

# Recipe comparison (placeholder)
st.subheader(t("analytics.recipe_compare"))

recipes = list_recipes()
if recipes:
    st.info(t("analytics.found_recipes", count=len(recipes)))
else:
    st.info(t("analytics.create_recipes"))
