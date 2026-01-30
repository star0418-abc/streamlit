"""
Stability Window Page - Electrochemical stability from LSV.
"""
import streamlit as st
import numpy as np
from pathlib import Path

# Check plotly availability before importing
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from logic.lsv import find_onset_potential, find_stability_window
from utils.i18n import t, init_language, language_selector

# Initialize language
init_language()

st.set_page_config(page_title=t("stability.page_title"), page_icon="📈", layout="wide")

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

st.title(t("stability.title"))
st.markdown(t("stability.subtitle"))

# Check for imported data
imported = st.session_state.get("imported_data")

if imported and imported.get("type") == "LSV":
    df = imported["df"]
    st.success(t("stability.using_data", filename=imported['filename'], count=len(df)))
    
    e_v = df["e_v"].values
    j_ma_cm2 = df["j_ma_cm2"].values
    
    # Parameters
    st.subheader(t("stability.analysis_params"))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        threshold = st.number_input(
            t("stability.threshold_label"),
            value=0.1,
            min_value=0.001,
            max_value=10.0,
            format="%.3f"
        )
    
    with col2:
        smooth_window = st.slider(
            t("stability.smooth_window"),
            min_value=3,
            max_value=21,
            value=7,
            step=2
        )
    
    with col3:
        baseline_correct = st.checkbox(t("stability.baseline_correct"), value=True)
    
    # Calculate
    if st.button(t("stability.btn_find"), type="primary"):
        # Find both onsets
        result = find_stability_window(e_v, j_ma_cm2, threshold, smooth_window)
        
        ox_onset = result.get("oxidation_onset_v")
        red_onset = result.get("reduction_onset_v")
        window = result.get("window_v")
        
        # Results
        st.subheader(t("stability.results"))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if ox_onset is not None:
                st.metric(t("stability.ox_onset"), f"{ox_onset:.3f} V")
            else:
                st.warning(t("stability.ox_not_found"))
        
        with col2:
            if red_onset is not None:
                st.metric(t("stability.red_onset"), f"{red_onset:.3f} V")
            else:
                st.warning(t("stability.red_not_found"))
        
        with col3:
            if window is not None:
                st.metric(t("stability.window"), f"{window:.2f} V")
        
        # Warnings
        for w in result.get("warnings", []):
            st.warning(w)
        
        # Plot
        st.subheader(t("stability.lsv_plot"))
        
        # Get smoothed data for plotting
        ox_result = find_onset_potential(e_v, j_ma_cm2, threshold, "oxidation", smooth_window, baseline_correct)
        
        fig = go.Figure()
        
        # Raw data
        fig.add_trace(go.Scatter(
            x=e_v,
            y=j_ma_cm2,
            mode='lines',
            name='Raw data',
            line=dict(color='lightblue', width=1)
        ))
        
        # Corrected/smoothed data
        j_plot = ox_result.get("j_corrected", j_ma_cm2)
        fig.add_trace(go.Scatter(
            x=e_v,
            y=j_plot,
            mode='lines',
            name='Corrected',
            line=dict(color='blue', width=2)
        ))
        
        # Threshold lines
        fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                      annotation_text=f"+{threshold} mA/cm²")
        fig.add_hline(y=-threshold, line_dash="dash", line_color="red",
                      annotation_text=f"-{threshold} mA/cm²")
        
        # Onset markers
        if ox_onset is not None:
            fig.add_vline(x=ox_onset, line_dash="dot", line_color="orange",
                          annotation_text=f"Ox: {ox_onset:.2f}V")
        if red_onset is not None:
            fig.add_vline(x=red_onset, line_dash="dot", line_color="purple",
                          annotation_text=f"Red: {red_onset:.2f}V")
        
        fig.update_layout(
            xaxis_title=t("stability.axis_e"),
            yaxis_title=t("stability.axis_j"),
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Store result
        st.session_state["lsv_result"] = result

else:
    st.warning(t("stability.no_lsv_data"))
    
    st.info(t("stability.usage_steps"))
