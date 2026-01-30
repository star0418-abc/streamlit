"""
Smart Window Page - Electrochromic device analysis.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Check plotly availability before importing
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    make_subplots = None

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from logic.smart_window import (
    align_ca_transmittance, compute_delta_t, compute_response_time,
    compute_charge_density, compute_coloration_efficiency,
    segment_cycles_by_voltage, segment_cycles_by_transmittance,
    compute_cycling_metrics
)
from utils.i18n import t, init_language, language_selector

# Initialize language
init_language()

st.set_page_config(page_title=t("smart_window.page_title"), page_icon="🪟", layout="wide")

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

st.title(t("smart_window.title"))
st.markdown(t("smart_window.subtitle"))

# Initialize session state
if "ca_data" not in st.session_state:
    st.session_state["ca_data"] = None
if "tt_data" not in st.session_state:
    st.session_state["tt_data"] = None

st.subheader(t("smart_window.data_input"))

col1, col2 = st.columns(2)

with col1:
    st.markdown(t("smart_window.ca_data"))
    
    imported = st.session_state.get("imported_data")
    if imported and imported.get("type") == "CA":
        st.success(t("smart_window.ca_loaded", filename=imported['filename']))
        if st.button(t("smart_window.btn_use_ca")):
            st.session_state["ca_data"] = imported["df"]
            st.rerun()
    
    if st.session_state["ca_data"] is not None:
        st.info(t("smart_window.ca_points", count=len(st.session_state['ca_data'])))

with col2:
    st.markdown(t("smart_window.tt_data"))
    
    if imported and imported.get("type") == "Transmittance":
        st.success(t("smart_window.tt_loaded", filename=imported['filename']))
        if st.button(t("smart_window.btn_use_tt")):
            st.session_state["tt_data"] = imported["df"]
            st.rerun()
    
    if st.session_state["tt_data"] is not None:
        st.info(t("smart_window.tt_points", count=len(st.session_state['tt_data'])))

# Demo mode with synthetic data
with st.expander(t("smart_window.demo_data")):
    if st.button(t("smart_window.btn_load_demo")):
        # Generate synthetic EC data
        t_arr = np.linspace(0, 600, 1000)
        
        # Simulated voltage steps
        v = np.where((t_arr % 120) < 60, 1.5, -0.5)
        
        # Simulated current response
        i = 0.001 * np.exp(-t_arr % 60 / 10) * np.sign(np.diff(v, prepend=v[0]) + 0.1)
        
        # Simulated transmittance
        T = 0.3 + 0.4 * (1 - np.exp(-((t_arr % 120) / 15))) * ((t_arr % 120) < 60).astype(float)
        T += 0.4 * np.exp(-((t_arr % 120 - 60) / 15)) * ((t_arr % 120) >= 60).astype(float)
        
        st.session_state["ca_data"] = pd.DataFrame({"t_s": t_arr, "i_a": i, "v_v": v})
        st.session_state["tt_data"] = pd.DataFrame({"t_s": t_arr, "t_frac": T})
        st.success(t("smart_window.demo_loaded"))
        st.rerun()

# Analysis
ca_data = st.session_state["ca_data"]
tt_data = st.session_state["tt_data"]

if ca_data is not None and tt_data is not None:
    st.subheader(t("smart_window.parameters"))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        area = st.number_input(t("smart_window.active_area"), value=1.0, min_value=0.01)
    
    with col2:
        response_threshold = st.selectbox(t("smart_window.response_threshold"), [0.90, 0.95], index=0)
    
    with col3:
        time_tolerance = st.number_input(t("smart_window.time_tolerance"), value=0.5, min_value=0.01)
    
    # Align data
    merged_df, align_meta = align_ca_transmittance(ca_data, tt_data, time_tolerance)
    
    st.caption(t("smart_window.aligned_points", 
                merged=align_meta['merged_points'], 
                ca=align_meta['ca_points'], 
                tt=align_meta['tt_points']))
    
    # Overall metrics
    st.subheader(t("smart_window.overall_metrics"))
    
    if st.button(t("smart_window.btn_calc_metrics"), type="primary"):
        t_s = merged_df["t_s"].values
        i_a = merged_df["i_a"].values
        t_frac = merged_df["t_frac"].values
        
        # Basic metrics
        t_max = np.nanmax(t_frac)
        t_min = np.nanmin(t_frac)
        delta_t = compute_delta_t(t_max, t_min)
        
        # Charge
        q_result = compute_charge_density(t_s, i_a, area)
        
        # CE
        ce_result = compute_coloration_efficiency(t_max, t_min, q_result["q_abs_c_cm2"])
        
        # Display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("ΔT", f"{delta_t:.3f}")
            st.caption(f"({delta_t*100:.1f}%)")
        
        with col2:
            st.metric("T_bleached", f"{t_max:.3f}")
        
        with col3:
            st.metric("T_colored", f"{t_min:.3f}")
        
        with col4:
            ce = ce_result.get("ce_cm2_c")
            if ce is not None:
                st.metric("CE", f"{ce:.1f} cm²/C")
            else:
                st.metric("CE", "N/A")
        
        # Warnings
        for w in ce_result.get("warnings", []):
            st.warning(w)
        
        st.caption(f"ΔOD = {ce_result.get('delta_od', 0):.4f} (log₁₀ base)")
        st.caption(f"Q = {q_result['q_abs_c_cm2']:.4f} C/cm²")
        
        # Store results
        st.session_state["sw_result"] = {
            "delta_t": delta_t,
            "t_bleached": t_max,
            "t_colored": t_min,
            "ce": ce,
            "q_c_cm2": q_result["q_abs_c_cm2"]
        }
    
    # Visualization
    st.subheader(t("smart_window.dual_axis_plot"))
    
    t_s = merged_df["t_s"].values
    i_a = merged_df["i_a"].values
    t_frac = merged_df["t_frac"].values
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Transmittance
    fig.add_trace(
        go.Scatter(x=t_s, y=t_frac * 100, name="T (%)", line=dict(color="blue")),
        secondary_y=False
    )
    
    # Current density
    j_ma_cm2 = i_a * 1000 / area
    fig.add_trace(
        go.Scatter(x=t_s, y=j_ma_cm2, name="j (mA/cm²)", line=dict(color="red")),
        secondary_y=True
    )
    
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="T (%)", secondary_y=False)
    fig.update_yaxes(title_text="j (mA/cm²)", secondary_y=True)
    fig.update_layout(height=400)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Cycle segmentation
    st.subheader(t("smart_window.cycle_analysis"))
    
    seg_method = st.radio(t("smart_window.seg_method"), 
                          [t("smart_window.seg_voltage"), t("smart_window.seg_transmittance")])
    
    if st.button(t("smart_window.btn_segment")):
        if seg_method == t("smart_window.seg_voltage") and "v_v" in merged_df.columns:
            cycles = segment_cycles_by_voltage(merged_df["v_v"].values, t_s)
        else:
            cycles = segment_cycles_by_transmittance(t_frac, t_s)
        
        if cycles:
            st.success(t("smart_window.found_segments", count=len(cycles)))
            
            # Compute per-cycle metrics
            cycle_df = compute_cycling_metrics(cycles, t_frac, i_a, t_s, area)
            
            if len(cycle_df) > 0:
                st.dataframe(cycle_df, use_container_width=True)
                
                # Retention plot
                if len(cycle_df) > 1:
                    st.subheader(t("smart_window.cycling_retention"))
                    
                    fig_ret = go.Figure()
                    fig_ret.add_trace(go.Scatter(
                        x=cycle_df["cycle_num"],
                        y=cycle_df["delta_t"] * 100,
                        mode="markers+lines",
                        name="ΔT (%)"
                    ))
                    fig_ret.update_layout(
                        xaxis_title="Cycle",
                        yaxis_title="ΔT (%)",
                        height=300
                    )
                    st.plotly_chart(fig_ret, use_container_width=True)
        else:
            st.warning(t("smart_window.segment_failed"))

else:
    st.info(t("smart_window.load_both"))
    st.markdown(t("smart_window.steps"))
