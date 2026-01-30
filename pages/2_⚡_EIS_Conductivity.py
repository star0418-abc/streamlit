"""
EIS Conductivity Page - Calculate ionic conductivity from impedance spectroscopy.
"""
import streamlit as st
import pandas as pd
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

from logic.eis import (
    compute_conductivity, estimate_rb_intercept_linear,
    find_hf_intercept_direct, prepare_nyquist_data
)
from logic.utils import format_sigma
from database.db import create_measurement, update_measurement_results
from utils.i18n import t, init_language, language_selector

# Initialize language
init_language()

st.set_page_config(page_title=t("eis.page_title"), page_icon="⚡", layout="wide")

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

st.title(t("eis.title"))
st.markdown(t("eis.subtitle"))

# Check for imported data
imported = st.session_state.get("imported_data")

if imported and imported.get("type") == "EIS":
    df = imported["df"]
    st.success(t("eis.using_data", filename=imported['filename'], count=len(df)))
    
    # Prepare data for plotting
    nyquist_data = prepare_nyquist_data(df)
    
    # Nyquist plot
    st.subheader(t("eis.nyquist_plot"))
    st.markdown(t("eis.nyquist_hint"))
    
    fig = go.Figure()
    
    # Main data
    fig.add_trace(go.Scatter(
        x=nyquist_data["z_re"],
        y=nyquist_data["z_im_plot"],
        mode='markers+lines',
        name='EIS Data',
        marker=dict(size=6, color='blue'),
        line=dict(width=1, color='lightblue'),
        hovertemplate='Z_re: %{x:.2f} Ω<br>-Z_im: %{y:.2f} Ω<br>Freq: %{customdata:.1f} Hz',
        customdata=nyquist_data["freq"]
    ))
    
    fig.update_layout(
        xaxis_title=t("eis.axis_z_re"),
        yaxis_title=t("eis.axis_z_im"),
        height=500,
        showlegend=True,
        hovermode='closest'
    )
    
    # Equal aspect ratio for Nyquist
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Auto Rb estimation
    st.subheader(t("eis.rb_extraction"))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(t("eis.auto_suggestions"))
        
        # Linear extrapolation estimate
        rb_linear = estimate_rb_intercept_linear(
            nyquist_data["z_re"], 
            nyquist_data["z_im"],
            nyquist_data["freq"]
        )
        
        # Direct HF intercept
        rb_direct = find_hf_intercept_direct(
            nyquist_data["z_re"],
            nyquist_data["z_im"],
            nyquist_data["freq"]
        )
        
        if rb_linear:
            st.info(t("eis.linear_extrap", value=rb_linear))
        if rb_direct:
            st.info(t("eis.direct_hf", value=rb_direct))
    
    with col2:
        st.markdown(t("eis.manual_entry"))
        
        default_rb = rb_linear or rb_direct or 100.0
        rb_input = st.number_input(
            t("eis.rb_label"),
            min_value=0.01,
            max_value=1e9,
            value=float(default_rb),
            format="%.2f",
            help=t("eis.rb_help")
        )
    
    # Sample parameters
    st.subheader(t("eis.sample_params"))
    
    param_col1, param_col2 = st.columns(2)
    
    with param_col1:
        thickness_input = st.number_input(
            t("eis.thickness_label"),
            min_value=1e-6,
            max_value=10.0,
            value=0.01,
            format="%.4f",
            help=t("eis.thickness_help")
        )
        
        # Thickness unit helper
        thickness_um = thickness_input * 10000
        st.caption(f"= {thickness_um:.1f} µm")
    
    with param_col2:
        area_input = st.number_input(
            t("eis.area_label"),
            min_value=1e-6,
            max_value=100.0,
            value=1.0,
            format="%.4f",
            help=t("eis.area_help")
        )
    
    # Calculate conductivity
    st.subheader(t("eis.calculate_sigma"))
    
    if st.button(t("eis.btn_calculate"), type="primary"):
        result = compute_conductivity(rb_input, thickness_input, area_input)
        
        sigma = result["sigma_s_cm"]
        qc_checks = result["qc_checks"]
        
        if sigma is not None:
            st.success(f"## σ = {format_sigma(sigma)}")
            st.markdown(f"= **{sigma:.4e}** S/cm")
            
            # Show QC warnings
            if qc_checks:
                for qc in qc_checks:
                    st.warning(qc)
            
            # Store result
            st.session_state["eis_result"] = {
                "sigma_s_cm": sigma,
                "rb_ohm": rb_input,
                "thickness_cm": thickness_input,
                "area_cm2": area_input,
                "filename": imported["filename"],
                "file_hash": imported["file_hash"]
            }
            
            # Save to database option
            st.subheader(t("eis.save_result"))
            
            with st.form("save_result"):
                notes = st.text_area(t("common.notes_optional"))
                
                if st.form_submit_button(t("common.save_to_db")):
                    measurement_id = create_measurement(
                        measurement_type="EIS",
                        raw_file_path=imported["filename"],
                        raw_file_hash=imported["file_hash"],
                        import_mapping=imported["mapping"],
                        params=result["params"],
                        results={"sigma_s_cm": sigma, "qc_checks": qc_checks},
                        software_version="0.1.0"
                    )
                    st.success(t("import.saved_id", id=measurement_id))
        else:
            st.error(t("eis.calc_failed"))
            for qc in qc_checks:
                st.error(qc)

else:
    st.warning(t("eis.no_eis_data"))
    
    st.subheader(t("eis.manual_calc"))
    st.markdown(t("eis.manual_calc_hint"))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rb_manual = st.number_input(t("eis.rb_label"), value=100.0, min_value=0.01, key="rb_manual")
    with col2:
        thickness_manual = st.number_input("L (cm)", value=0.01, min_value=1e-6, format="%.4f", key="l_manual")
    with col3:
        area_manual = st.number_input("S (cm²)", value=1.0, min_value=1e-6, format="%.4f", key="s_manual")
    
    if st.button(t("common.calculate")):
        result = compute_conductivity(rb_manual, thickness_manual, area_manual)
        sigma = result["sigma_s_cm"]
        
        if sigma:
            st.success(f"## σ = {format_sigma(sigma)}")
            for qc in result["qc_checks"]:
                st.warning(qc)
