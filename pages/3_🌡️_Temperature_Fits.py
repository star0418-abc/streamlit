"""
Temperature Fits Page - Arrhenius and VFT analysis.
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

# Idempotent path setup (avoids duplicate insertions on reruns)
import sys
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from logic.temperature_fits import (
    arrhenius_fit, vft_fit, apparent_ea_vft, compare_fits, generate_fit_curves
)
from logic.importers import extract_temp_from_filename
from utils.i18n import t, init_language, language_selector

# Initialize language
init_language()

st.set_page_config(page_title=t("temp_fits.page_title"), page_icon="🌡️", layout="wide")

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

st.title(t("temp_fits.title"))
st.markdown(f"""
{t("temp_fits.subtitle")}

- {t("temp_fits.arrhenius_desc")}
- {t("temp_fits.vft_desc")}
""")

# Initialize session state for temperature data
if "temp_data" not in st.session_state:
    st.session_state["temp_data"] = pd.DataFrame(columns=["filename", "temp_K", "sigma_s_cm"])

st.subheader(t("temp_fits.data_input"))

tab1, tab2 = st.tabs([t("temp_fits.tab_files"), t("temp_fits.tab_manual")])

with tab1:
    st.markdown(t("temp_fits.from_files_desc"))
    
    # Check for current EIS result
    eis_result = st.session_state.get("eis_result")
    
    if eis_result:
        filename = eis_result.get("filename", "unknown")
        sigma = eis_result.get("sigma_s_cm")
        
        # Try to extract temperature from filename
        detected_temp = extract_temp_from_filename(filename)
        
        st.info(t("temp_fits.current_eis", sigma=sigma, filename=filename))
        
        col1, col2 = st.columns(2)
        with col1:
            if detected_temp:
                st.success(t("temp_fits.detected_temp", temp_c=detected_temp-273.15, temp_k=detected_temp))
            else:
                st.warning(t("temp_fits.temp_not_detected"))
        
        with col2:
            temp_input = st.number_input(
                t("temp_fits.temp_input"),
                value=float(detected_temp) if detected_temp else 298.15,
                min_value=100.0,
                max_value=500.0
            )
        
        if st.button(t("temp_fits.btn_add_dataset")):
            new_row = pd.DataFrame({
                "filename": [filename],
                "temp_K": [temp_input],
                "sigma_s_cm": [sigma]
            })
            st.session_state["temp_data"] = pd.concat(
                [st.session_state["temp_data"], new_row], 
                ignore_index=True
            )
            st.success(t("temp_fits.added"))
            st.rerun()
    else:
        st.info(t("temp_fits.import_first"))

with tab2:
    st.markdown(t("temp_fits.manual_desc"))
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        manual_temp = st.number_input(t("temp_fits.temp_celsius"), value=25.0, key="manual_temp")
    with col2:
        manual_sigma = st.number_input(t("temp_fits.sigma_label"), value=1e-4, format="%.2e", key="manual_sigma")
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(t("temp_fits.btn_add")):
            new_row = pd.DataFrame({
                "filename": ["manual"],
                "temp_K": [manual_temp + 273.15],
                "sigma_s_cm": [manual_sigma]
            })
            st.session_state["temp_data"] = pd.concat(
                [st.session_state["temp_data"], new_row],
                ignore_index=True
            )
            st.rerun()

# Display current dataset
st.subheader(t("temp_fits.current_dataset"))

temp_df = st.session_state["temp_data"]

if len(temp_df) > 0:
    # Add display columns
    display_df = temp_df.copy()
    display_df["temp_C"] = display_df["temp_K"] - 273.15
    display_df["1000/T"] = 1000 / display_df["temp_K"]
    
    st.dataframe(
        display_df[["filename", "temp_C", "temp_K", "sigma_s_cm", "1000/T"]],
        use_container_width=True
    )
    
    if st.button(t("temp_fits.btn_clear")):
        st.session_state["temp_data"] = pd.DataFrame(columns=["filename", "temp_K", "sigma_s_cm"])
        st.rerun()
    
    # Fitting section
    if len(temp_df) >= 2:
        st.subheader(t("temp_fits.fit_results"))
        
        temps = temp_df["temp_K"].values
        sigmas = temp_df["sigma_s_cm"].values
        
        # Perform fits
        arr_result = arrhenius_fit(temps, sigmas)
        vft_result = vft_fit(temps, sigmas)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### {t('temp_fits.arrhenius_fit')}")
            if arr_result.get("success"):
                st.metric("Ea", f"{arr_result['ea_kj_mol']:.2f} kJ/mol")
                st.caption(f"= {arr_result['ea_ev']:.4f} eV")
                st.metric("R²", f"{arr_result['r_squared']:.4f}")
                st.caption(f"ln(A) = {arr_result['ln_A']:.2f}")
            else:
                st.error(arr_result.get("error", t("temp_fits.fit_failed")))
        
        with col2:
            st.markdown(f"### {t('temp_fits.vft_fit')}")
            if vft_result.get("success"):
                st.metric("B", f"{vft_result['B_K']:.1f} K")
                st.metric("T₀", f"{vft_result['T0_K']:.1f} K")
                st.metric("R²", f"{vft_result['r_squared']:.4f}")
                
                st.warning(t("temp_fits.vft_warning"))
                
                # Apparent Ea at reference temperature
                T_ref = st.number_input(
                    t("temp_fits.apparent_ea_at"),
                    value=float(temps.mean()),
                    min_value=float(vft_result['T0_K'] + 10),
                    max_value=500.0
                )
                
                ea_app = apparent_ea_vft(vft_result['B_K'], vft_result['T0_K'], T_ref)
                st.info(t("temp_fits.apparent_ea_result", temp=T_ref, ea=ea_app['ea_apparent_kj_mol']))
            else:
                st.error(vft_result.get("error", t("temp_fits.fit_failed")))
        
        # Plot
        st.subheader(t("temp_fits.arrhenius_plot"))
        
        fig = go.Figure()
        
        # Data points
        inv_T = 1000 / temps
        ln_sigma = np.log(sigmas)
        
        fig.add_trace(go.Scatter(
            x=inv_T,
            y=ln_sigma,
            mode='markers',
            name='Data',
            marker=dict(size=10, color='blue')
        ))
        
        # Fit curves
        curves = generate_fit_curves(temps, arr_result, vft_result)
        
        if "sigma_arrhenius" in curves:
            fig.add_trace(go.Scatter(
                x=1000 / curves["T_K"],
                y=np.log(curves["sigma_arrhenius"]),
                mode='lines',
                name='Arrhenius fit',
                line=dict(color='red', dash='dash')
            ))
        
        if "sigma_vft" in curves:
            fig.add_trace(go.Scatter(
                x=1000 / curves["T_K"],
                y=np.log(curves["sigma_vft"]),
                mode='lines',
                name='VFT fit',
                line=dict(color='green', dash='dot')
            ))
        
        fig.update_layout(
            xaxis_title="1000/T (K⁻¹)",
            yaxis_title="ln(σ / S·cm⁻¹)",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(t("temp_fits.add_points_hint"))
else:
    st.info(t("temp_fits.no_data_hint"))
