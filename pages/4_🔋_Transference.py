"""
Transference Number Page - tLi+ calculation using Bruce-Vincent method.
"""
import streamlit as st
import numpy as np
from pathlib import Path

# Idempotent path setup (avoids duplicate insertions on reruns)
import sys
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from logic.transference import compute_transference
from utils.i18n import t, init_language, language_selector
from utils.ui_header import render_top_banner

# Initialize language
init_language()

st.set_page_config(page_title=t("transference.page_title"), page_icon="🔋", layout="wide")

# Render top banner
render_top_banner()

# Sidebar language selector
with st.sidebar:
    language_selector()

st.title(t("transference.title"))
st.markdown(f"""
{t("transference.subtitle")}

$$t_{{Li^+}} = \\frac{{I_{{ss}}(\\Delta V - I_0 R_0)}}{{I_0(\\Delta V - I_{{ss}} R_{{ss}})}}$$
""")

st.subheader(t("transference.input_params"))

col1, col2 = st.columns(2)

with col1:
    st.markdown(t("transference.current_values"))
    I0 = st.number_input(
        t("transference.i0_label"),
        value=1e-4,
        format="%.2e",
        help=t("transference.i0_help")
    )
    
    Iss = st.number_input(
        t("transference.iss_label"),
        value=5e-5,
        format="%.2e",
        help=t("transference.iss_help")
    )

with col2:
    st.markdown(t("transference.resistance_values"))
    R0 = st.number_input(
        t("transference.r0_label"),
        value=100.0,
        help=t("transference.r0_help")
    )
    
    Rss = st.number_input(
        t("transference.rss_label"),
        value=150.0,
        help=t("transference.rss_help")
    )

st.markdown(t("transference.applied_potential"))
delta_V = st.number_input(
    t("transference.delta_v_label"),
    value=0.01,
    format="%.4f",
    help=t("transference.delta_v_help")
)

# Calculate
st.subheader(t("transference.results"))

if st.button(t("transference.btn_calculate"), type="primary"):
    result = compute_transference(I0, Iss, R0, Rss, delta_V)
    
    t_li = result.get("t_li_plus")
    warnings = result.get("warnings", [])
    
    if t_li is not None:
        # Color code based on typical range
        if 0.2 <= t_li <= 0.6:
            st.success(f"## tLi⁺ = {t_li:.4f}")
            st.caption(t("transference.typical_range"))
        elif 0 <= t_li <= 1:
            st.warning(f"## tLi⁺ = {t_li:.4f}")
        else:
            st.error(f"## tLi⁺ = {t_li:.4f}")
        
        # Show warnings
        for w in warnings:
            st.warning(w)
        
        # Details
        with st.expander(t("transference.calc_details")):
            st.markdown(f"""
            - {t("transference.numerator", value=result.get('numerator', 0))}
            - {t("transference.denominator", value=result.get('denominator', 0))}
            - **I₀R₀** = {I0*R0:.4e} V
            - **IₛₛRₛₛ** = {Iss*Rss:.4e} V
            """)
    else:
        st.error(t("transference.calc_failed"))
        for w in warnings:
            st.error(w)

# Reference info
with st.expander(t("transference.method_reference")):
    st.markdown(f"""
{t("transference.method_desc")}

{t("transference.expected_values")}

{t("transference.common_issues")}
    """)
