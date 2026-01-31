"""
UI Header Utilities for GPE Lab

Provides reusable UI components for consistent page headers.
"""
import streamlit as st
from utils.i18n import t


def render_top_banner() -> None:
    """
    Render the founder/maintainer banner at the top of the page.
    
    This banner displays institutional affiliation information in a clean,
    journal-style format. It should be called after init_language() and
    before the page title on every page.
    
    The banner uses i18n keys:
    - common.founder_label / common.founder_value
    - common.maintainer_label / common.maintainer_value
    """
    founder_label = t("common.founder_label")
    founder_value = t("common.founder_value")
    maintainer_label = t("common.maintainer_label")
    maintainer_value = t("common.maintainer_value")
    
    # Clean, journal-style banner with safe inline CSS
    # - Small caption-like font (0.85rem)
    # - Muted secondary color (#6c757d)
    # - Centered flexbox layout
    # - Responsive: wraps naturally on narrow screens
    st.markdown(
        f"""
        <div style="
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            align-items: center;
            gap: 0.5rem 1.5rem;
            font-size: 0.85rem;
            color: #6c757d;
            padding: 0.5rem 0;
            text-align: center;
        ">
            <span><strong>{founder_label}:</strong> {founder_value}</span>
            <span><strong>{maintainer_label}:</strong> {maintainer_value}</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")
