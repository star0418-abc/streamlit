"""
GPE Lab - Gel Polymer Electrolyte & Smart Window Analysis Platform

Main entry point for the Streamlit multi-page application.
"""
import streamlit as st

# Import i18n utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from utils.i18n import t, init_language, language_selector

__version__ = "0.1.0"

st.set_page_config(
    page_title="GPE Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize language and add selector to sidebar
init_language()

# Sidebar with language selector at top
with st.sidebar:
    language_selector()
    st.markdown("---")
    st.caption(f"GPE Lab v{__version__}")

st.title(t("home.title"))
st.markdown(f"""
**{t("home.subtitle")}**

{t("home.welcome")}

### {t("home.module_a")}
- {t("home.module_a_import")}
- {t("home.module_a_eis")}
- {t("home.module_a_temp")}
- {t("home.module_a_trans")}
- {t("home.module_a_lsv")}

### {t("home.module_b")}
- {t("home.module_b_sw")}

### {t("home.module_c")}
- {t("home.module_c_db")}
- {t("home.module_c_analytics")}
- {t("home.module_c_reports")}

---
*{t("common.version")} {__version__}*
""")

# Environment Check Section
with st.expander("🔧 环境检查 / Environment Check"):
    st.markdown("### 依赖状态 / Dependency Status")
    
    from utils.deps import check_all_dependencies
    
    deps_status = check_all_dependencies()
    
    # Build status data
    status_data = []
    missing_count = 0
    for pkg, (available, info) in deps_status.items():
        if available:
            status_data.append({"Package": pkg, "Status": "✅ OK", "Version": info})
        else:
            status_data.append({"Package": pkg, "Status": "❌ MISSING", "Version": "-"})
            missing_count += 1
    
    import pandas as pd
    status_df = pd.DataFrame(status_data)
    st.dataframe(status_df, use_container_width=True, hide_index=True)
    
    if missing_count > 0:
        st.warning(f"⚠️ 发现 {missing_count} 个缺失的依赖 / Found {missing_count} missing dependencies")
        st.markdown("**安装命令 / Install command:**")
        st.code("pip install -r requirements.txt", language="bash")
    else:
        st.success("✅ 所有依赖已安装 / All dependencies installed")

