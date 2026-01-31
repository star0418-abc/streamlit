"""
GPE Lab - Gel Polymer Electrolyte & Smart Window Analysis Platform

Main entry point for the Streamlit multi-page application.
"""
import streamlit as st

# Bootstrap project path (must be first, before other local imports)
import sys
from pathlib import Path

# Idempotent path setup: find project root and add to sys.path if not present
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Now import local modules
from utils.i18n import t, init_language, language_selector
from utils.version import get_app_version

__version__ = get_app_version()

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

# ----- Home Page Title -----
st.title(t("home.title"))
st.markdown(f"**{t('home.subtitle')}**")
st.markdown(t("home.welcome"))

# ----- Navigation Dashboard -----
# Check for st.page_link availability (Streamlit >= 1.31)
_HAS_PAGE_LINK = hasattr(st, "page_link")


def _render_nav_card(icon: str, title: str, description: str, page_path: str) -> None:
    """Render a navigation card with icon, title, and link."""
    with st.container():
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, rgba(100,100,255,0.05), rgba(150,100,255,0.08));
                border: 1px solid rgba(150,150,200,0.2);
                border-radius: 10px;
                padding: 1rem;
                margin-bottom: 0.5rem;
                transition: all 0.2s ease;
            ">
                <span style="font-size: 1.5rem;">{icon}</span>
                <strong style="margin-left: 0.5rem;">{title}</strong>
                <div style="color: #888; font-size: 0.9rem; margin-top: 0.3rem;">{description}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if _HAS_PAGE_LINK:
            st.page_link(page_path, label=f"➡️ {t('common.navigate') if 'common.navigate' in t('common.navigate') else 'Go'}", use_container_width=True)
        else:
            # Fallback for older Streamlit: show path hint
            st.caption(f"📂 {page_path}")


# Module A: GPE Electrochem Calculator
st.markdown(f"### {t('home.module_a')}")
col_a1, col_a2 = st.columns(2)

with col_a1:
    with st.container(border=True):
        st.markdown("📊 **" + t("home.module_a_import").replace("**📊 ", "").replace("**", "") + "**")
        if _HAS_PAGE_LINK:
            st.page_link("pages/1_📊_Import_Data.py", label="➡️ " + t("home.module_a_import").split(":")[0].replace("**", "").strip(), use_container_width=True)
    
    with st.container(border=True):
        st.markdown("🌡️ **" + t("home.module_a_temp").replace("**🌡️ ", "").replace("**", "") + "**")
        if _HAS_PAGE_LINK:
            st.page_link("pages/3_🌡️_Temperature_Fits.py", label="➡️ " + t("home.module_a_temp").split(":")[0].replace("**", "").strip(), use_container_width=True)
    
    with st.container(border=True):
        st.markdown("📈 **" + t("home.module_a_lsv").replace("**📈 ", "").replace("**", "") + "**")
        if _HAS_PAGE_LINK:
            st.page_link("pages/5_📈_Stability_Window.py", label="➡️ " + t("home.module_a_lsv").split(":")[0].replace("**", "").strip(), use_container_width=True)

with col_a2:
    with st.container(border=True):
        st.markdown("⚡ **" + t("home.module_a_eis").replace("**⚡ ", "").replace("**", "") + "**")
        if _HAS_PAGE_LINK:
            st.page_link("pages/2_⚡_EIS_Conductivity.py", label="➡️ " + t("home.module_a_eis").split(":")[0].replace("**", "").strip(), use_container_width=True)
    
    with st.container(border=True):
        st.markdown("🔋 **" + t("home.module_a_trans").replace("**🔋 ", "").replace("**", "") + "**")
        if _HAS_PAGE_LINK:
            st.page_link("pages/4_🔋_Transference.py", label="➡️ " + t("home.module_a_trans").split(":")[0].replace("**", "").strip(), use_container_width=True)

# Module B: Smart Window Analysis
st.markdown(f"### {t('home.module_b')}")
with st.container(border=True):
    st.markdown("🪟 **" + t("home.module_b_sw").replace("**🪟 ", "").replace("**", "") + "**")
    if _HAS_PAGE_LINK:
        st.page_link("pages/6_🪟_Smart_Window.py", label="➡️ " + t("home.module_b_sw").split(":")[0].replace("**", "").strip(), use_container_width=True)

# Module C: Lab Database
st.markdown(f"### {t('home.module_c')}")
col_c1, col_c2, col_c3 = st.columns(3)

with col_c1:
    with st.container(border=True):
        st.markdown("🗃️ **" + t("home.module_c_db").replace("**🗃️ ", "").replace("**", "") + "**")
        if _HAS_PAGE_LINK:
            st.page_link("pages/7_🗃️_Lab_Database.py", label="➡️ " + t("home.module_c_db").split(":")[0].replace("**", "").strip(), use_container_width=True)

with col_c2:
    with st.container(border=True):
        st.markdown("📉 **" + t("home.module_c_analytics").replace("**📉 ", "").replace("**", "") + "**")
        if _HAS_PAGE_LINK:
            st.page_link("pages/8_📉_Analytics.py", label="➡️ " + t("home.module_c_analytics").split(":")[0].replace("**", "").strip(), use_container_width=True)

with col_c3:
    with st.container(border=True):
        st.markdown("📝 **" + t("home.module_c_reports").replace("**📝 ", "").replace("**", "") + "**")
        if _HAS_PAGE_LINK:
            st.page_link("pages/9_📝_Reports.py", label="➡️ " + t("home.module_c_reports").split(":")[0].replace("**", "").strip(), use_container_width=True)

# Changelog
st.markdown(f"### {t('home.module_changelog')}")
with st.container(border=True):
    st.markdown("📋 **" + t("home.module_changelog").replace("**📋 ", "").replace("**", "") + "**")
    if _HAS_PAGE_LINK:
        st.page_link("pages/10_📋_Update_Report.py", label="➡️ " + t("changelog.title"), use_container_width=True)

st.markdown("---")
st.caption(f"*{t('common.version')} {__version__}*")

# ----- Environment Check Section (runs only on button click) -----
with st.expander("🔧 环境检查 / Environment Check"):
    st.markdown("### 依赖状态 / Dependency Status")
    st.info("💡 点击下方按钮运行诊断 / Click the button below to run diagnostics")
    
    if st.button("🔧 运行诊断 / Run Diagnostics", key="run_diagnostics"):
        try:
            @st.cache_data(ttl=60)
            def _cached_dependency_check():
                """Cached dependency check to avoid repeated work."""
                from utils.deps import check_all_dependencies
                deps_status = check_all_dependencies()
                status_data = []
                missing_count = 0
                for pkg, (available, info) in deps_status.items():
                    if available:
                        status_data.append({"Package": pkg, "Status": "✅ OK", "Version": info})
                    else:
                        status_data.append({"Package": pkg, "Status": "❌ MISSING", "Version": "-"})
                        missing_count += 1
                return status_data, missing_count
            
            status_data, missing_count = _cached_dependency_check()
            
            # Display without pandas (st.dataframe accepts list[dict])
            st.dataframe(status_data, use_container_width=True, hide_index=True)
            
            if missing_count > 0:
                st.warning(f"⚠️ 发现 {missing_count} 个缺失的依赖 / Found {missing_count} missing dependencies")
                st.markdown("**安装命令 / Install command:**")
                st.code("pip install -r requirements.txt", language="bash")
            else:
                st.success("✅ 所有依赖已安装 / All dependencies installed")
            
            # Database diagnostics
            st.markdown("---")
            st.markdown("### 数据库状态 / Database Status")
            
            try:
                from database.db import get_diagnostics
                diag = get_diagnostics()
                
                col1, col2 = st.columns(2)
                with col1:
                    env_emoji = "☁️" if diag["runtime_env"] == "cloud" else "💻"
                    st.info(f"{env_emoji} **环境 / Environment**: {diag['runtime_env'].upper()}")
                    st.text(f"工作目录 / CWD: {diag['cwd']}")
                    st.text(f"项目根 / Root: {diag['project_root']}")
                
                with col2:
                    db_emoji = "✅" if diag["db_exists"] else "⚠️"
                    write_emoji = "✅" if diag["db_writable"] else "❌"
                    st.info(f"📁 **数据库 / Database**: {diag['db_path']}")
                    st.text(f"文件存在 / Exists: {db_emoji} {diag['db_exists']}")
                    st.text(f"可写入 / Writable: {write_emoji} {diag['db_writable']}")
                
                if diag["init_error"]:
                    st.error(f"❌ 初始化错误 / Init Error: {diag['init_error']}")
                
                if diag["tables"]:
                    st.markdown("**表格行数 / Table Row Counts:**")
                    table_data = [{"Table": k, "Rows": v} for k, v in diag["tables"].items()]
                    st.dataframe(table_data, use_container_width=True, hide_index=True)
                elif diag["db_exists"]:
                    st.caption("无表格 / No tables found")
                    
            except Exception as e:
                st.error(f"❌ 无法获取数据库诊断信息 / Cannot get DB diagnostics: {e}")
                
        except Exception as e:
            st.error(f"❌ 诊断失败 / Diagnostics failed: {e}")
            st.info("💡 这可能是因为某些依赖未安装 / This may be due to missing dependencies")
