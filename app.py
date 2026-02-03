"""
GPE Lab - Gel Polymer Electrolyte & Smart Window Analysis Platform

Main entry point for the Streamlit multi-page application.

ROBUSTNESS DESIGN:
- st.set_page_config is the FIRST Streamlit command
- All critical imports are wrapped in try/except
- The app renders even if i18n/header fails (shows warning banner)
- Navigation is data-driven from utils/pages.py
"""
import sys
from pathlib import Path

# Idempotent path setup: find project root and add to sys.path if not present
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# === STREAMLIT IMPORT AND PAGE CONFIG (MUST BE FIRST) ===
import streamlit as st

# Version retrieval - pure function, no st.* calls
try:
    from utils.version import get_app_version
    __version__ = get_app_version()
except Exception:
    __version__ = "0.0.0"

st.set_page_config(
    page_title="GPE Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === CSS FALLBACK: Hide default sidebar nav for older Streamlit versions ===
# config.toml has showSidebarNavigation=false, but older versions ignore it
st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# === SAFE IMPORTS WITH FALLBACKS ===
_IMPORT_WARNINGS: list[str] = []
_I18N_AVAILABLE = False
_HEADER_AVAILABLE = False
_PAGES_AVAILABLE = False

# Import i18n with fallback
try:
    from utils.i18n import t, init_language, language_selector, is_fallback_active
    _I18N_AVAILABLE = True
except Exception as e:
    _IMPORT_WARNINGS.append(f"i18n module failed: {e}")
    # Fallback t() function
    def t(key: str, default: str | None = None, **kwargs) -> str:
        return default if default else key
    def init_language() -> None:
        pass
    def language_selector() -> None:
        st.sidebar.caption("Language: English (fallback)")
    def is_fallback_active() -> bool:
        return True

# Import UI header with fallback
try:
    from utils.ui_header import render_top_banner
    _HEADER_AVAILABLE = True
except Exception as e:
    _IMPORT_WARNINGS.append(f"ui_header module failed: {e}")
    def render_top_banner() -> None:
        pass  # Silently skip banner

# Import page registry with fallback
try:
    from utils.pages import NAV_PAGES, get_pages_by_section
    _PAGES_AVAILABLE = True
except Exception as e:
    _IMPORT_WARNINGS.append(f"pages module failed: {e}")
    # Minimal fallback page registry
    from typing import NamedTuple
    class PageDef(NamedTuple):
        id: str
        path: str
        icon: str
        section: str
    NAV_PAGES = (
        PageDef("import", "pages/1_📊_Import_Data.py", "📊", "a"),
    )
    def get_pages_by_section(section: str) -> list:
        return [p for p in NAV_PAGES if p.section == section]

# === INITIALIZE LANGUAGE ===
init_language()

# === RENDER HEADER (safe) ===
render_top_banner()

# === SIDEBAR ===
with st.sidebar:
    language_selector()
    st.markdown("---")
    st.caption(f"GPE Lab v{__version__}")

# === WARNING BANNERS ===
# Show warnings if we're in fallback mode
if _IMPORT_WARNINGS:
    with st.expander("⚠️ Initialization Warnings", expanded=True):
        for warning in _IMPORT_WARNINGS:
            st.warning(warning)
        st.info("The app is running in fallback mode. Some features may be limited.")

if _I18N_AVAILABLE and is_fallback_active():
    st.warning(t("warnings.i18n_fallback", default="⚠️ Translation files failed to load. Using fallback English."))

# === HOME PAGE CONTENT ===
st.title(t("home.title", default="🔬 GPE Lab"))
st.markdown(f"**{t('home.subtitle', default='Gel Polymer Electrolyte & Smart Window Analysis Platform')}**")
st.markdown(t("home.welcome", default="Welcome to GPE Lab. Use the navigation below to access modules:"))

# === NAVIGATION DASHBOARD ===
# Check for st.page_link availability (Streamlit >= 1.31)
_HAS_PAGE_LINK = hasattr(st, "page_link")


def _render_nav_card(page_id: str, icon: str, page_path: str) -> None:
    """Render a navigation card with icon, title, and link.
    
    Uses structured i18n keys: home.nav.<page_id>.title and home.nav.<page_id>.desc
    """
    title = t(f"home.nav.{page_id}.title", default=page_id.replace("_", " ").title())
    desc = t(f"home.nav.{page_id}.desc", default="")
    nav_label = t("common.navigate", default="Go")
    
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
                <div style="color: #888; font-size: 0.9rem; margin-top: 0.3rem;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if _HAS_PAGE_LINK:
            st.page_link(page_path, label=f"➡️ {nav_label}", use_container_width=True)
        else:
            # Fallback for older Streamlit: show path hint
            st.caption(f"📂 {page_path}")


def _render_section(section_key: str, section_i18n_key: str, columns: int = 2) -> None:
    """Render a navigation section with its pages."""
    st.markdown(f"### {t(f'home.{section_i18n_key}', default=section_key.upper())}")
    pages = get_pages_by_section(section_key)
    
    if not pages:
        return
    
    if columns == 1 or len(pages) == 1:
        for page in pages:
            with st.container(border=True):
                _render_nav_card(page.id, page.icon, page.path)
    else:
        cols = st.columns(columns)
        for i, page in enumerate(pages):
            with cols[i % columns]:
                with st.container(border=True):
                    _render_nav_card(page.id, page.icon, page.path)


# === RENDER ALL SECTIONS ===
_render_section("a", "module_a", columns=2)
_render_section("b", "module_b", columns=1)
_render_section("c", "module_c", columns=3)
_render_section("changelog", "module_changelog", columns=1)

st.markdown("---")
st.caption(f"*{t('common.version', default='Version')} {__version__}*")

# === ENVIRONMENT CHECK SECTION (lazy, runs only on button click) ===
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
