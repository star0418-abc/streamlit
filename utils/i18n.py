"""
Internationalization (i18n) helper module for GPE Lab.

Provides translation utilities with Chinese (zh-CN) as default and English (en) as fallback.
Designed for robustness: the app will never crash if i18n files are missing/corrupted.
"""
import json
import logging
from pathlib import Path
from typing import Optional, Any

# Configure logging for developer warnings (not user-facing)
logger = logging.getLogger(__name__)

# Path to translation files
I18N_DIR = Path(__file__).parent.parent / "i18n"

# Supported languages
LANGUAGES = {
    "zh-CN": "中文（默认）",
    "en": "English"
}

DEFAULT_LANGUAGE = "zh-CN"

# Cache for loaded translations
_translations: dict[str, dict] = {}

# Global flag: True if we're in fallback mode (JSON failed to load)
I18N_FALLBACK_ACTIVE = False

# Emergency fallback dictionary for critical keys (English)
# Used when all JSON loading fails - keeps the app functional
_EMERGENCY_FALLBACK: dict[str, str] = {
    "home.title": "GPE Lab",
    "home.subtitle": "Gel Polymer Electrolyte & Smart Window Analysis Platform",
    "home.welcome": "Welcome to GPE Lab. Use the sidebar to navigate.",
    "home.module_a": "Module A: GPE Electrochem Calculator",
    "home.module_b": "Module B: Smart Window Analysis",
    "home.module_c": "Module C: Lab Database",
    "home.module_changelog": "Update Report",
    "home.nav.import.title": "Import Data",
    "home.nav.import.desc": "Upload and map data files",
    "home.nav.eis.title": "EIS Conductivity",
    "home.nav.eis.desc": "Calculate ionic conductivity from impedance",
    "home.nav.temp_fits.title": "Temperature Fits",
    "home.nav.temp_fits.desc": "Arrhenius and VFT analysis",
    "home.nav.transference.title": "Transference",
    "home.nav.transference.desc": "Li+ transference number",
    "home.nav.stability.title": "Stability Window",
    "home.nav.stability.desc": "Electrochemical stability from LSV",
    "home.nav.smart_window.title": "Smart Window",
    "home.nav.smart_window.desc": "ΔT, response time, coloration efficiency",
    "home.nav.database.title": "Lab Database",
    "home.nav.database.desc": "Recipes, batches, samples management",
    "home.nav.analytics.title": "Analytics",
    "home.nav.analytics.desc": "Trends and composition analysis",
    "home.nav.reports.title": "Reports",
    "home.nav.reports.desc": "Export measurement reports",
    "home.nav.changelog.title": "Update Report",
    "home.nav.changelog.desc": "View version changelog",
    "common.navigate": "Go",
    "common.version": "Version",
    "common.founder_label": "Founder",
    "common.maintainer_label": "Maintainer",
    "common.founder_value": "USTC-National Synchrotron Radiation Laboratory",
    "common.maintainer_value": "HENU",
    "sidebar.language_label": "Language",
    "warnings.i18n_fallback": "⚠️ Translation files failed to load. Using fallback English.",
}


def _load_translations(lang: str) -> dict:
    """Load translation file for a language."""
    global I18N_FALLBACK_ACTIVE
    
    if lang in _translations:
        return _translations[lang]
    
    file_path = I18N_DIR / f"{lang}.json"
    
    if not file_path.exists():
        logger.warning(f"Translation file not found: {file_path}")
        return {}
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            _translations[lang] = json.load(f)
        return _translations[lang]
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        I18N_FALLBACK_ACTIVE = True
        return {}
    except Exception as e:
        logger.error(f"Failed to load translations from {file_path}: {e}")
        I18N_FALLBACK_ACTIVE = True
        return {}


def init_language() -> None:
    """
    Initialize language in session state. Call at app startup.
    
    NOTE: This function uses Streamlit session_state, so it must only be called
    after st.set_page_config. Import this module freely; calling this function
    is what triggers st.* access.
    """
    import streamlit as st
    if "language" not in st.session_state:
        st.session_state["language"] = DEFAULT_LANGUAGE


def get_current_language() -> str:
    """Get the current language code."""
    try:
        import streamlit as st
        return st.session_state.get("language", DEFAULT_LANGUAGE)
    except Exception:
        return DEFAULT_LANGUAGE


def set_language(lang: str) -> None:
    """Set the current language."""
    import streamlit as st
    if lang in LANGUAGES:
        st.session_state["language"] = lang
    else:
        logger.warning(f"Unknown language: {lang}, using default")
        st.session_state["language"] = DEFAULT_LANGUAGE


def t(key: str, default: Optional[str] = None, **kwargs) -> str:
    """
    Translate a key to the current language.
    
    Args:
        key: Dot-separated key path, e.g., "home.title" or "import.btn_save"
        default: Optional default value if key is not found. If None, returns the key itself.
        **kwargs: Format arguments for string interpolation
        
    Returns:
        Translated string, or default if provided, or raw key if not found.
        
    Example:
        t("import.import_success", count=100)  -> "✅ 成功导入 100 行"
        t("nonexistent.key", default="Fallback")  -> "Fallback"
    """
    lang = get_current_language()
    
    # Try current language first
    result = _get_translation(lang, key)
    
    # Fall back to English if not found
    if result is None and lang != "en":
        result = _get_translation("en", key)
        if result is not None:
            logger.debug(f"Translation key '{key}' not found in {lang}, using English fallback")
    
    # Fall back to emergency dictionary
    if result is None:
        result = _EMERGENCY_FALLBACK.get(key)
        if result is not None:
            logger.debug(f"Translation key '{key}' found in emergency fallback")
    
    # Fall back to provided default or raw key
    if result is None:
        if default is not None:
            return default
        logger.warning(f"Translation key '{key}' not found in any language")
        return key
    
    # Apply format arguments
    if kwargs:
        try:
            result = result.format(**kwargs)
        except (KeyError, ValueError) as e:
            logger.warning(f"Failed to format translation '{key}': {e}")
    
    return result


def _get_translation(lang: str, key: str) -> Optional[str]:
    """Get a translation value by dot-separated key path."""
    translations = _load_translations(lang)
    
    if not translations:
        return None
    
    # Navigate the nested dict
    parts = key.split(".")
    value: Any = translations
    
    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return None
    
    return value if isinstance(value, str) else None


def language_selector() -> None:
    """
    Render a language selector in the sidebar.
    Should be called after st.set_page_config().
    """
    import streamlit as st
    
    init_language()
    
    current_lang = get_current_language()
    
    # Get display names
    options = list(LANGUAGES.values())
    lang_codes = list(LANGUAGES.keys())
    
    # Find current index
    current_idx = lang_codes.index(current_lang) if current_lang in lang_codes else 0
    
    selected_display = st.sidebar.selectbox(
        t("sidebar.language_label"),
        options,
        index=current_idx,
        key="language_selector"
    )
    
    # Map back to language code
    selected_idx = options.index(selected_display)
    selected_lang = lang_codes[selected_idx]
    
    if selected_lang != current_lang:
        set_language(selected_lang)
        st.rerun()


def get_language_display_name(lang: str) -> str:
    """Get the display name for a language code."""
    return LANGUAGES.get(lang, lang)


def is_fallback_active() -> bool:
    """Check if i18n is running in fallback mode (JSON failed to load)."""
    return I18N_FALLBACK_ACTIVE
