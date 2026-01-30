"""
Internationalization (i18n) helper module for GPE Lab.

Provides translation utilities with Chinese (zh-CN) as default and English (en) as fallback.
"""
import json
import logging
from pathlib import Path
from typing import Optional, Any

import streamlit as st

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


def _load_translations(lang: str) -> dict:
    """Load translation file for a language."""
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
    except Exception as e:
        logger.error(f"Failed to load translations from {file_path}: {e}")
        return {}


def init_language() -> None:
    """Initialize language in session state. Call at app startup."""
    if "language" not in st.session_state:
        st.session_state["language"] = DEFAULT_LANGUAGE


def get_current_language() -> str:
    """Get the current language code."""
    return st.session_state.get("language", DEFAULT_LANGUAGE)


def set_language(lang: str) -> None:
    """Set the current language."""
    if lang in LANGUAGES:
        st.session_state["language"] = lang
    else:
        logger.warning(f"Unknown language: {lang}, using default")
        st.session_state["language"] = DEFAULT_LANGUAGE


def t(key: str, **kwargs) -> str:
    """
    Translate a key to the current language.
    
    Args:
        key: Dot-separated key path, e.g., "home.title" or "import.btn_save"
        **kwargs: Format arguments for string interpolation
        
    Returns:
        Translated string, or fallback to English, or raw key if not found.
        
    Example:
        t("import.import_success", count=100)  -> "✅ 成功导入 100 行"
    """
    lang = get_current_language()
    
    # Try current language first
    result = _get_translation(lang, key)
    
    # Fall back to English if not found
    if result is None and lang != "en":
        result = _get_translation("en", key)
        if result is not None:
            logger.debug(f"Translation key '{key}' not found in {lang}, using English fallback")
    
    # Fall back to raw key if still not found
    if result is None:
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
    Should be called at the top of app.py.
    """
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
