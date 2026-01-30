"""
Dependency checking utilities for graceful error handling.

Provides functions to check if dependencies are available and show
user-friendly error messages with install instructions.
"""
import importlib
from typing import Tuple, List, Dict


# Required packages with their pip install names
REQUIRED_PACKAGES = {
    "plotly": "plotly",
    "pandas": "pandas",
    "numpy": "numpy",
    "scipy": "scipy",
    "chardet": "chardet",
    "jinja2": "Jinja2",
    "streamlit": "streamlit",
}


def check_dependency(module_name: str) -> Tuple[bool, str]:
    """
    Check if a Python module is importable.
    
    Args:
        module_name: Name of the module to check (e.g., 'plotly')
    
    Returns:
        Tuple of (is_available, error_message)
    """
    try:
        importlib.import_module(module_name)
        return True, ""
    except ImportError as e:
        pip_name = REQUIRED_PACKAGES.get(module_name, module_name)
        error_msg = (
            f"模块 '{module_name}' 未安装。\n"
            f"Module '{module_name}' is not installed.\n\n"
            f"请运行 / Please run:\n"
            f"  pip install {pip_name}\n\n"
            f"或安装所有依赖 / Or install all dependencies:\n"
            f"  pip install -r requirements.txt"
        )
        return False, error_msg


def get_install_command(package_name: str) -> str:
    """Get the pip install command for a package."""
    pip_name = REQUIRED_PACKAGES.get(package_name, package_name)
    return f"pip install {pip_name}"


def check_all_dependencies() -> Dict[str, Tuple[bool, str]]:
    """
    Check all required dependencies.
    
    Returns:
        Dict mapping module name to (is_available, version_or_error)
    """
    results = {}
    for module_name in REQUIRED_PACKAGES:
        available, error = check_dependency(module_name)
        if available:
            try:
                mod = importlib.import_module(module_name)
                version = getattr(mod, "__version__", "unknown")
                results[module_name] = (True, version)
            except Exception:
                results[module_name] = (True, "unknown")
        else:
            results[module_name] = (False, "NOT INSTALLED")
    return results


def require_plotly():
    """
    Check for plotly and show Streamlit error if missing.
    
    Call this at the top of pages that use plotly.
    Returns True if plotly is available, False otherwise (after showing error).
    """
    import streamlit as st
    
    available, error_msg = check_dependency("plotly")
    if not available:
        st.error(f"❌ **plotly 未安装 / plotly is not installed**")
        st.code("pip install plotly", language="bash")
        st.info(
            "💡 建议运行以下命令安装所有依赖：\n\n"
            "Recommended: install all dependencies:\n\n"
            "```bash\n"
            "pip install -r requirements.txt\n"
            "```"
        )
        st.stop()
        return False
    return True
