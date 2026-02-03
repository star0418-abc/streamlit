"""
Dependency checking utilities for graceful error handling.

Provides functions to check if dependencies are available, distinguish
"not installed" from "broken install" (DLL errors, version conflicts),
and show user-friendly error messages with install/repair instructions.

Design:
- Safe to import: uses only stdlib at module import time
- Streamlit imported lazily inside require_* functions only
- Catches all exceptions on import (not just ImportError)
"""
import importlib
import sys
import traceback
from typing import Dict, List, Optional, Any


# -----------------------------------------------------------------------------
# Package Registry
# -----------------------------------------------------------------------------

# Core packages: app cannot run without these
CORE_PACKAGES: Dict[str, str] = {
    "plotly": "plotly",
    "pandas": "pandas",
    "numpy": "numpy",
    "chardet": "chardet",
    "jinja2": "Jinja2",
    "streamlit": "streamlit",
}

# Optional packages: specific features degrade without them
OPTIONAL_PACKAGES: Dict[str, Dict[str, str]] = {
    "scipy": {
        "pip_name": "scipy",
        "impact": "Nonlinear fitting (VFT, RC circuit) disabled. Savitzky-Golay smoothing degrades to moving average.",
    },
}

# Combined for backwards compatibility
REQUIRED_PACKAGES: Dict[str, str] = {
    **CORE_PACKAGES,
    **{k: v["pip_name"] for k, v in OPTIONAL_PACKAGES.items()},
}


# -----------------------------------------------------------------------------
# Low-Level Check (Pure Python, No Streamlit)
# -----------------------------------------------------------------------------

def check_dependency(module_name: str) -> Dict[str, Any]:
    """
    Check if a Python module is importable and return rich diagnostic info.
    
    This function catches ALL exceptions, not just ImportError, to handle
    "installed but broken" packages (DLL load failures, ABI mismatches, etc.).
    
    Args:
        module_name: Name of the module to check (e.g., 'plotly')
    
    Returns:
        Dict with keys:
            - ok: bool - True if import succeeded
            - required: bool - True if this is a core package
            - version: str|None - Package version if available
            - error: str|None - User-readable error message
            - pip_name: str - Package name for pip install
            - impact: str|None - Feature degradation note (optional packages)
            - category: "missing" | "broken" | None
            - traceback_summary: str|None - Last ~8 lines of traceback (for bug reports)
    """
    pip_name = REQUIRED_PACKAGES.get(module_name, module_name)
    is_required = module_name in CORE_PACKAGES
    impact = None
    if module_name in OPTIONAL_PACKAGES:
        impact = OPTIONAL_PACKAGES[module_name].get("impact")
    
    base_result = {
        "ok": False,
        "required": is_required,
        "version": None,
        "error": None,
        "pip_name": pip_name,
        "impact": impact,
        "category": None,
        "traceback_summary": None,
    }
    
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "unknown")
        return {
            **base_result,
            "ok": True,
            "version": version,
        }
    
    except ImportError as e:
        # Case (a): Not installed or import chain broken
        error_msg = _format_missing_error(module_name, pip_name, str(e))
        return {
            **base_result,
            "error": error_msg,
            "category": "missing",
        }
    
    except Exception as e:
        # Case (b): Installed but broken (OSError, RuntimeError, ValueError, etc.)
        tb_lines = traceback.format_exc().splitlines()
        # Keep last 8 lines for user to paste into issues
        tb_summary = "\n".join(tb_lines[-8:]) if len(tb_lines) > 8 else "\n".join(tb_lines)
        
        error_msg = _format_broken_error(module_name, pip_name, e, tb_summary)
        return {
            **base_result,
            "error": error_msg,
            "category": "broken",
            "traceback_summary": tb_summary,
        }


def _format_missing_error(module_name: str, pip_name: str, original_error: str) -> str:
    """Format error message for missing package."""
    return (
        f"模块 '{module_name}' 未安装。\n"
        f"Module '{module_name}' is not installed.\n\n"
        f"请运行 / Please run:\n"
        f"  pip install {pip_name}\n\n"
        f"或安装所有依赖 / Or install all dependencies:\n"
        f"  pip install -r requirements.txt"
    )


def _format_broken_error(
    module_name: str,
    pip_name: str,
    exc: Exception,
    tb_summary: str,
) -> str:
    """Format error message for broken/corrupt package install."""
    exc_type = type(exc).__name__
    exc_msg = str(exc)[:200]  # Truncate very long messages
    
    repair_text = _get_repair_instructions(module_name, pip_name)
    
    return (
        f"模块 '{module_name}' 已安装但加载失败 (可能是 DLL/二进制文件损坏)。\n"
        f"Module '{module_name}' is installed but failed to load (possibly corrupt DLL/binary).\n\n"
        f"错误类型 / Error: {exc_type}: {exc_msg}\n\n"
        f"修复建议 / Repair suggestions:\n"
        f"{repair_text}\n\n"
        f"调试信息 (可粘贴到 issue) / Debug info (paste in issue):\n"
        f"```\n{tb_summary}\n```"
    )


def _get_repair_instructions(module_name: str, pip_name: str) -> str:
    """Get package-specific repair instructions."""
    # Check if conda is available
    conda_hint = ""
    try:
        # Simple heuristic: check if CONDA_PREFIX is set
        import os
        if os.environ.get("CONDA_PREFIX") or os.environ.get("CONDA_DEFAULT_ENV"):
            conda_hint = (
                f"\n\nConda 用户 / Conda users:\n"
                f"  conda uninstall {pip_name}\n"
                f"  conda install -c conda-forge {pip_name}\n"
                f"  # 或在干净环境中重装 / Or reinstall in clean env:\n"
                f"  # conda create -n fresh python=3.10 && conda activate fresh"
            )
    except Exception:
        pass
    
    return (
        f"pip 用户 / pip users:\n"
        f"  pip uninstall {pip_name}\n"
        f"  pip install --force-reinstall {pip_name}\n"
        f"  # 如仍失败，尝试重建虚拟环境 / If still failing, try rebuilding venv"
        f"{conda_hint}"
    )


# -----------------------------------------------------------------------------
# Check All Dependencies
# -----------------------------------------------------------------------------

def check_all_dependencies() -> Dict[str, Dict[str, Any]]:
    """
    Check all registered dependencies (core + optional).
    
    Returns:
        Dict mapping module name to diagnostic dict (see check_dependency).
        Each dict includes a 'required' field to distinguish core vs optional.
    """
    results = {}
    
    # Check core packages
    for module_name in CORE_PACKAGES:
        results[module_name] = check_dependency(module_name)
    
    # Check optional packages
    for module_name in OPTIONAL_PACKAGES:
        results[module_name] = check_dependency(module_name)
    
    return results


def get_install_command(package_name: str) -> str:
    """Get the pip install command for a package."""
    pip_name = REQUIRED_PACKAGES.get(package_name, package_name)
    return f"pip install {pip_name}"


# -----------------------------------------------------------------------------
# Streamlit Integration (Lazy Import)
# -----------------------------------------------------------------------------

def require_dependency(
    name: str,
    required: bool = True,
    impact: Optional[str] = None,
) -> bool:
    """
    Check a dependency and show Streamlit UI on failure.
    
    This function imports streamlit lazily (not at module level) to ensure
    deps.py can be safely imported even if streamlit is unavailable.
    
    Args:
        name: Module name to check (e.g., 'plotly', 'scipy')
        required: If True, st.stop() on failure; if False, warning only
        impact: Custom impact message (overrides registry default)
    
    Returns:
        True if dependency is available, False otherwise.
        
    Side effects:
        - If required and missing/broken: shows st.error + st.code, then st.stop()
        - If optional and missing/broken: shows st.warning, returns False
    """
    import streamlit as st
    
    result = check_dependency(name)
    
    if result["ok"]:
        return True
    
    # Determine impact message
    impact_msg = impact or result.get("impact") or ""
    category = result["category"]
    pip_name = result["pip_name"]
    
    # Build title
    if category == "missing":
        title_zh = f"❌ **{name} 未安装**"
        title_en = f"**{name} is not installed**"
    else:  # broken
        title_zh = f"⚠️ **{name} 加载失败 (安装损坏)**"
        title_en = f"**{name} failed to load (broken install)**"
    
    title = f"{title_zh} / {title_en}"
    
    # Build install/repair code block
    if category == "missing":
        code_block = f"pip install {pip_name}"
    else:
        code_block = (
            f"# Repair / 修复:\n"
            f"pip uninstall {pip_name}\n"
            f"pip install --force-reinstall {pip_name}\n"
            f"\n# Or reinstall all / 或重装所有依赖:\n"
            f"pip install -r requirements.txt --force-reinstall"
        )
    
    if required:
        st.error(title)
        st.code(code_block, language="bash")
        if result.get("traceback_summary"):
            with st.expander("🔍 调试信息 / Debug Info"):
                st.code(result["traceback_summary"], language="text")
        st.info(
            "💡 建议运行以下命令安装所有依赖：\n\n"
            "Recommended: install all dependencies:\n\n"
            "```bash\n"
            "pip install -r requirements.txt\n"
            "```"
        )
        st.stop()
        return False  # Never reached, but for type checker
    else:
        # Optional: show warning but don't stop
        warning_text = title
        if impact_msg:
            warning_text += f"\n\n**影响 / Impact**: {impact_msg}"
        warning_text += f"\n\n**安装 / Install**: `pip install {pip_name}`"
        
        st.warning(warning_text)
        if result.get("traceback_summary"):
            with st.expander("🔍 调试信息 / Debug Info"):
                st.code(result["traceback_summary"], language="text")
        
        return False


def require_plotly() -> bool:
    """
    Check for plotly and show Streamlit error if missing.
    
    Call this at the top of pages that use plotly.
    Returns True if plotly is available, False otherwise (after showing error).
    
    Note: plotly is a core requirement; this always uses required=True.
    """
    return require_dependency("plotly", required=True)


def require_scipy(impact: Optional[str] = None) -> bool:
    """
    Check for scipy and show Streamlit warning if missing.
    
    SciPy is optional. If unavailable, shows warning and returns False.
    The caller should degrade gracefully (e.g., disable VFT fitting).
    
    Args:
        impact: Custom impact message describing what won't work.
                Defaults to registry impact text.
    
    Returns:
        True if scipy is available, False otherwise (after showing warning).
    """
    return require_dependency(
        "scipy",
        required=False,
        impact=impact or OPTIONAL_PACKAGES["scipy"]["impact"],
    )


# -----------------------------------------------------------------------------
# Utility: Quick Check Without UI
# -----------------------------------------------------------------------------

def is_available(module_name: str) -> bool:
    """
    Quick check if a module is available (no UI, no detailed diagnostics).
    
    Use this for conditional logic, not for user-facing error messages.
    """
    return check_dependency(module_name)["ok"]


def get_dependency_summary() -> Dict[str, bool]:
    """
    Get a simple {module: is_available} summary for all registered packages.
    """
    return {name: is_available(name) for name in REQUIRED_PACKAGES}
