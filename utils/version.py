"""
Version management for GPE Lab.

Provides a single source of truth for the application version,
with fallback chain: environment variable -> VERSION file -> constant.
"""
import os
from pathlib import Path
from typing import Optional

# Fallback version if all other sources fail
_FALLBACK_VERSION = "0.0.0"

# Environment variable name for override
_VERSION_ENV_VAR = "GPE_LAB_VERSION"


def _read_version_file() -> Optional[str]:
    """Read version from VERSION file at project root."""
    try:
        # Import here to avoid circular dependency
        from utils.pathing import get_project_root
        version_file = get_project_root() / "VERSION"
        
        if version_file.exists():
            content = version_file.read_text(encoding="utf-8").strip()
            if content:
                return content
    except Exception:
        pass
    
    # Fallback: try relative to this file
    try:
        version_file = Path(__file__).resolve().parent.parent / "VERSION"
        if version_file.exists():
            content = version_file.read_text(encoding="utf-8").strip()
            if content:
                return content
    except Exception:
        pass
    
    return None


def get_app_version() -> str:
    """
    Get the application version string.
    
    Resolution order:
    1. Environment variable GPE_LAB_VERSION (for CI/CD overrides)
    2. VERSION file at project root
    3. Fallback constant "0.0.0"
    
    Returns:
        Version string (e.g., "0.1.0")
    """
    # 1. Environment variable (highest priority)
    env_version = os.environ.get(_VERSION_ENV_VAR, "").strip()
    if env_version:
        return env_version
    
    # 2. VERSION file
    file_version = _read_version_file()
    if file_version:
        return file_version
    
    # 3. Fallback
    return _FALLBACK_VERSION
