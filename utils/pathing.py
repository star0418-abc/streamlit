"""
Robust path resolution utilities for GPE Lab.

Provides idempotent project root detection and sys.path management
that works both locally and on Streamlit Cloud (Windows/macOS/Linux).

DESIGN NOTES:
- Uses ONLY Python standard library (no third-party imports).
- NEVER calls any Streamlit APIs.
- Marker strategy: requires multiple unique markers to avoid false positives
  from docs/README.md or nested requirements.txt.
- sys.path idempotency uses normcase/normpath for cross-platform consistency.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Set, Tuple

# =============================================================================
# MARKER CONFIGURATION
# =============================================================================
# Primary markers: files/dirs that are highly unique to the project root.
# We require AT LEAST 2 markers to hit before accepting a directory as root.
# This prevents false positives from docs/README.md, nested requirements.txt, etc.
_PRIMARY_MARKERS: Tuple[str, ...] = (
    "app.py",       # Streamlit entry point
    "VERSION",      # Single source of version truth
    "pages",        # Streamlit multipage directory
    ".streamlit",   # Streamlit config directory
    ".git",         # Git repository root
)

# Secondary markers: common but less unique (only used if primary markers fail)
_SECONDARY_MARKERS: Tuple[str, ...] = (
    "pyproject.toml",
    "requirements.txt",
)

# Minimum number of primary markers required to accept a directory as root
_MIN_MARKER_HITS = 2


# =============================================================================
# PATH NORMALIZATION HELPERS
# =============================================================================
def _normalize_path(p: Path) -> str:
    """
    Normalize a path for consistent comparison across platforms.
    
    Handles:
    - Case differences on Windows (C:/Foo vs c:/foo)
    - Path separators (/ vs \\)
    - Symlinks (resolved to real path)
    - Relative vs absolute paths
    
    Args:
        p: Path to normalize.
        
    Returns:
        Normalized path string suitable for comparison.
    """
    # resolve() handles symlinks and makes path absolute
    # os.path.normcase() lowercases on Windows, no-op on Unix
    # os.path.normpath() normalizes separators and removes redundant parts
    try:
        resolved = p.resolve()
    except OSError:
        # Fallback for paths that can't be resolved (e.g., broken symlinks)
        resolved = p.absolute()
    return os.path.normcase(os.path.normpath(str(resolved)))


def _get_normalized_sys_path() -> Set[str]:
    """
    Get all current sys.path entries as a normalized set.
    
    Returns:
        Set of normalized path strings for comparison.
    """
    normalized = set()
    for entry in sys.path:
        if entry:  # Skip empty strings
            try:
                normalized.add(_normalize_path(Path(entry)))
            except (OSError, ValueError):
                # Skip malformed paths
                pass
    return normalized


# =============================================================================
# MARKER DETECTION
# =============================================================================
def _count_marker_hits(directory: Path, markers: Tuple[str, ...]) -> int:
    """
    Count how many markers exist in the given directory.
    
    Args:
        directory: Directory to check.
        markers: Tuple of marker names (files or directories).
        
    Returns:
        Number of markers found.
    """
    hits = 0
    for marker in markers:
        marker_path = directory / marker
        if marker_path.exists():
            hits += 1
    return hits


def _is_valid_project_root(directory: Path) -> bool:
    """
    Check if directory is a valid project root.
    
    Validation strategy:
    1. Count primary marker hits (app.py, VERSION, pages/, .streamlit/, .git)
    2. Accept if >= _MIN_MARKER_HITS primary markers found
    3. If exactly 1 primary marker, require at least 1 secondary marker
    
    This prevents false positives from:
    - docs/README.md (no app.py, no pages/)
    - nested requirements.txt (no app.py, no VERSION)
    
    Args:
        directory: Directory to validate.
        
    Returns:
        True if directory appears to be a valid project root.
    """
    primary_hits = _count_marker_hits(directory, _PRIMARY_MARKERS)
    
    if primary_hits >= _MIN_MARKER_HITS:
        return True
    
    if primary_hits == 1:
        # One primary marker + at least one secondary = acceptable
        secondary_hits = _count_marker_hits(directory, _SECONDARY_MARKERS)
        return secondary_hits >= 1
    
    # No primary markers at all = not a valid root
    return False


# =============================================================================
# PROJECT ROOT DETECTION (PUBLIC API)
# =============================================================================
@lru_cache(maxsize=1)
def find_project_root(
    start_path: Optional[str] = None,
    max_depth: int = 10
) -> Optional[Path]:
    """
    Find the project root by walking upward until validated markers are found.
    
    Uses a multi-marker strategy to avoid false positives from nested files
    like docs/README.md or tests/requirements.txt.
    
    Args:
        start_path: Starting directory (as string for cache hashability).
            Defaults to this file's directory.
        max_depth: Maximum levels to walk up (prevents infinite loops).
            Configurable but defaults to 10 which should cover most layouts.
        
    Returns:
        Path to project root if found and validated, None otherwise.
        
    Note:
        Result is cached. The cache uses start_path (or None) as key.
        For most uses, call with default args for consistent caching.
    """
    if start_path is None:
        current = Path(__file__).resolve().parent
    else:
        current = Path(start_path).resolve()
    
    for _ in range(max_depth):
        if _is_valid_project_root(current):
            return current
        
        parent = current.parent
        if parent == current:  # Reached filesystem root
            break
        current = parent
    
    return None


def get_project_root() -> Path:
    """
    Get the project root directory with validated fallback.
    
    Unlike find_project_root(), this function:
    1. First tries find_project_root() with marker validation
    2. Falls back to parent of utils/ (common layout) BUT validates
       that fallback also meets marker requirements
    3. Raises RuntimeError if neither approach yields a valid root
    
    Returns:
        Path to validated project root.
        
    Raises:
        RuntimeError: If project root cannot be determined or validated.
            Error message includes diagnostic info for debugging.
    """
    root = find_project_root()
    if root is not None:
        return root
    
    # Fallback: assume utils/ is directly under project root
    # This handles edge cases like running from an installed package
    fallback = Path(__file__).resolve().parent.parent
    
    # CRITICAL: Validate fallback to avoid silent misconfiguration
    # The fallback must contain app.py OR pages/ to be considered valid
    # This prevents returning a random parent directory
    if fallback.exists():
        has_app = (fallback / "app.py").exists()
        has_pages = (fallback / "pages").is_dir()
        
        if has_app or has_pages:
            return fallback
    
    # If we get here, neither approach worked - raise with diagnostics
    this_file = Path(__file__).resolve()
    primary_str = ", ".join(_PRIMARY_MARKERS)
    raise RuntimeError(
        f"Cannot determine project root.\n"
        f"  Searched from: {this_file.parent}\n"
        f"  Expected markers (need {_MIN_MARKER_HITS}+): {primary_str}\n"
        f"  Fallback ({fallback}) missing app.py and pages/.\n"
        f"Ensure you're running from within the GPE Lab project structure."
    )


# =============================================================================
# SYS.PATH MANAGEMENT (PUBLIC API)
# =============================================================================
def ensure_project_root_in_path() -> Path:
    """
    Add project root to sys.path if not already present.
    
    This function is TRULY idempotent across all platforms:
    - Windows: case-insensitive comparison (C:/X == c:/x)
    - All platforms: normalized separators and resolved symlinks
    - Repeated calls never add duplicate entries
    
    The function is safe to call on every Streamlit rerun.
    
    Returns:
        Path to the project root (added or already present).
        
    Raises:
        RuntimeError: If project root cannot be determined (propagated from
            get_project_root()).
    """
    root = get_project_root()
    root_normalized = _normalize_path(root)
    
    # Get all current sys.path entries normalized for comparison
    existing_normalized = _get_normalized_sys_path()
    
    # Only insert if not already present (normalized comparison)
    if root_normalized not in existing_normalized:
        sys.path.insert(0, str(root))
    
    return root


# =============================================================================
# UTILITIES (PUBLIC API)
# =============================================================================
def get_data_path() -> Path:
    """
    Get the data directory path, handling Cloud vs local environments.
    
    On Streamlit Cloud, writable paths are restricted to /tmp.
    Locally, uses data/ under project root.
    
    Returns:
        Path to data directory (may not exist yet).
    """
    # Check if running on Streamlit Cloud (STREAMLIT_SHARING env var)
    if os.environ.get("STREAMLIT_SHARING") or os.environ.get("STREAMLIT_SERVER_PORT"):
        # Cloud environment: use /tmp for writable data
        return Path("/tmp")
    
    # Local environment: use data/ under project root
    return get_project_root() / "data"


# =============================================================================
# CACHE MANAGEMENT
# =============================================================================
def clear_root_cache() -> None:
    """
    Clear the cached project root.
    
    Useful for testing or if the directory structure changes at runtime
    (rare in production).
    """
    find_project_root.cache_clear()
