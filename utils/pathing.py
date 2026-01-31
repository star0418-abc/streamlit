"""
Robust path resolution utilities for GPE Lab.

Provides idempotent project root detection and sys.path management
that works both locally and on Streamlit Cloud.
"""
import sys
from pathlib import Path
from typing import Optional, Tuple

# Marker files used to identify project root (in priority order)
_MARKER_FILES = ("README.md", ".git", "pyproject.toml", "requirements.txt")


def find_project_root(
    start_path: Optional[Path] = None,
    markers: Tuple[str, ...] = _MARKER_FILES,
    max_depth: int = 10
) -> Optional[Path]:
    """
    Find the project root by walking upward until a marker file/directory is found.
    
    Args:
        start_path: Starting directory. Defaults to this file's directory.
        markers: Tuple of marker filenames to look for.
        max_depth: Maximum levels to walk up (prevents infinite loops).
        
    Returns:
        Path to project root, or None if not found.
    """
    if start_path is None:
        start_path = Path(__file__).resolve().parent
    
    current = start_path.resolve()
    
    for _ in range(max_depth):
        for marker in markers:
            if (current / marker).exists():
                return current
        
        parent = current.parent
        if parent == current:  # Reached filesystem root
            break
        current = parent
    
    return None


def get_project_root() -> Path:
    """
    Get the project root directory, with fallback to parent of utils/.
    
    Returns:
        Path to project root.
        
    Raises:
        RuntimeError: If project root cannot be determined.
    """
    root = find_project_root()
    if root is not None:
        return root
    
    # Fallback: assume utils/ is directly under project root
    fallback = Path(__file__).resolve().parent.parent
    if fallback.exists():
        return fallback
    
    raise RuntimeError(
        "Cannot determine project root. "
        "Ensure README.md or .git exists in the project directory."
    )


def ensure_project_root_in_path() -> Path:
    """
    Add project root to sys.path if not already present.
    
    This function is idempotent: calling it multiple times (e.g., on Streamlit reruns)
    will not add duplicate entries.
    
    Returns:
        Path to the project root that was added (or already present).
    """
    root = get_project_root()
    root_str = str(root)
    
    # Check if already in path (idempotent)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    
    return root
