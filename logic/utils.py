"""
Utility functions for the GPE Lab application.
"""
import hashlib
import numpy as np
from typing import Any
from pathlib import Path


def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file for traceability."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safe division with default for zero/invalid divisor."""
    if b == 0 or not np.isfinite(b):
        return default
    return a / b


def format_scientific(value: float, precision: int = 2) -> str:
    """Format a number in scientific notation."""
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.{precision}e}"


def format_sigma(sigma: float) -> str:
    """Format conductivity with appropriate units."""
    if sigma is None:
        return "N/A"
    if sigma >= 1e-3:
        return f"{sigma*1000:.2f} mS/cm"
    elif sigma >= 1e-6:
        return f"{sigma*1e6:.2f} µS/cm"
    else:
        return f"{sigma:.2e} S/cm"
