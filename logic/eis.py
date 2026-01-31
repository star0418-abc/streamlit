"""
EIS (Electrochemical Impedance Spectroscopy) analysis module.

Provides Rb extraction, conductivity calculation, and QC checks for SS/GPE/SS cells.

SIGN CONVENTION:
- z_im_cap: capacitive semicircle is NEGATIVE (internal computation)
- z_im_plot: Nyquist display uses -z_im_cap (positive upper half)

HF INDUCTIVE HANDLING:
- Lead inductance can cause >100 kHz artifacts (positive z_im in capacitive convention)
- Rb extraction automatically drops initial HF inductive points

SCIPY OPTIONAL:
- Basic Rb extraction works without SciPy
- fit_simple_rc() requires SciPy; returns {success: False} if missing
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any

# =============================================================================
# Optional SciPy import (safe for Streamlit Cloud without scipy)
# =============================================================================
SCIPY_AVAILABLE = False
curve_fit = None

try:
    from scipy.optimize import curve_fit as _curve_fit
    curve_fit = _curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# Sign Convention Helpers
# =============================================================================

def _standardize_sign_capacitive(
    z_im: np.ndarray, 
    freq: np.ndarray,
    min_magnitude_threshold: float = 1.0
) -> Tuple[np.ndarray, bool, str]:
    """
    Ensure z_im follows capacitive-negative convention.
    
    Uses median of mid-frequency band (exclude top/bottom 10%) with minimum 
    magnitude check to avoid flipping noise-dominated data.
    
    Args:
        z_im: Imaginary impedance array
        freq: Frequency array (Hz)
        min_magnitude_threshold: Minimum |median| to trigger auto-flip (Ω)
    
    Returns:
        (z_im_cap, was_flipped, warning_msg)
        - z_im_cap: Sign-standardized imaginary impedance (capacitive = negative)
        - was_flipped: True if the sign was inverted
        - warning_msg: Non-empty if ambiguous or below magnitude threshold
    """
    if len(z_im) < 5:
        # Too few points for robust estimate
        return z_im.copy(), False, "Too few points for robust sign detection"
    
    # Sort by frequency
    sort_idx = np.argsort(freq)
    z_im_sorted = z_im[sort_idx]
    
    # Use mid-frequency band: exclude top and bottom 10%
    n = len(z_im_sorted)
    start_idx = max(1, int(n * 0.1))
    end_idx = min(n - 1, int(n * 0.9))
    
    if end_idx <= start_idx:
        # Fallback if range too small
        mid_band = z_im_sorted
    else:
        mid_band = z_im_sorted[start_idx:end_idx]
    
    median_val = np.median(mid_band)
    warning = ""
    
    # Check magnitude threshold
    if np.abs(median_val) < min_magnitude_threshold:
        warning = f"Ambiguous sign: |median z_im| = {np.abs(median_val):.2f} Ω < threshold"
        # Don't flip if magnitude too small - could be noise
        return z_im.copy(), False, warning
    
    # Convention: capacitive should be NEGATIVE
    if median_val > 0:
        # Data has positive z_im for capacitive -> flip
        return -z_im, True, ""
    else:
        # Already in capacitive-negative convention
        return z_im.copy(), False, ""


def _drop_hf_inductive_points(
    z_re: np.ndarray, 
    z_im_cap: np.ndarray, 
    freq: np.ndarray,
    min_keep_fraction: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, str]:
    """
    Drop consecutive HF inductive points (z_im_cap > 0) from highest frequency.
    
    In SS/GPE/SS cells, >100 kHz can show lead inductance artifacts.
    
    Args:
        z_re: Real impedance array
        z_im_cap: Imaginary impedance (capacitive = negative)
        freq: Frequency array
        min_keep_fraction: Minimum fraction of points to keep (safety)
    
    Returns:
        (z_re_clean, z_im_clean, freq_clean, n_dropped, warning)
    """
    # Sort by freq descending (highest first)
    sort_idx = np.argsort(freq)[::-1]
    z_re_s = z_re[sort_idx]
    z_im_s = z_im_cap[sort_idx]
    freq_s = freq[sort_idx]
    
    # Find first capacitive point (z_im <= 0)
    n_total = len(z_re_s)
    min_keep = max(3, int(n_total * min_keep_fraction))
    
    n_drop = 0
    for i in range(n_total):
        if z_im_s[i] > 0:  # Inductive
            n_drop += 1
        else:
            break
        # Safety: don't drop too many
        if n_total - n_drop < min_keep:
            break
    
    warning = ""
    if n_drop > 0:
        if n_drop >= n_total - min_keep:
            warning = f"Most points appear inductive; kept {n_total - n_drop} of {n_total}"
        # Drop the inductive points (they are at the start after sorting)
        z_re_clean = z_re_s[n_drop:]
        z_im_clean = z_im_s[n_drop:]
        freq_clean = freq_s[n_drop:]
    else:
        z_re_clean = z_re_s
        z_im_clean = z_im_s
        freq_clean = freq_s
    
    return z_re_clean, z_im_clean, freq_clean, n_drop, warning


def _select_hf_band(
    z_re: np.ndarray, 
    z_im: np.ndarray, 
    freq: np.ndarray,
    decades: float = 1.0,
    min_points: int = 6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select high-frequency band by frequency (log-space), not point count.
    
    Uses top `decades` decades: freq >= fmax / 10^decades.
    Falls back to top `min_points` if selection is too sparse.
    
    Args:
        z_re, z_im, freq: Full arrays
        decades: Number of decades from fmax (default 1.0 = top decade)
        min_points: Minimum points to return
    
    Returns:
        (z_re_hf, z_im_hf, freq_hf) - sorted by freq descending
    """
    # Sort by freq descending
    sort_idx = np.argsort(freq)[::-1]
    z_re_s = z_re[sort_idx]
    z_im_s = z_im[sort_idx]
    freq_s = freq[sort_idx]
    
    fmax = freq_s[0]
    f_threshold = fmax / (10 ** decades)
    
    # Select points in HF band
    hf_mask = freq_s >= f_threshold
    n_hf = np.sum(hf_mask)
    
    if n_hf >= min_points:
        return z_re_s[hf_mask], z_im_s[hf_mask], freq_s[hf_mask]
    else:
        # Fallback: take top min_points
        n_take = min(min_points, len(freq_s))
        return z_re_s[:n_take], z_im_s[:n_take], freq_s[:n_take]


# =============================================================================
# Core Functions
# =============================================================================

def compute_conductivity(rb_ohm: float, thickness_cm: float, area_cm2: float) -> Dict[str, Any]:
    """
    Calculate ionic conductivity from bulk resistance.
    
    σ = L / (Rb × S)
    
    Args:
        rb_ohm: Bulk resistance in Ω (from Nyquist intercept)
        thickness_cm: Sample thickness in cm
        area_cm2: Electrode area in cm²
    
    Returns:
        dict with:
            - sigma_s_cm: Conductivity in S/cm
            - qc_checks: List of QC warnings
            - params: Input parameters for traceability
    """
    if rb_ohm <= 0:
        return {
            "sigma_s_cm": None,
            "qc_checks": ["Rb must be positive"],
            "params": {"rb_ohm": rb_ohm, "thickness_cm": thickness_cm, "area_cm2": area_cm2}
        }
    
    if thickness_cm <= 0 or area_cm2 <= 0:
        return {
            "sigma_s_cm": None,
            "qc_checks": ["Thickness and area must be positive"],
            "params": {"rb_ohm": rb_ohm, "thickness_cm": thickness_cm, "area_cm2": area_cm2}
        }
    
    sigma = thickness_cm / (rb_ohm * area_cm2)
    
    qc_checks = []
    if sigma < 1e-8:
        qc_checks.append(f"σ = {sigma:.2e} S/cm is unusually low (< 1e-8)")
    elif sigma > 1:
        qc_checks.append(f"σ = {sigma:.2e} S/cm is unusually high (> 1)")
    elif sigma < 1e-6:
        qc_checks.append(f"σ = {sigma:.2e} S/cm is low for typical SPE (< 1e-6)")
    
    return {
        "sigma_s_cm": sigma,
        "qc_checks": qc_checks,
        "params": {
            "rb_ohm": rb_ohm,
            "thickness_cm": thickness_cm,
            "area_cm2": area_cm2
        }
    }


def estimate_rb_intercept_linear(
    z_re: np.ndarray, 
    z_im: np.ndarray,
    freq: np.ndarray,
    hf_fraction: float = 0.3,  # Kept for API compat, but not used internally
    hf_decades: float = 1.0,
    near_axis_fraction: float = 0.3
) -> Optional[float]:
    """
    Estimate Rb from high-frequency Nyquist intercept using linear extrapolation.
    
    Uses frequency-based HF selection (top 1 decade) and fits only near-axis points.
    Excludes HF inductive artifacts automatically.
    
    Args:
        z_re: Real impedance array (Ω)
        z_im: Imaginary impedance array (Ω, any sign convention)
        freq: Frequency array (Hz)
        hf_fraction: DEPRECATED - kept for API compatibility, use hf_decades
        hf_decades: Number of decades from fmax for HF band (default 1.0)
        near_axis_fraction: Fraction of arc height for near-axis selection
    
    Returns:
        Estimated Rb in Ω, or None if estimation fails
    """
    if len(z_re) < 3:
        return None
    
    # Standardize sign to capacitive-negative
    z_im_cap, _, _ = _standardize_sign_capacitive(z_im, freq)
    
    # Drop HF inductive points
    z_re_c, z_im_c, freq_c, n_drop, _ = _drop_hf_inductive_points(z_re, z_im_cap, freq)
    
    if len(z_re_c) < 3:
        return None
    
    # Select HF band (frequency-based)
    z_re_hf, z_im_hf, freq_hf = _select_hf_band(z_re_c, z_im_c, freq_c, decades=hf_decades)
    
    if len(z_re_hf) < 2:
        return None
    
    # Select near-axis points: |z_im| < threshold
    arc_height = np.max(np.abs(z_im_hf)) - np.min(np.abs(z_im_hf))
    if arc_height < 1e-10:
        arc_height = np.max(np.abs(z_im_hf))
    
    threshold = np.min(np.abs(z_im_hf)) + arc_height * near_axis_fraction
    near_axis_mask = np.abs(z_im_hf) <= threshold
    
    z_re_fit = z_re_hf[near_axis_mask]
    z_im_fit = z_im_hf[near_axis_mask]
    
    if len(z_re_fit) < 2:
        # Fallback: use all HF points
        z_re_fit = z_re_hf
        z_im_fit = z_im_hf
    
    # Linear fit: Z_re = m * Z_im + Rb (intercept at Z_im = 0)
    if np.std(z_im_fit) < 1e-10:
        # Flat Z_im - just return min Z_re
        return float(np.min(z_re_fit))
    
    try:
        coeffs = np.polyfit(z_im_fit, z_re_fit, 1)
        rb_estimate = coeffs[1]  # Intercept
        
        # Sanity check: Rb should be positive and reasonable
        if rb_estimate <= 0:
            rb_estimate = np.min(z_re_hf)
        elif rb_estimate > np.max(z_re_hf):
            rb_estimate = np.min(z_re_hf)
        
        return float(rb_estimate)
    except Exception:
        # Fallback to direct intercept
        return find_hf_intercept_direct(z_re, z_im, freq)


def find_hf_intercept_direct(
    z_re: np.ndarray, 
    z_im: np.ndarray,
    freq: np.ndarray,
    z_im_threshold: float = 0.1,
    hf_decades: float = 1.0
) -> Optional[float]:
    """
    Find the high-frequency real axis intercept directly.
    
    Looks for where Z_im crosses zero at high frequencies. If a sign change
    exists, linearly interpolates Z_re at Z_im=0. Otherwise, returns median
    Z_re of points where |Z_im| is small.
    
    Args:
        z_re: Real impedance array (Ω)
        z_im: Imaginary impedance array (Ω, any sign convention)
        freq: Frequency array (Hz)
        z_im_threshold: Relative threshold for "near zero" (fraction of max |Z_im|)
        hf_decades: Number of decades from fmax for HF band selection
    
    Returns:
        Rb estimate in Ω, or None if estimation fails
    """
    if len(z_re) < 2:
        return None
    
    # Standardize sign to capacitive-negative
    z_im_cap, _, _ = _standardize_sign_capacitive(z_im, freq)
    
    # Drop HF inductive points
    z_re_c, z_im_c, freq_c, n_drop, warn = _drop_hf_inductive_points(z_re, z_im_cap, freq)
    
    if len(z_re_c) < 2:
        # Fallback: use original data
        z_re_c, z_im_c, freq_c = z_re, z_im_cap, freq
    
    # Select HF band
    z_re_hf, z_im_hf, freq_hf = _select_hf_band(z_re_c, z_im_c, freq_c, decades=hf_decades)
    
    if len(z_re_hf) < 1:
        return None
    
    # Check for sign change in HF band (zero crossing)
    has_positive = np.any(z_im_hf > 0)
    has_negative = np.any(z_im_hf < 0)
    
    if has_positive and has_negative:
        # Find the closest bracketing pair around zero
        # Sort by |z_im| to find points closest to zero
        sorted_indices = np.argsort(np.abs(z_im_hf))
        
        # Find one positive and one negative closest to zero
        pos_mask = z_im_hf > 0
        neg_mask = z_im_hf < 0
        
        pos_indices = np.where(pos_mask)[0]
        neg_indices = np.where(neg_mask)[0]
        
        if len(pos_indices) > 0 and len(neg_indices) > 0:
            # Find the pair with smallest |z_im| difference
            best_pos = pos_indices[np.argmin(z_im_hf[pos_indices])]
            best_neg = neg_indices[np.argmax(z_im_hf[neg_indices])]
            
            # Linear interpolate Z_re at Z_im = 0
            z_im1, z_re1 = z_im_hf[best_pos], z_re_hf[best_pos]
            z_im2, z_re2 = z_im_hf[best_neg], z_re_hf[best_neg]
            
            if abs(z_im1 - z_im2) > 1e-12:
                rb = z_re1 + (z_re2 - z_re1) * (0 - z_im1) / (z_im2 - z_im1)
                return float(rb)
    
    # No zero crossing: use threshold-based selection
    max_abs_zim = np.max(np.abs(z_im_hf))
    if max_abs_zim < 1e-10:
        # Essentially flat - return median Z_re
        return float(np.median(z_re_hf))
    
    threshold_val = z_im_threshold * max_abs_zim
    near_zero_mask = np.abs(z_im_hf) <= threshold_val
    
    if np.sum(near_zero_mask) > 0:
        return float(np.median(z_re_hf[near_zero_mask]))
    else:
        # Fallback: return Z_re at minimum |Z_im|
        min_idx = np.argmin(np.abs(z_im_hf))
        return float(z_re_hf[min_idx])


def prepare_nyquist_data(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Prepare data for Nyquist plot visualization.
    
    Handles sign conventions and sorting. Uses robust sign detection.
    
    Args:
        df: DataFrame with columns freq_hz, z_re_ohm, z_im_ohm
    
    Returns:
        dict with:
            - z_re: Real impedance (sorted by freq ascending)
            - z_im: Original imaginary impedance (sorted)
            - z_im_cap: Capacitive-negative convention (sorted)
            - z_im_plot: For Nyquist upper half (-z_im_cap, positive)
            - freq: Frequency array (sorted ascending)
            - sign_was_flipped: Whether auto-flip was applied
            - sign_warning: Any warning from sign detection
    """
    z_re = df["z_re_ohm"].values
    z_im = df["z_im_ohm"].values
    freq = df["freq_hz"].values
    
    # Sort by frequency (low to high for Nyquist plot)
    sort_idx = np.argsort(freq)
    z_re_sorted = z_re[sort_idx]
    z_im_sorted = z_im[sort_idx]
    freq_sorted = freq[sort_idx]
    
    # Standardize sign to capacitive-negative
    z_im_cap, was_flipped, sign_warning = _standardize_sign_capacitive(z_im_sorted, freq_sorted)
    
    # z_im_plot: for Nyquist display (capacitive in upper half = positive)
    z_im_plot = -z_im_cap
    
    return {
        "z_re": z_re_sorted,
        "z_im": z_im_sorted,  # Original (backward compat)
        "z_im_cap": z_im_cap,  # Capacitive-negative convention
        "z_im_plot": z_im_plot,  # For plotting (positive upper half)
        "freq": freq_sorted,
        "sign_was_flipped": was_flipped,
        "sign_warning": sign_warning
    }


# =============================================================================
# Equivalent Circuit Fitting (optional, requires SciPy)
# =============================================================================

def rc_impedance(freq: np.ndarray, r: float, c: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate impedance of parallel RC circuit.
    
    Z = R / (1 + j*ω*R*C)
    
    Returns (z_re, z_im) where z_im is NEGATIVE for capacitive.
    """
    omega = 2 * np.pi * freq
    denom = 1 + (omega * r * c) ** 2
    z_re = r / denom
    z_im = -omega * r**2 * c / denom  # Negative for capacitive
    return z_re, z_im


def randles_impedance(
    freq: np.ndarray, 
    rs: float, 
    rct: float, 
    cdl: float,
    sigma_w: float = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate impedance of Randles circuit: Rs + (Rct || Cdl) + Warburg.
    
    Simplified version without full Warburg implementation.
    """
    omega = 2 * np.pi * freq
    
    # Parallel Rct || Cdl
    z_rct_cdl_re, z_rct_cdl_im = rc_impedance(freq, rct, cdl)
    
    # Total impedance
    z_re = rs + z_rct_cdl_re
    z_im = z_rct_cdl_im
    
    # Add Warburg if specified
    if sigma_w > 0:
        z_w = sigma_w / np.sqrt(omega) * (1 - 1j)
        z_re = z_re + np.real(z_w)
        z_im = z_im + np.imag(z_w)
    
    return z_re, z_im


def _detect_diffusion_tail(z_re: np.ndarray, z_im: np.ndarray, freq: np.ndarray) -> np.ndarray:
    """
    Detect low-frequency diffusion tail points (Warburg-like 45° slope).
    
    Returns boolean mask where True = likely diffusion tail (exclude from RC fit).
    """
    if len(z_re) < 5:
        return np.zeros(len(z_re), dtype=bool)
    
    # Sort by freq ascending (LF first)
    sort_idx = np.argsort(freq)
    z_re_s = z_re[sort_idx]
    z_im_s = np.abs(z_im[sort_idx])  # Use absolute for slope calc
    
    # Calculate local slopes in Nyquist (dZ_im/dZ_re)
    is_diffusion = np.zeros(len(z_re), dtype=bool)
    
    # Check LF region (first 30% of points by freq)
    n_lf = max(3, int(len(z_re) * 0.3))
    
    for i in range(1, n_lf):
        dz_re = z_re_s[i] - z_re_s[i-1]
        dz_im = z_im_s[i] - z_im_s[i-1]
        
        if abs(dz_re) > 1e-10:
            slope = dz_im / dz_re
            # Warburg has slope ~1 (45° line)
            if 0.5 < slope < 2.0:
                is_diffusion[sort_idx[i]] = True
                is_diffusion[sort_idx[i-1]] = True
    
    return is_diffusion


def fit_simple_rc(
    freq: np.ndarray, 
    z_re: np.ndarray, 
    z_im: np.ndarray,
    rs_init: Optional[float] = None,
    hf_decades: float = 1.5,
    exclude_diffusion: bool = True
) -> Dict[str, Any]:
    """
    Fit a simple Rs + (R || C) circuit to HIGH-FREQUENCY EIS data only.
    
    WARNING: This fit is only valid for the HF semicircle in SS/GPE/SS cells.
    It does NOT model Warburg diffusion or blocking electrode effects.
    For full Randles+Warburg analysis, use specialized software (ZView, Relaxis).
    
    Args:
        freq: Frequency array (Hz)
        z_re: Real impedance array (Ω)
        z_im: Imaginary impedance array (Ω, any sign convention)
        rs_init: Initial guess for Rs (auto-detected if None)
        hf_decades: Number of decades from fmax for HF band (default 1.5)
        exclude_diffusion: If True, auto-detect and exclude LF diffusion tail
    
    Returns:
        dict with Rs (bulk resistance), R, C, r_squared, and fit metadata
    """
    if not SCIPY_AVAILABLE:
        return {
            "success": False,
            "error": "SciPy not installed. Install with: pip install scipy"
        }
    
    if len(freq) < 6:
        return {
            "success": False,
            "error": "Insufficient data points (need >= 6)"
        }
    
    # Standardize sign to capacitive-negative
    z_im_cap, was_flipped, _ = _standardize_sign_capacitive(z_im, freq)
    
    # Exclude diffusion tail if requested
    if exclude_diffusion:
        diffusion_mask = _detect_diffusion_tail(z_re, z_im_cap, freq)
        keep_mask = ~diffusion_mask
    else:
        keep_mask = np.ones(len(freq), dtype=bool)
    
    # Select HF band
    z_re_hf, z_im_hf, freq_hf = _select_hf_band(
        z_re[keep_mask], z_im_cap[keep_mask], freq[keep_mask], 
        decades=hf_decades, min_points=6
    )
    
    if len(freq_hf) < 4:
        return {
            "success": False,
            "error": f"Insufficient HF data points after filtering ({len(freq_hf)} < 4)"
        }
    
    # Initial guess for Rs from high-frequency intercept
    if rs_init is None:
        rs_init = find_hf_intercept_direct(z_re_hf, z_im_hf, freq_hf) or np.min(z_re_hf)
    
    def model(f, rs, r, c):
        z_re_calc, z_im_calc = rc_impedance(f, r, c)
        return np.concatenate([rs + z_re_calc, z_im_calc])
    
    # Initial guesses
    r_init = max(1.0, np.max(z_re_hf) - rs_init)
    c_init = 1e-6  # 1 µF typical
    
    # Widen C bounds for high-area gels (up to 10 mF)
    c_max = 1e-2
    
    try:
        z_data = np.concatenate([z_re_hf, z_im_hf])
        popt, pcov = curve_fit(
            model, freq_hf, z_data,
            p0=[rs_init, r_init, c_init],
            bounds=([0, 0, 1e-12], [np.inf, np.inf, c_max]),
            maxfev=5000
        )
        
        rs, r, c = popt
        
        # Calculate fit quality with safeguard
        z_fit = model(freq_hf, rs, r, c)
        ss_res = np.sum((z_data - z_fit) ** 2)
        ss_tot = np.sum((z_data - np.mean(z_data)) ** 2)
        
        if ss_tot < 1e-20:
            r_squared = np.nan
        else:
            r_squared = 1 - ss_res / ss_tot
        
        # Warn if C near bounds
        warnings = []
        if c > c_max * 0.9:
            warnings.append(f"C near upper bound ({c:.2e} F ≈ {c_max:.0e} F limit)")
        if c < 1e-11:
            warnings.append(f"C near lower bound ({c:.2e} F)")
        
        return {
            "Rs_ohm": float(rs),
            "R_ohm": float(r),
            "C_F": float(c),
            "r_squared": float(r_squared) if not np.isnan(r_squared) else None,
            "success": True,
            "n_points_fitted": len(freq_hf),
            "hf_only": True,
            "warnings": warnings,
            "note": "HF semicircle fit only; not valid for diffusion/Warburg analysis"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
