"""
EIS (Electrochemical Impedance Spectroscopy) analysis module.

Provides Rb extraction, conductivity calculation, and QC checks.
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


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


def estimate_rb_intercept_linear(z_re: np.ndarray, z_im: np.ndarray,
                                  freq: np.ndarray,
                                  hf_fraction: float = 0.3) -> Optional[float]:
    """
    Estimate Rb from high-frequency Nyquist intercept using linear extrapolation.
    
    Uses the high-frequency portion of the semicircle and extrapolates to Z_im = 0.
    
    Args:
        z_re: Real impedance array (Ω)
        z_im: Imaginary impedance array (Ω, positive = capacitive)
        freq: Frequency array (Hz)
        hf_fraction: Fraction of data points to use from high-frequency end
    
    Returns:
        Estimated Rb in Ω, or None if estimation fails
    """
    # Sort by frequency (highest first)
    sort_idx = np.argsort(freq)[::-1]
    z_re_sorted = z_re[sort_idx]
    z_im_sorted = z_im[sort_idx]
    
    # Take high-frequency portion
    n_hf = max(3, int(len(z_re) * hf_fraction))
    z_re_hf = z_re_sorted[:n_hf]
    z_im_hf = z_im_sorted[:n_hf]
    
    # Filter for capacitive behavior (positive Z_im for typical convention)
    # Some instruments use negative Z_im for capacitive - handle both
    if np.mean(z_im_hf) < 0:
        z_im_hf = -z_im_hf  # Flip sign
    
    # Linear fit: Z_re = m * Z_im + Rb (intercept at Z_im = 0)
    if len(z_im_hf) < 2 or np.std(z_im_hf) < 1e-10:
        return None
    
    try:
        coeffs = np.polyfit(z_im_hf, z_re_hf, 1)
        rb_estimate = coeffs[1]  # Intercept
        
        # Sanity check: Rb should be positive and less than min(Z_re)
        if rb_estimate <= 0:
            rb_estimate = np.min(z_re_hf)
        elif rb_estimate > np.min(z_re):
            rb_estimate = np.min(z_re)
        
        return rb_estimate
    except Exception:
        return None


def find_hf_intercept_direct(z_re: np.ndarray, z_im: np.ndarray,
                              freq: np.ndarray,
                              z_im_threshold: float = 0.05) -> Optional[float]:
    """
    Find the high-frequency real axis intercept directly.
    
    Looks for where |Z_im| is minimal at high frequencies.
    
    Args:
        z_re: Real impedance array (Ω)
        z_im: Imaginary impedance array (Ω)
        freq: Frequency array (Hz)
        z_im_threshold: Relative threshold for "near zero" Z_im
    
    Returns:
        Rb estimate in Ω
    """
    # Sort by frequency descending
    sort_idx = np.argsort(freq)[::-1]
    z_re_sorted = z_re[sort_idx]
    z_im_sorted = np.abs(z_im[sort_idx])
    
    # Find minimum |Z_im| in high-frequency region
    n_check = max(5, len(z_re) // 4)
    min_idx = np.argmin(z_im_sorted[:n_check])
    
    return z_re_sorted[min_idx]


def prepare_nyquist_data(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Prepare data for Nyquist plot visualization.
    
    Handles sign conventions and sorting.
    
    Args:
        df: DataFrame with columns freq_hz, z_re_ohm, z_im_ohm
    
    Returns:
        dict with z_re, z_im (sign-corrected), freq arrays
    """
    z_re = df["z_re_ohm"].values
    z_im = df["z_im_ohm"].values
    freq = df["freq_hz"].values
    
    # Sort by frequency (low to high for Nyquist plot)
    sort_idx = np.argsort(freq)
    
    # Convention: plot -Z_im vs Z_re for capacitive semicircle in upper half
    # If Z_im is already negative (some instruments), flip it
    z_im_plot = z_im.copy()
    if np.mean(z_im) > 0:
        z_im_plot = -z_im  # Standard convention: -Z_im on y-axis
    
    return {
        "z_re": z_re[sort_idx],
        "z_im": z_im[sort_idx],
        "z_im_plot": -z_im_plot[sort_idx],  # For plotting: positive in upper half
        "freq": freq[sort_idx]
    }


# =============================================================================
# Equivalent Circuit Fitting (optional advanced feature)
# =============================================================================

def rc_impedance(freq: np.ndarray, r: float, c: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate impedance of parallel RC circuit.
    
    Z = R / (1 + j*ω*R*C)
    """
    omega = 2 * np.pi * freq
    denom = 1 + (omega * r * c) ** 2
    z_re = r / denom
    z_im = -omega * r**2 * c / denom  # Negative for capacitive
    return z_re, z_im


def randles_impedance(freq: np.ndarray, rs: float, rct: float, cdl: float,
                       sigma_w: float = 0) -> Tuple[np.ndarray, np.ndarray]:
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


def fit_simple_rc(freq: np.ndarray, z_re: np.ndarray, z_im: np.ndarray,
                   rs_init: Optional[float] = None) -> Dict[str, Any]:
    """
    Fit a simple Rs + (R || C) circuit to EIS data.
    
    Returns:
        dict with Rs (bulk resistance), R, C, and fit quality
    """
    # Initial guess for Rs from high-frequency intercept
    if rs_init is None:
        rs_init = find_hf_intercept_direct(z_re, z_im, freq) or np.min(z_re)
    
    def model(f, rs, r, c):
        z_re_calc, z_im_calc = rc_impedance(f, r, c)
        return np.concatenate([rs + z_re_calc, z_im_calc])
    
    # Initial guesses
    r_init = np.max(z_re) - rs_init
    c_init = 1e-6  # 1 µF typical
    
    try:
        z_data = np.concatenate([z_re, z_im])
        popt, pcov = curve_fit(
            model, freq, z_data,
            p0=[rs_init, r_init, c_init],
            bounds=([0, 0, 1e-12], [np.inf, np.inf, 1e-3]),
            maxfev=5000
        )
        
        rs, r, c = popt
        
        # Calculate fit quality
        z_fit = model(freq, rs, r, c)
        ss_res = np.sum((z_data - z_fit) ** 2)
        ss_tot = np.sum((z_data - np.mean(z_data)) ** 2)
        r_squared = 1 - ss_res / ss_tot
        
        return {
            "Rs_ohm": rs,
            "R_ohm": r,
            "C_F": c,
            "r_squared": r_squared,
            "success": True
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
