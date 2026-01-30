"""
Transference number (tLi+) calculation using Bruce-Vincent method.
"""
import numpy as np
from typing import Dict, Any, List


def compute_transference(I0: float, Iss: float, R0: float, Rss: float,
                         delta_V: float) -> Dict[str, Any]:
    """
    Calculate Li+ transference number using Bruce-Vincent (Evans) method.
    
    tLi+ = Iss × (ΔV - I0×R0) / [I0 × (ΔV - Iss×Rss)]
    
    Args:
        I0: Initial current (A) - from chronoamperometry start
        Iss: Steady-state current (A) - from chronoamperometry end
        R0: Initial interfacial resistance (Ω) - from EIS before polarization
        Rss: Steady-state interfacial resistance (Ω) - from EIS after polarization
        delta_V: Applied potential step (V)
    
    Returns:
        dict with t_li_plus, warnings, params
    """
    warnings = []
    params = {
        "I0_A": I0,
        "Iss_A": Iss,
        "R0_ohm": R0,
        "Rss_ohm": Rss,
        "delta_V_V": delta_V
    }
    
    # Input validation
    if I0 <= 0:
        warnings.append("I0 should be positive (current at t=0)")
    if Iss <= 0:
        warnings.append("Iss should be positive (steady-state current)")
    if R0 <= 0:
        warnings.append("R0 should be positive (initial resistance)")
    if Rss <= 0:
        warnings.append("Rss should be positive (steady-state resistance)")
    if delta_V <= 0:
        warnings.append("ΔV should be positive (applied potential)")
    
    # Calculate numerator and denominator
    numerator = Iss * (delta_V - I0 * R0)
    denominator = I0 * (delta_V - Iss * Rss)
    
    # Check for division issues
    if abs(denominator) < 1e-15:
        warnings.append("Denominator near zero - check values")
        return {
            "t_li_plus": None,
            "warnings": warnings,
            "params": params,
            "success": False
        }
    
    t = numerator / denominator
    
    # QC checks on result
    if t < 0:
        warnings.append(f"tLi+ = {t:.4f} < 0: Likely causes: wrong R assignment, sign errors, or non-blocking electrodes")
    elif t > 1:
        warnings.append(f"tLi+ = {t:.4f} > 1: Likely causes: measurement not at steady state, parasitic reactions, or incorrect Rss")
    elif t < 0.1:
        warnings.append(f"tLi+ = {t:.4f} is unusually low for SPE (expected 0.2-0.6)")
    elif t > 0.8:
        warnings.append(f"tLi+ = {t:.4f} is unusually high for SPE (expected 0.2-0.6)")
    
    # Check if voltage drops are reasonable
    v_drop_initial = I0 * R0
    v_drop_ss = Iss * Rss
    
    if v_drop_initial > delta_V:
        warnings.append(f"I0×R0 = {v_drop_initial:.4f} V > ΔV: Initial resistive drop exceeds applied voltage")
    if v_drop_ss > delta_V:
        warnings.append(f"Iss×Rss = {v_drop_ss:.4f} V > ΔV: Steady-state resistive drop exceeds applied voltage")
    
    return {
        "t_li_plus": float(t),
        "warnings": warnings,
        "params": params,
        "success": True,
        "numerator": float(numerator),
        "denominator": float(denominator)
    }


def extract_currents_from_chrono(t_s: np.ndarray, i_a: np.ndarray,
                                  t0_fraction: float = 0.01,
                                  ss_fraction: float = 0.1) -> Dict[str, float]:
    """
    Extract I0 and Iss from chronoamperometry data.
    
    Args:
        t_s: Time array in seconds
        i_a: Current array in A
        t0_fraction: Fraction of total time to average for I0 (early time)
        ss_fraction: Fraction of total time to average for Iss (late time)
    
    Returns:
        dict with I0, Iss, t0_range, tss_range
    """
    t_total = t_s[-1] - t_s[0]
    
    # I0: average of early-time current
    t0_end = t_s[0] + t_total * t0_fraction
    mask_0 = t_s <= t0_end
    I0 = np.abs(np.mean(i_a[mask_0])) if mask_0.any() else np.abs(i_a[0])
    
    # Iss: average of late-time current
    tss_start = t_s[-1] - t_total * ss_fraction
    mask_ss = t_s >= tss_start
    Iss = np.abs(np.mean(i_a[mask_ss])) if mask_ss.any() else np.abs(i_a[-1])
    
    return {
        "I0_A": float(I0),
        "Iss_A": float(Iss),
        "t0_range_s": [float(t_s[0]), float(t0_end)],
        "tss_range_s": [float(tss_start), float(t_s[-1])]
    }
