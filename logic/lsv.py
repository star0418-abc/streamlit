"""
LSV (Linear Sweep Voltammetry) analysis for electrochemical stability window.
"""
import numpy as np
from typing import Dict, Any, Optional
from scipy.signal import savgol_filter


def find_onset_potential(e_v: np.ndarray, j_ma_cm2: np.ndarray,
                         threshold_ma_cm2: float = 0.1,
                         direction: str = "oxidation",
                         smooth_window: int = 5,
                         baseline_correct: bool = True) -> Dict[str, Any]:
    """
    Find onset potential where current density exceeds threshold.
    
    Args:
        e_v: Potential array in V (vs reference)
        j_ma_cm2: Current density array in mA/cm²
        threshold_ma_cm2: Current density threshold (default 0.1 mA/cm²)
        direction: "oxidation" (positive j) or "reduction" (negative j)
        smooth_window: Savitzky-Golay window for noise reduction (must be odd)
        baseline_correct: Whether to apply baseline correction
    
    Returns:
        dict with onset_v, baseline_corrected, j_smoothed
    """
    warnings = []
    
    # Input validation
    if len(e_v) < 5:
        return {"onset_v": None, "error": "Insufficient data points"}
    
    # Ensure odd window
    if smooth_window % 2 == 0:
        smooth_window += 1
    
    # Smooth current density
    if len(j_ma_cm2) > smooth_window:
        try:
            j_smooth = savgol_filter(j_ma_cm2, smooth_window, 2)
        except Exception:
            j_smooth = j_ma_cm2.copy()
    else:
        j_smooth = j_ma_cm2.copy()
    
    # Baseline correction
    j_corrected = j_smooth.copy()
    if baseline_correct:
        # Use low-current region for baseline
        low_j_mask = np.abs(j_smooth) < threshold_ma_cm2 / 2
        if low_j_mask.sum() > 5:
            try:
                baseline_coeffs = np.polyfit(e_v[low_j_mask], j_smooth[low_j_mask], 1)
                baseline = np.polyval(baseline_coeffs, e_v)
                j_corrected = j_smooth - baseline
            except Exception:
                # Fallback: simple mean subtraction
                j_corrected = j_smooth - np.mean(j_smooth[low_j_mask])
        else:
            warnings.append("Insufficient low-current data for baseline correction")
    
    # Find threshold crossing
    if direction == "oxidation":
        mask = j_corrected > threshold_ma_cm2
    elif direction == "reduction":
        mask = j_corrected < -threshold_ma_cm2
    else:
        return {"onset_v": None, "error": f"Unknown direction: {direction}"}
    
    onset_v = None
    onset_idx = None
    
    if mask.any():
        onset_idx = np.argmax(mask)
        onset_v = float(e_v[onset_idx])
        
        # Interpolate for better precision
        if onset_idx > 0:
            j1 = j_corrected[onset_idx - 1]
            j2 = j_corrected[onset_idx]
            e1 = e_v[onset_idx - 1]
            e2 = e_v[onset_idx]
            
            target_j = threshold_ma_cm2 if direction == "oxidation" else -threshold_ma_cm2
            
            if abs(j2 - j1) > 1e-10:
                onset_v = float(e1 + (target_j - j1) * (e2 - e1) / (j2 - j1))
    else:
        warnings.append(f"No {direction} onset found at threshold {threshold_ma_cm2} mA/cm²")
    
    return {
        "onset_v": onset_v,
        "onset_idx": int(onset_idx) if onset_idx is not None else None,
        "direction": direction,
        "threshold_ma_cm2": threshold_ma_cm2,
        "baseline_corrected": baseline_correct,
        "warnings": warnings,
        "j_smoothed": j_smooth,
        "j_corrected": j_corrected
    }


def find_stability_window(e_v: np.ndarray, j_ma_cm2: np.ndarray,
                          threshold_ma_cm2: float = 0.1,
                          smooth_window: int = 5) -> Dict[str, Any]:
    """
    Find both oxidation and reduction onset potentials.
    
    Returns the electrochemical stability window.
    
    Args:
        e_v: Potential array in V
        j_ma_cm2: Current density array in mA/cm²
        threshold_ma_cm2: Current density threshold
        smooth_window: Smoothing window size
    
    Returns:
        dict with oxidation_onset_v, reduction_onset_v, window_v
    """
    # Find oxidation onset (positive direction)
    ox_result = find_onset_potential(
        e_v, j_ma_cm2, threshold_ma_cm2, 
        direction="oxidation", smooth_window=smooth_window
    )
    
    # Find reduction onset (negative direction)  
    red_result = find_onset_potential(
        e_v, j_ma_cm2, threshold_ma_cm2,
        direction="reduction", smooth_window=smooth_window
    )
    
    ox_onset = ox_result.get("onset_v")
    red_onset = red_result.get("onset_v")
    
    window = None
    if ox_onset is not None and red_onset is not None:
        window = abs(ox_onset - red_onset)
    
    warnings = ox_result.get("warnings", []) + red_result.get("warnings", [])
    
    return {
        "oxidation_onset_v": ox_onset,
        "reduction_onset_v": red_onset,
        "window_v": window,
        "threshold_ma_cm2": threshold_ma_cm2,
        "warnings": warnings
    }
