"""
LSV (Linear Sweep Voltammetry) analysis for electrochemical stability window.

SCIPY OPTIONAL:
- Savitzky-Golay smoothing requires SciPy; degrades to numpy moving-average if missing.

ONSET METHODS:
- "threshold": Fixed |j| > threshold (requires k consecutive points for robustness)
- "tangent": Baseline–tangent intersection (recommended for GPE with current creep)

CV HANDLING:
- Non-monotonic potential is auto-detected
- Longest monotonic segment extracted with warning
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Literal

# =============================================================================
# Optional SciPy import (safe for Streamlit Cloud without scipy)
# =============================================================================
SCIPY_AVAILABLE = False
savgol_filter = None

try:
    from scipy.signal import savgol_filter as _savgol
    savgol_filter = _savgol
    SCIPY_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# Helper Functions
# =============================================================================

def _optional_savgol(
    j: np.ndarray, 
    window: int, 
    polyorder: int = 2
) -> Tuple[np.ndarray, List[str]]:
    """
    Apply Savitzky-Golay smoothing if SciPy available, else numpy moving-average.
    
    Returns:
        (j_smooth, warnings)
    """
    warnings = []
    
    # Ensure odd window
    if window % 2 == 0:
        window += 1
    
    if len(j) <= window:
        return j.copy(), ["Insufficient points for smoothing"]
    
    if SCIPY_AVAILABLE and savgol_filter is not None:
        try:
            j_smooth = savgol_filter(j, window, polyorder)
            return j_smooth, warnings
        except Exception as e:
            warnings.append(f"Savgol failed: {e}; using moving average")
    else:
        warnings.append("SciPy not installed; using numpy moving-average for smoothing")
    
    # Fallback: simple moving average
    kernel = np.ones(window) / window
    # Pad to avoid edge shrinkage
    pad_left = window // 2
    pad_right = window - pad_left - 1
    j_padded = np.concatenate([
        np.full(pad_left, j[0]),
        j,
        np.full(pad_right, j[-1])
    ])
    j_smooth = np.convolve(j_padded, kernel, mode='valid')
    
    return j_smooth, warnings


def _select_monotonic_segment(
    e_v: np.ndarray, 
    j: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Detect CV (non-monotonic potential) and extract longest monotonic segment.
    
    Returns:
        (e_seg, j_seg, idx_map, warnings)
        - idx_map: indices in original array corresponding to segment
    """
    warnings = []
    
    if len(e_v) < 3:
        return e_v.copy(), j.copy(), np.arange(len(e_v)), warnings
    
    # Check for monotonicity - handle de=0 at direction reversals
    de = np.diff(e_v)
    
    # Get signs, treating zero as continuation of previous sign
    signs = np.sign(de)
    # Forward-fill zeros with previous nonzero sign
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i-1]
    
    # Count actual direction reversals
    sign_changes = np.sum(signs[:-1] * signs[1:] < 0)
    
    # Also check if we have both positive and negative nonzero derivatives
    nonzero_de = de[np.abs(de) > 1e-12]
    if len(nonzero_de) > 0:
        has_both_directions = np.any(nonzero_de > 0) and np.any(nonzero_de < 0)
        if has_both_directions and sign_changes == 0:
            # Edge case: sign_changes missed due to zero padding
            sign_changes = 1
    
    if sign_changes == 0:
        # Already monotonic
        return e_v.copy(), j.copy(), np.arange(len(e_v)), warnings
    
    warnings.append(f"Non-monotonic potential detected ({sign_changes} sign changes); "
                   "likely CV input. Extracting longest monotonic segment.")
    
    # Find all monotonic segments
    segments = []
    start_idx = 0
    current_sign = np.sign(de[0]) if de[0] != 0 else 1
    
    for i in range(1, len(de)):
        if de[i] == 0:
            continue
        if np.sign(de[i]) != current_sign:
            # End of segment
            segments.append((start_idx, i + 1))  # +1 because de is len-1
            start_idx = i
            current_sign = np.sign(de[i])
    
    # Last segment
    segments.append((start_idx, len(e_v)))
    
    # Find longest segment
    longest = max(segments, key=lambda x: x[1] - x[0])
    start, end = longest
    
    idx_map = np.arange(start, end)
    e_seg = e_v[start:end]
    j_seg = j[start:end]
    
    warnings.append(f"Using segment [{start}:{end}] ({len(e_seg)} points) "
                   f"from {len(e_v)} total points")
    
    return e_seg, j_seg, idx_map, warnings


def _robust_baseline(
    e_v: np.ndarray, 
    j: np.ndarray,
    mode: str = "auto"
) -> Tuple[callable, Dict[str, Any], List[str]]:
    """
    Compute robust baseline from early portion of sweep.
    
    Uses median-based constant baseline by default. Linear only if clear trend.
    
    Args:
        mode: "auto", "constant", or "linear"
    
    Returns:
        (baseline_fn, params, warnings)
        - baseline_fn: function(e) -> baseline value(s)
        - params: dict with 'type', 'value' or 'slope'/'intercept'
    """
    warnings = []
    
    # Use first portion of sweep for baseline
    n_baseline = max(20, int(len(e_v) * 0.1))
    n_baseline = min(n_baseline, len(e_v) // 2)  # Don't use more than half
    
    j_baseline = j[:n_baseline]
    e_baseline = e_v[:n_baseline]
    
    # Constant baseline: median
    median_val = float(np.median(j_baseline))
    
    if mode == "constant":
        def baseline_fn(e):
            return np.full_like(e, median_val, dtype=float) if hasattr(e, '__len__') else median_val
        return baseline_fn, {"type": "constant", "value": median_val}, warnings
    
    # Check if linear fit is warranted (mode="auto" or "linear")
    if len(e_baseline) >= 5 and mode in ("auto", "linear"):
        try:
            coeffs = np.polyfit(e_baseline, j_baseline, 1)
            slope, intercept = coeffs[0], coeffs[1]
            
            # Calculate R² for linear fit
            j_fit = np.polyval(coeffs, e_baseline)
            ss_res = np.sum((j_baseline - j_fit) ** 2)
            ss_tot = np.sum((j_baseline - np.mean(j_baseline)) ** 2)
            
            r_squared = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0
            
            # Only use linear if R² is good and slope is meaningful
            use_linear = False
            if mode == "linear":
                use_linear = True
            elif r_squared > 0.7 and abs(slope) > 1e-6:
                use_linear = True
                warnings.append(f"Using linear baseline (R^2={r_squared:.3f}, slope={slope:.4f})")
            
            if use_linear:
                # Safety check: ensure linear baseline doesn't create large negative artifacts
                j_corrected_test = j - np.polyval(coeffs, e_v)
                if np.min(j_corrected_test) < -abs(np.max(j)) * 0.5:
                    warnings.append("Linear baseline would create large negative artifacts; "
                                   "falling back to constant baseline")
                else:
                    def baseline_fn(e):
                        return slope * e + intercept
                    return baseline_fn, {
                        "type": "linear", 
                        "slope": float(slope), 
                        "intercept": float(intercept),
                        "r_squared": float(r_squared)
                    }, warnings
        except Exception as e:
            warnings.append(f"Linear baseline fit failed: {e}")
    
    # Default: constant baseline
    def baseline_fn(e):
        return np.full_like(e, median_val, dtype=float) if hasattr(e, '__len__') else median_val
    
    return baseline_fn, {"type": "constant", "value": median_val}, warnings


def _threshold_onset(
    e_v: np.ndarray, 
    j_corrected: np.ndarray,
    threshold: float,
    direction: str,
    min_consecutive: int = 3,
    min_e_span: float = 0.0
) -> Tuple[Optional[float], Optional[int], List[str]]:
    """
    Find onset using threshold with consecutive-point requirement.
    
    Args:
        min_consecutive: Minimum consecutive points above threshold
        min_e_span: Minimum potential span (V) that consecutive points must cover.
                    Helps reject single-point spikes spread by smoothing.
    
    Returns:
        (onset_v, onset_idx, warnings)
    """
    warnings = []
    
    # Create mask based on direction
    if direction == "oxidation":
        mask = j_corrected > threshold
        target_j = threshold
    else:  # reduction
        mask = j_corrected < -threshold
        target_j = -threshold
    
    if not mask.any():
        warnings.append(f"No {direction} onset found at threshold {threshold} mA/cm²")
        return None, None, warnings
    
    # Find first occurrence of min_consecutive consecutive True values
    # Also require minimum potential span if specified
    onset_idx = None
    run_start = None
    
    for i, val in enumerate(mask):
        if val:
            if run_start is None:
                run_start = i
            consecutive_count = i - run_start + 1
            
            if consecutive_count >= min_consecutive:
                # Check potential span
                e_span = abs(e_v[i] - e_v[run_start])
                if e_span >= min_e_span:
                    onset_idx = run_start
                    break
        else:
            run_start = None
    
    if onset_idx is None:
        warnings.append(f"Found threshold crossings but none with {min_consecutive} "
                       "consecutive points (likely noise spikes)")
        return None, None, warnings
    
    onset_v = float(e_v[onset_idx])
    
    # Interpolate for better precision
    if onset_idx > 0:
        j1 = j_corrected[onset_idx - 1]
        j2 = j_corrected[onset_idx]
        e1 = e_v[onset_idx - 1]
        e2 = e_v[onset_idx]
        
        if abs(j2 - j1) > 1e-10:
            onset_v = float(e1 + (target_j - j1) * (e2 - e1) / (j2 - j1))
    
    return onset_v, onset_idx, warnings


def _tangent_onset(
    e_v: np.ndarray, 
    j_corrected: np.ndarray,
    baseline_fn: callable,
    baseline_params: Dict[str, Any],
    direction: str,
    smooth_window: int = 5
) -> Tuple[Optional[float], Dict[str, Any], List[str]]:
    """
    Find onset via baseline–tangent intersection method.
    
    1. Transform current so rise is always positive (j_eff)
    2. Compute derivative dj_eff/dE on smoothed data
    3. Find max derivative index (excluding edges)
    4. Fit tangent line in window around max derivative
    5. Find intersection of tangent with baseline
    
    Returns:
        (onset_v, fit_params, warnings)
    """
    warnings = []
    fit_params = {}
    
    # Transform for consistent logic
    if direction == "oxidation":
        j_eff = j_corrected.copy()
    else:
        j_eff = -j_corrected.copy()
    
    # Smooth for derivative calculation
    j_smooth, smooth_warns = _optional_savgol(j_eff, smooth_window, polyorder=2)
    warnings.extend(smooth_warns)
    
    if len(e_v) < 10:
        warnings.append("Insufficient points for tangent method")
        return None, fit_params, warnings
    
    # Compute derivative
    de = np.diff(e_v)
    dj = np.diff(j_smooth)
    
    # Avoid division by zero
    de_safe = np.where(np.abs(de) < 1e-12, 1e-12, de)
    derivative = dj / de_safe
    
    # Find max derivative, excluding first/last 10% of points
    exclude_n = max(3, int(len(derivative) * 0.1))
    search_range = slice(exclude_n, len(derivative) - exclude_n)
    
    if len(derivative[search_range]) < 5:
        warnings.append("Insufficient range for derivative peak search")
        return None, fit_params, warnings
    
    # Only consider positive derivatives (rising current)
    derivative_search = derivative[search_range].copy()
    derivative_search[derivative_search < 0] = 0
    
    if derivative_search.max() <= 0:
        warnings.append("No rising region found for tangent method")
        return None, fit_params, warnings
    
    idx_peak_local = np.argmax(derivative_search)
    idx_peak = exclude_n + idx_peak_local
    
    # Fit tangent line in window around peak (±5 points)
    window_half = 5
    fit_start = max(0, idx_peak - window_half)
    fit_end = min(len(e_v) - 1, idx_peak + window_half + 1)
    
    if fit_end - fit_start < 4:
        warnings.append("Insufficient points around derivative peak for tangent fit")
        return None, fit_params, warnings
    
    e_fit = e_v[fit_start:fit_end]
    j_fit = j_eff[fit_start:fit_end]
    
    try:
        tangent_coeffs = np.polyfit(e_fit, j_fit, 1)
        tangent_slope = tangent_coeffs[0]
        tangent_intercept = tangent_coeffs[1]
    except Exception as e:
        warnings.append(f"Tangent line fit failed: {e}")
        return None, fit_params, warnings
    
    fit_params["tangent_slope"] = float(tangent_slope)
    fit_params["tangent_intercept"] = float(tangent_intercept)
    fit_params["derivative_peak_idx"] = int(idx_peak)
    
    # Get baseline value/function
    if baseline_params.get("type") == "linear":
        bl_slope = baseline_params["slope"]
        bl_intercept = baseline_params["intercept"]
        
        # For reduction, baseline was computed on -j, so we need to flip
        if direction == "reduction":
            bl_slope = -bl_slope
            bl_intercept = -bl_intercept
        
        # Solve: tangent_slope * E + tangent_intercept = bl_slope * E + bl_intercept
        denom = tangent_slope - bl_slope
        if abs(denom) < 1e-10:
            warnings.append("Tangent and baseline are parallel; cannot find intersection")
            return None, fit_params, warnings
        
        onset_v = (bl_intercept - tangent_intercept) / denom
    else:
        # Constant baseline
        bl_value = baseline_params.get("value", 0)
        if direction == "reduction":
            bl_value = -bl_value
        
        # Solve: tangent_slope * E + tangent_intercept = bl_value
        if abs(tangent_slope) < 1e-10:
            warnings.append("Tangent is horizontal; cannot find intersection")
            return None, fit_params, warnings
        
        onset_v = (bl_value - tangent_intercept) / tangent_slope
    
    fit_params["baseline_params"] = baseline_params
    
    # Validate onset is within sweep bounds
    e_min, e_max = float(np.min(e_v)), float(np.max(e_v))
    if onset_v < e_min or onset_v > e_max:
        warnings.append(f"Tangent intersection at {onset_v:.3f}V is outside sweep range "
                       f"[{e_min:.3f}, {e_max:.3f}]V")
        # Fallback: use the point at max derivative
        onset_v = float(e_v[idx_peak])
        warnings.append(f"Falling back to derivative peak at {onset_v:.3f}V")
    
    return float(onset_v), fit_params, warnings


# =============================================================================
# Main Functions
# =============================================================================

def find_onset_potential(
    e_v: np.ndarray, 
    j_ma_cm2: np.ndarray,
    threshold_ma_cm2: float = 0.1,
    direction: str = "oxidation",
    smooth_window: int = 5,
    baseline_correct: bool = True,
    onset_method: Literal["threshold", "tangent"] = "threshold",
    min_consecutive: int = 5
) -> Dict[str, Any]:
    """
    Find onset potential for electrochemical stability window.
    
    Args:
        e_v: Potential array in V (vs reference)
        j_ma_cm2: Current density array in mA/cm²
        threshold_ma_cm2: Current density threshold for threshold method (default 0.1)
        direction: "oxidation" (positive j) or "reduction" (negative j)
        smooth_window: Smoothing window size (must be odd; auto-corrected if even)
        baseline_correct: Whether to apply baseline correction
        onset_method: "threshold" (consecutive-point crossing) or "tangent" 
                      (baseline–tangent intersection, recommended for GPE)
        min_consecutive: Minimum consecutive points above threshold (threshold method);
                          default 5 helps filter out smoothing-spread noise spikes
    
    Returns:
        dict with:
            - onset_v: Onset potential in V, or None if not found
            - onset_idx: Index in (segmented) array, or None
            - onset_method: Method actually used
            - direction: Direction analyzed
            - threshold_ma_cm2: Threshold used (threshold method)
            - baseline_corrected: Whether baseline was applied
            - baseline_params: Baseline fit parameters
            - tangent_params: Tangent fit details (tangent method only)
            - segment_info: Dict if CV was detected and segmented
            - warnings: List of warnings
            - j_smoothed: Smoothed current array
            - j_corrected: Baseline-corrected current array
    """
    warnings = []
    result = {
        "onset_v": None,
        "onset_idx": None,
        "onset_method": onset_method,
        "direction": direction,
        "threshold_ma_cm2": threshold_ma_cm2,
        "baseline_corrected": baseline_correct,
        "baseline_params": {},
        "tangent_params": {},
        "segment_info": None,
        "warnings": warnings,
        "j_smoothed": None,
        "j_corrected": None
    }
    
    # Input validation
    if len(e_v) < 5:
        result["error"] = "Insufficient data points"
        return result
    
    if direction not in ("oxidation", "reduction"):
        result["error"] = f"Unknown direction: {direction}"
        return result
    
    # =========================================================================
    # CV Detection: Extract monotonic segment if needed
    # =========================================================================
    e_seg, j_seg, idx_map, cv_warnings = _select_monotonic_segment(e_v, j_ma_cm2)
    warnings.extend(cv_warnings)
    
    if len(cv_warnings) > 0:
        result["segment_info"] = {
            "original_length": len(e_v),
            "segment_length": len(e_seg),
            "segment_indices": [int(idx_map[0]), int(idx_map[-1])]
        }
    
    # =========================================================================
    # Direction/Sign Sanity Check
    # =========================================================================
    if direction == "oxidation":
        # Expect some positive current
        if np.max(j_seg) <= 0:
            warnings.append("Direction is 'oxidation' but all currents are non-positive; "
                           "check data or direction setting")
            return result
    else:  # reduction
        # Expect some negative current
        if np.min(j_seg) >= 0:
            warnings.append("Direction is 'reduction' but all currents are non-negative. "
                           "This may indicate data uses absolute values. "
                           "Onset detection skipped - verify data sign convention.")
            return result
    
    # =========================================================================
    # Smoothing
    # =========================================================================
    j_smooth, smooth_warnings = _optional_savgol(j_seg, smooth_window, polyorder=2)
    warnings.extend(smooth_warnings)
    result["j_smoothed"] = j_smooth
    
    # =========================================================================
    # Baseline Correction
    # =========================================================================
    j_corrected = j_smooth.copy()
    baseline_fn = lambda e: 0
    baseline_params = {"type": "none"}
    
    if baseline_correct:
        baseline_fn, baseline_params, bl_warnings = _robust_baseline(e_seg, j_smooth, mode="auto")
        warnings.extend(bl_warnings)
        
        baseline_values = baseline_fn(e_seg)
        j_corrected = j_smooth - baseline_values
        
        # Safety check: ensure correction didn't create huge negative artifacts
        if baseline_params.get("type") == "linear":
            min_corrected = np.min(j_corrected)
            if min_corrected < -abs(np.max(np.abs(j_smooth))) * 0.5:
                warnings.append("Baseline correction created large negative artifacts; "
                               "reverting to constant baseline")
                baseline_fn, baseline_params, _ = _robust_baseline(e_seg, j_smooth, mode="constant")
                baseline_values = baseline_fn(e_seg)
                j_corrected = j_smooth - baseline_values
    
    result["baseline_params"] = baseline_params
    result["j_corrected"] = j_corrected
    
    # =========================================================================
    # Onset Detection
    # =========================================================================
    # Default min_e_span: 1% of sweep range (helps reject smoothing-spread spikes)
    sweep_range = np.max(e_seg) - np.min(e_seg)
    default_min_e_span = sweep_range * 0.01  # 1% of sweep range
    
    if onset_method == "threshold":
        onset_v, onset_idx, onset_warnings = _threshold_onset(
            e_seg, j_corrected, threshold_ma_cm2, direction, min_consecutive,
            min_e_span=default_min_e_span
        )
        warnings.extend(onset_warnings)
        result["onset_v"] = onset_v
        result["onset_idx"] = onset_idx
        
    elif onset_method == "tangent":
        onset_v, tangent_params, onset_warnings = _tangent_onset(
            e_seg, j_corrected, baseline_fn, baseline_params, direction, smooth_window
        )
        warnings.extend(onset_warnings)
        result["onset_v"] = onset_v
        result["tangent_params"] = tangent_params
        
        # For tangent, onset_idx is approximate (at intersection point)
        if onset_v is not None:
            diffs = np.abs(e_seg - onset_v)
            result["onset_idx"] = int(np.argmin(diffs))
    else:
        result["error"] = f"Unknown onset_method: {onset_method}"
    
    return result


def find_stability_window(
    e_v: np.ndarray, 
    j_ma_cm2: np.ndarray,
    threshold_ma_cm2: float = 0.1,
    smooth_window: int = 5,
    onset_method: Literal["threshold", "tangent"] = "threshold",
    min_consecutive: int = 5
) -> Dict[str, Any]:
    """
    Find both oxidation and reduction onset potentials.
    
    Returns the electrochemical stability window as a signed value:
    window_v = oxidation_onset - reduction_onset
    
    A positive window indicates oxidation onset is at higher potential than
    reduction onset (expected behavior). A non-positive window suggests
    data or direction inconsistency.
    
    Args:
        e_v: Potential array in V
        j_ma_cm2: Current density array in mA/cm²
        threshold_ma_cm2: Current density threshold
        smooth_window: Smoothing window size
        onset_method: "threshold" or "tangent"
        min_consecutive: Consecutive points for threshold method
    
    Returns:
        dict with:
            - oxidation_onset_v: Oxidation onset potential or None
            - reduction_onset_v: Reduction onset potential or None
            - window_v: Stability window (signed, ox - red) or None
            - threshold_ma_cm2: Threshold used
            - onset_method: Method used
            - warnings: Combined warnings from both directions
            - oxidation_result: Full result dict from oxidation analysis
            - reduction_result: Full result dict from reduction analysis
    """
    # Find oxidation onset (positive direction)
    ox_result = find_onset_potential(
        e_v, j_ma_cm2, 
        threshold_ma_cm2=threshold_ma_cm2,
        direction="oxidation", 
        smooth_window=smooth_window,
        onset_method=onset_method,
        min_consecutive=min_consecutive
    )
    
    # Find reduction onset (negative direction)
    red_result = find_onset_potential(
        e_v, j_ma_cm2, 
        threshold_ma_cm2=threshold_ma_cm2,
        direction="reduction", 
        smooth_window=smooth_window,
        onset_method=onset_method,
        min_consecutive=min_consecutive
    )
    
    ox_onset = ox_result.get("onset_v")
    red_onset = red_result.get("onset_v")
    
    warnings = ox_result.get("warnings", []) + red_result.get("warnings", [])
    
    window_v = None
    if ox_onset is not None and red_onset is not None:
        # Use signed difference (physically meaningful)
        window_v = ox_onset - red_onset
        
        if window_v <= 0:
            warnings.append(
                f"Stability window ≤ 0 ({window_v:.3f} V): "
                "oxidation onset is not higher than reduction onset. "
                "Check scan directions, data polarity, or direction settings."
            )
    
    return {
        "oxidation_onset_v": ox_onset,
        "reduction_onset_v": red_onset,
        "window_v": window_v,
        "threshold_ma_cm2": threshold_ma_cm2,
        "onset_method": onset_method,
        "warnings": warnings,
        "oxidation_result": ox_result,
        "reduction_result": red_result
    }
