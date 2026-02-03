"""
Transference number (tLi+) calculation using Bruce-Vincent method.

SIGN CONVENTION:
- Instruments often export negative current; formula uses magnitudes internally.
- Raw values preserved in output; abs() applied with `current_abs_applied` flag.

I0 EXTRACTION:
- Excludes initial capacitive charging spike (auto-detected via derivative)
- Uses early-biased estimator within post-transient window to avoid Cottrell-decay bias
- Configurable via `i0_method`: "early_median" (default), "legacy_median", "early_quantile", "first_point"

Iss EXTRACTION:
- Detects steady-state via derivative analysis on tail USING TAIL SCALE (not global median)
- If steady not detected, returns approximate Iss with qc_pass=False

RESISTANCE REQUIREMENTS (CRITICAL):
- R0 and Rss must be INTERFACIAL resistance (R_ct + R_SEI), NOT bulk electrolyte Rb
- Using Rb instead of R_interface will severely bias tLi+ (usually upward)
- Extract R_interface from EIS: high-frequency semicircle intercepts, not the HF real-axis intercept

STRICT MODE:
- strict=True (default): Fails hard on invalid inputs
- strict=False: Returns qc_pass=False with warnings, attempts computation where safe
"""
import numpy as np
from typing import Dict, Any, List, Literal, Tuple, Optional


# =============================================================================
# Helper Functions
# =============================================================================

def _rolling_median(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling median with edge handling (numpy-only).
    
    Uses reflect padding at edges for stable boundary behavior.
    """
    if len(arr) < window:
        return np.full_like(arr, np.median(arr))
    
    half = window // 2
    result = np.zeros_like(arr, dtype=float)
    
    # Pad with reflection
    padded = np.concatenate([
        arr[:half][::-1],
        arr,
        arr[-half:][::-1]
    ])
    
    for i in range(len(arr)):
        result[i] = np.median(padded[i:i + window])
    
    return result


def _compute_derivative(
    t: np.ndarray, 
    y: np.ndarray, 
    smooth_window: int = 5
) -> Tuple[np.ndarray, float]:
    """
    Compute derivative dy/dt on smoothed data.
    
    Returns (derivative array, median dt for scaling).
    """
    # Smooth first
    y_smooth = _rolling_median(y, smooth_window)
    
    # Compute dt
    dt = np.diff(t)
    dt_median = np.median(dt) if len(dt) > 0 else 1.0
    
    # Avoid division by zero
    eps = 1e-12
    dt_safe = np.where(np.abs(dt) < eps, eps, dt)
    
    # Derivative (one fewer point than input)
    dy = np.diff(y_smooth)
    derivative = dy / dt_safe
    
    return derivative, dt_median


def _deduplicate_time_series(
    t: np.ndarray, 
    i: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Deduplicate time series by grouping identical timestamps and taking median of i.
    
    Returns (t_deduped, i_deduped, n_merged).
    """
    if len(t) == 0:
        return t, i, 0
    
    unique_t, inverse_indices, counts = np.unique(t, return_inverse=True, return_counts=True)
    
    if len(unique_t) == len(t):
        # No duplicates
        return t, i, 0
    
    # There are duplicates - aggregate by median
    n_merged = len(t) - len(unique_t)
    i_deduped = np.zeros(len(unique_t), dtype=float)
    
    for idx in range(len(unique_t)):
        mask = (inverse_indices == idx)
        i_deduped[idx] = np.median(i[mask])
    
    return unique_t, i_deduped, n_merged


# =============================================================================
# I0 and Iss Extraction
# =============================================================================

def extract_currents_from_chrono(
    t_s: np.ndarray, 
    i_a: np.ndarray,
    t0_fraction: float = 0.01,
    ss_fraction: float = 0.1,
    *,
    transient_mode: Literal["auto", "none"] = "auto",
    min_ignore_s: float = 0.0,
    max_ignore_fraction: float = 0.05,
    di_dt_tol_rel: float = 0.02,
    i0_window_fraction: float = 0.01,
    i0_min_points: int = 5,
    i0_method: Literal["early_median", "legacy_median", "early_quantile", "first_point"] = "early_median",
    i0_early_quantile: float = 0.85,
    i0_decay_threshold: float = 0.8,
    tail_fraction: float = 0.2,
    tail_min_points: int = 20,
    ss_di_dt_tol_rel: float = 0.01,
    ss_min_consecutive: int = 10,
    ss_flatness_tol: float = 0.15,
    min_transient_points: int = 3
) -> Dict[str, Any]:
    """
    Extract I0 and Iss from chronoamperometry data with robust transient handling.
    
    I0 is the "faradaic-controlled initial current" after capacitive settling.
    Iss is the steady-state current, validated by derivative analysis.
    
    Args:
        t_s: Time array in seconds
        i_a: Current array in A (sign doesn't matter; abs used internally)
        t0_fraction: Legacy parameter (fraction for I0 averaging, kept for API compat)
        ss_fraction: Legacy parameter (fraction for Iss averaging, kept for API compat)
        
        transient_mode: "auto" to detect capacitive spike, "none" to skip detection
        min_ignore_s: Minimum time to always ignore at start (s)
        max_ignore_fraction: Maximum fraction of total time to ignore for transient
        di_dt_tol_rel: Relative derivative threshold for "settled" detection
        min_transient_points: Minimum number of points to ignore for transient
        
        i0_window_fraction: Fraction of total time for I0 averaging window
        i0_min_points: Minimum points in I0 window
        i0_method: I0 extraction method:
            - "early_median": median of first K points in window (default, early-biased)
            - "legacy_median": median of full window (backward compatible)
            - "early_quantile": high quantile (i0_early_quantile) of window
            - "first_point": first point after transient (most aggressive)
        i0_early_quantile: Quantile to use for "early_quantile" method (default 0.85)
        i0_decay_threshold: If i_end/i_start < this in I0 window, use early-biased method
        
        tail_fraction: Fraction of data to consider as tail for Iss
        tail_min_points: Minimum points in tail
        ss_di_dt_tol_rel: Derivative threshold for steady-state detection
        ss_min_consecutive: Minimum consecutive points for steady-state
        ss_flatness_tol: Relative range (p90-p10)/median tolerance for flatness
    
    Returns:
        dict with I0_A, Iss_A, t0_range_s, tss_range_s, and new QC fields:
        - I0_raw_A: Median of raw (signed) current in I0 window
        - Iss_raw_A: Median of raw (signed) current in Iss window
        - current_abs_applied: True if I0 window or Iss window had negative median
        - i0_method_used: Actual method used for I0 extraction
        - i0_method_params: Parameters for I0 method (K, quantile, etc.)
        - transient_ignored_s: Time ignored for capacitive transient
        - i0_n_points: Points used for I0 calculation
        - iss_n_points: Points used for Iss calculation  
        - ss_detected: Whether true steady-state was detected
        - sorted_applied: Whether time-sorting was applied
        - n_raw: Raw data length
        - n_clean: Clean data length after filtering
        - n_duplicates_merged: Number of duplicate timestamps merged
        - qc_pass: Overall QC pass flag
        - qc_flags: List of machine-readable QC flags
        - warnings: List of warnings
    """
    warnings: List[str] = []
    qc_flags: List[str] = []
    n_raw = len(t_s)
    
    # =========================================================================
    # Preprocessing
    # =========================================================================
    
    # Validate input lengths
    if len(t_s) != len(i_a):
        return {
            "I0_A": None, "Iss_A": None,
            "I0_raw_A": None, "Iss_raw_A": None,
            "current_abs_applied": False,
            "i0_method_used": None, "i0_method_params": {},
            "t0_range_s": None, "tss_range_s": None,
            "transient_ignored_s": 0.0, "i0_n_points": 0, "iss_n_points": 0,
            "ss_detected": False, "sorted_applied": False,
            "n_raw": n_raw, "n_clean": 0, "n_duplicates_merged": 0,
            "qc_pass": False, "qc_flags": ["input_length_mismatch"],
            "warnings": ["Input arrays have different lengths"]
        }
    
    if len(t_s) < 10:
        return {
            "I0_A": None, "Iss_A": None,
            "I0_raw_A": None, "Iss_raw_A": None,
            "current_abs_applied": False,
            "i0_method_used": None, "i0_method_params": {},
            "t0_range_s": None, "tss_range_s": None,
            "transient_ignored_s": 0.0, "i0_n_points": 0, "iss_n_points": 0,
            "ss_detected": False, "sorted_applied": False,
            "n_raw": n_raw, "n_clean": len(t_s), "n_duplicates_merged": 0,
            "qc_pass": False, "qc_flags": ["insufficient_data"],
            "warnings": ["Insufficient data points (< 10)"]
        }
    
    # Convert to numpy arrays and copy
    t = np.asarray(t_s, dtype=float).copy()
    i = np.asarray(i_a, dtype=float).copy()
    
    # Check for finite values
    finite_mask = np.isfinite(t) & np.isfinite(i)
    n_nonfinite = np.sum(~finite_mask)
    if n_nonfinite > 0:
        warnings.append(f"Removed {n_nonfinite} non-finite values")
        qc_flags.append("nonfinite_values_removed")
        t = t[finite_mask]
        i = i[finite_mask]
    
    if len(t) < 10:
        return {
            "I0_A": None, "Iss_A": None,
            "I0_raw_A": None, "Iss_raw_A": None,
            "current_abs_applied": False,
            "i0_method_used": None, "i0_method_params": {},
            "t0_range_s": None, "tss_range_s": None,
            "transient_ignored_s": 0.0, "i0_n_points": 0, "iss_n_points": 0,
            "ss_detected": False, "sorted_applied": False,
            "n_raw": n_raw, "n_clean": len(t), "n_duplicates_merged": 0,
            "qc_pass": False, "qc_flags": qc_flags + ["insufficient_data_after_filter"],
            "warnings": warnings + ["Insufficient clean data points"]
        }
    
    # Check time monotonicity and sort if needed
    sorted_applied = False
    dt_check = np.diff(t)
    if np.any(dt_check < 0):
        warnings.append("Time array not monotonically increasing; sorted by time")
        qc_flags.append("time_sorted")
        sort_idx = np.argsort(t)
        t = t[sort_idx]
        i = i[sort_idx]
        sorted_applied = True
    
    # Handle duplicate times - CRITICAL FIX #3
    n_duplicates_merged = 0
    if len(np.unique(t)) < len(t):
        t, i, n_duplicates_merged = _deduplicate_time_series(t, i)
        warnings.append(f"Merged {n_duplicates_merged} duplicate time values (took median of current)")
        qc_flags.append("duplicate_time_deduped")
    
    n_clean = len(t)
    t_total = t[-1] - t[0]
    eps = 1e-15
    
    if t_total <= 0:
        return {
            "I0_A": None, "Iss_A": None,
            "I0_raw_A": None, "Iss_raw_A": None,
            "current_abs_applied": False,
            "i0_method_used": None, "i0_method_params": {},
            "t0_range_s": None, "tss_range_s": None,
            "transient_ignored_s": 0.0, "i0_n_points": 0, "iss_n_points": 0,
            "ss_detected": False, "sorted_applied": sorted_applied,
            "n_raw": n_raw, "n_clean": n_clean, "n_duplicates_merged": n_duplicates_merged,
            "qc_pass": False, "qc_flags": qc_flags + ["zero_time_span"],
            "warnings": warnings + ["Time span is zero or negative"]
        }
    
    # Work with absolute current
    i_abs = np.abs(i)
    
    # =========================================================================
    # Transient Detection and I0 Extraction
    # =========================================================================
    
    transient_ignored_s = 0.0
    transient_end_idx = 0
    t_ignore_end = t[0]
    
    if transient_mode == "auto" and len(t) > 20:
        # Compute derivative on smoothed i_abs
        # Use local scale for early region, not global
        derivative, dt_median = _compute_derivative(t, i_abs, smooth_window=5)
        
        # Use early region scale for threshold (first 10% or 20 points)
        n_early = max(min(int(len(i_abs) * 0.1), 20), 5)
        i_early_scale = np.median(i_abs[:n_early]) + eps
        
        di_dt_threshold = di_dt_tol_rel * i_early_scale / max(dt_median, eps)
        
        # Find earliest index k where derivative stays small for next M points
        M = min(5, len(derivative) - 1)
        k_settled = None
        
        # Start at min_transient_points to enforce minimum transient
        start_k = max(2, min_transient_points)
        for k in range(start_k, len(derivative) - M):
            window_abs_deriv = np.abs(derivative[k:k + M])
            if np.all(window_abs_deriv <= di_dt_threshold):
                k_settled = k
                break
        
        if k_settled is not None:
            # The index in original array (derivative is 1 shorter)
            t_settled = t[k_settled]
            t_ignore_end = max(t_settled, t[0] + min_ignore_s)
            
            # Cap by max_ignore_fraction
            t_max_ignore = t[0] + t_total * max_ignore_fraction
            t_ignore_end = min(t_ignore_end, t_max_ignore)
            
            transient_ignored_s = t_ignore_end - t[0]
            transient_end_idx = np.searchsorted(t, t_ignore_end)
            
            # Guard: check if transient end is very late (slow double-layer charging)
            if transient_ignored_s > t_total * 0.03:
                warnings.append(f"Transient end at {transient_ignored_s:.3f}s ({transient_ignored_s/t_total*100:.1f}% of total); "
                               "slow double-layer charging may bias I0")
                qc_flags.append("transient_end_late")
        else:
            warnings.append("Capacitive transient not clearly detected; using conservative fallback")
            qc_flags.append("transient_detection_failed")
            # Fallback: ignore min_ignore_s or 1% of total, whichever larger
            t_ignore_end = t[0] + max(min_ignore_s, t_total * 0.01)
            transient_ignored_s = t_ignore_end - t[0]
            transient_end_idx = np.searchsorted(t, t_ignore_end)
    else:
        # No transient mode: just apply min_ignore_s
        t_ignore_end = t[0] + min_ignore_s
        transient_ignored_s = min_ignore_s
        transient_end_idx = np.searchsorted(t, t_ignore_end)
    
    # I0 window: from t_ignore_end to t_ignore_end + (t_total * i0_window_fraction)
    t_i0_start = t_ignore_end
    t_i0_end_target = t_ignore_end + t_total * i0_window_fraction
    
    # Ensure minimum points - FIXED: window cap relative to start_idx, not absolute 5%
    i0_mask = (t >= t_i0_start) & (t <= t_i0_end_target)
    n_i0 = np.sum(i0_mask)
    
    if n_i0 < i0_min_points:
        # Extend window until we have enough points
        # Cap is relative: at most 2x the desired window or i0_min_points*2, not absolute 5%
        start_idx = np.searchsorted(t, t_i0_start)
        max_window_points = max(i0_min_points * 2, int((n_clean - start_idx) * 0.1))
        end_idx = min(start_idx + max(i0_min_points, n_i0), start_idx + max_window_points, len(t))
        
        if end_idx > start_idx:
            t_i0_end_target = t[end_idx - 1]
            i0_mask = (t >= t_i0_start) & (t <= t_i0_end_target)
            n_i0 = np.sum(i0_mask)
    
    # Get I0 window data
    i0_method_used = i0_method
    i0_method_params: Dict[str, Any] = {}
    I0_raw: Optional[float] = None
    current_abs_applied = False
    
    if n_i0 > 0:
        i0_window_abs = i_abs[i0_mask]
        i0_window_raw = i[i0_mask]
        
        # Compute raw median for output
        I0_raw = float(np.median(i0_window_raw))
        if I0_raw < 0:
            current_abs_applied = True
        
        # Determine if window shows strong decay (Cottrell-like)
        i_start_window = i0_window_abs[0] if len(i0_window_abs) > 0 else 1.0
        i_end_window = i0_window_abs[-1] if len(i0_window_abs) > 0 else 1.0
        decay_ratio = i_end_window / (i_start_window + eps)
        strong_decay = decay_ratio < i0_decay_threshold
        
        if strong_decay and i0_method == "early_median":
            # Use early-biased method (first K points)
            K = min(i0_min_points, len(i0_window_abs))
            I0 = float(np.median(i0_window_abs[:K]))
            i0_method_params = {"K": K, "decay_ratio": float(decay_ratio)}
            qc_flags.append("i0_strong_decay")
            warnings.append(f"I0 window shows strong decay (ratio={decay_ratio:.2f}); using early-biased median (K={K})")
        elif i0_method == "legacy_median":
            I0 = float(np.median(i0_window_abs))
            i0_method_params = {"window_size": len(i0_window_abs)}
        elif i0_method == "early_median":
            # No strong decay, but still use early portion
            K = min(i0_min_points, len(i0_window_abs))
            I0 = float(np.median(i0_window_abs[:K]))
            i0_method_params = {"K": K}
        elif i0_method == "early_quantile":
            # Skip first few points (capacitive residual), then take high quantile
            skip = min(2, len(i0_window_abs) // 4)
            window_for_quantile = i0_window_abs[skip:] if skip < len(i0_window_abs) else i0_window_abs
            I0 = float(np.percentile(window_for_quantile, i0_early_quantile * 100))
            i0_method_params = {"quantile": i0_early_quantile, "skip_initial": skip}
        elif i0_method == "first_point":
            I0 = float(i0_window_abs[0])
            i0_method_params = {"note": "single_point"}
            qc_flags.append("i0_single_point")
        else:
            # Default fallback
            I0 = float(np.median(i0_window_abs))
            i0_method_used = "legacy_median"
            i0_method_params = {"window_size": len(i0_window_abs)}
        
        t0_range = [float(t_i0_start), float(t_i0_end_target)]
    else:
        # Fallback to first available point after transient
        first_valid_idx = np.searchsorted(t, t_i0_start)
        if first_valid_idx < len(t):
            I0 = float(i_abs[first_valid_idx])
            I0_raw = float(i[first_valid_idx])
            if I0_raw < 0:
                current_abs_applied = True
            t0_range = [float(t[first_valid_idx]), float(t[first_valid_idx])]
            n_i0 = 1
            i0_method_used = "first_point"
            i0_method_params = {"note": "fallback_single_point"}
            warnings.append("I0 extracted from single point (insufficient data in window)")
            qc_flags.append("i0_single_point")
        else:
            I0 = float(i_abs[0])
            I0_raw = float(i[0])
            if I0_raw < 0:
                current_abs_applied = True
            t0_range = [float(t[0]), float(t[0])]
            n_i0 = 1
            i0_method_used = "first_point"
            i0_method_params = {"note": "fallback_first_point"}
            warnings.append("I0 fallback to first point")
            qc_flags.append("i0_single_point")
    
    # =========================================================================
    # Iss Extraction with Steady-State Detection
    # =========================================================================
    
    # Tail region
    tail_start_t = t[-1] - t_total * tail_fraction
    tail_mask = t >= tail_start_t
    n_tail = np.sum(tail_mask)
    
    # Ensure minimum points in tail
    if n_tail < tail_min_points:
        # Take last tail_min_points
        n_take = min(tail_min_points, len(t))
        tail_mask = np.zeros(len(t), dtype=bool)
        tail_mask[-n_take:] = True
        n_tail = n_take
        tail_start_t = t[tail_mask][0]
    
    t_tail = t[tail_mask]
    i_tail = i_abs[tail_mask]
    i_tail_raw = i[tail_mask]
    
    # CRITICAL FIX #2: Use TAIL scale for threshold, not global scale
    i_tail_scale = np.median(i_tail) + eps
    
    # Steady-state detection
    ss_detected = False
    ss_segment_mask = None
    
    if len(t_tail) >= 10:
        # Compute derivative in tail
        deriv_tail, dt_median_tail = _compute_derivative(t_tail, i_tail, smooth_window=5)
        
        # Threshold based on TAIL scale (not global)
        ss_threshold = ss_di_dt_tol_rel * i_tail_scale / max(dt_median_tail, eps)
        
        # Find segment where |deriv| is consistently small
        consecutive_count = 0
        ss_start_idx = None
        ss_end_idx = None
        
        for idx, d in enumerate(deriv_tail):
            if np.abs(d) <= ss_threshold:
                if ss_start_idx is None:
                    ss_start_idx = idx
                consecutive_count += 1
                if consecutive_count >= ss_min_consecutive:
                    ss_end_idx = idx + 1
                    ss_detected = True
                    break
            else:
                ss_start_idx = None
                consecutive_count = 0
        
        if ss_detected and ss_start_idx is not None:
            # Build segment mask
            ss_segment_mask = np.zeros(len(t_tail), dtype=bool)
            seg_start = ss_start_idx
            seg_end = min(ss_end_idx + 1, len(t_tail))
            ss_segment_mask[seg_start:seg_end] = True
            
            # ADDITIONAL FLATNESS CHECK
            segment_vals = i_tail[ss_segment_mask]
            if len(segment_vals) >= 3:
                p10 = np.percentile(segment_vals, 10)
                p90 = np.percentile(segment_vals, 90)
                seg_median = np.median(segment_vals)
                relative_range = (p90 - p10) / (seg_median + eps)
                
                if relative_range > ss_flatness_tol:
                    warnings.append(f"Steady-state segment has high variability (range/median={relative_range:.2f})")
                    qc_flags.append("ss_high_variability")
    
    # Compute Iss
    qc_pass = True
    Iss_raw: Optional[float] = None
    
    if ss_detected and ss_segment_mask is not None:
        Iss = float(np.median(i_tail[ss_segment_mask]))
        Iss_raw = float(np.median(i_tail_raw[ss_segment_mask]))
        if Iss_raw < 0:
            current_abs_applied = True
        n_iss = int(np.sum(ss_segment_mask))
        tss_range = [float(t_tail[ss_segment_mask][0]), float(t_tail[ss_segment_mask][-1])]
    else:
        # Fallback: use last portion (legacy behavior) but set qc_pass=False
        n_fallback = max(int(n_tail * 0.5), min(10, n_tail))
        Iss = float(np.median(i_tail[-n_fallback:]))
        Iss_raw = float(np.median(i_tail_raw[-n_fallback:]))
        if Iss_raw < 0:
            current_abs_applied = True
        n_iss = n_fallback
        tss_range = [float(t_tail[-n_fallback]), float(t_tail[-1])]
        warnings.append("Steady-state not detected; Iss is approximate (last window median)")
        qc_flags.append("ss_not_detected")
        qc_pass = False
    
    return {
        "I0_A": I0,
        "Iss_A": Iss,
        "I0_raw_A": I0_raw,
        "Iss_raw_A": Iss_raw,
        "current_abs_applied": current_abs_applied,
        "i0_method_used": i0_method_used,
        "i0_method_params": i0_method_params,
        "t0_range_s": t0_range,
        "tss_range_s": tss_range,
        "transient_ignored_s": float(transient_ignored_s),
        "i0_n_points": int(n_i0),
        "iss_n_points": int(n_iss),
        "ss_detected": ss_detected,
        "sorted_applied": sorted_applied,
        "n_raw": n_raw,
        "n_clean": n_clean,
        "n_duplicates_merged": n_duplicates_merged,
        "qc_pass": qc_pass,
        "qc_flags": qc_flags,
        "warnings": warnings
    }


# =============================================================================
# Main Computation
# =============================================================================

def compute_transference(
    I0: float, 
    Iss: float, 
    R0: float, 
    Rss: float,
    delta_V: float,
    *,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Calculate Li+ transference number using Bruce-Vincent (Evans) method.
    
    tLi+ = Iss × (ΔV - I0×R0) / [I0 × (ΔV - Iss×Rss)]
    
    This version includes rigorous input validation:
    - Uses absolute values of currents (instruments may export negative)
    - Validates effective polarization voltages
    - Checks delta_V is in linear regime (≤ 10-20 mV typical)
    - Strict mode fails hard on invalid inputs; lenient mode returns qc_pass=False
    
    CRITICAL: R0/Rss must be INTERFACIAL RESISTANCE (R_ct + R_SEI from EIS semicircle),
    NOT bulk electrolyte resistance Rb. Using Rb will severely bias tLi+ upward.
    
    Args:
        I0: Initial current (A) - from chronoamperometry after transient
        Iss: Steady-state current (A) - from chronoamperometry end
        R0: Initial interfacial resistance (Ω) - from EIS before polarization
            MUST be R_interface (Rct + Rsei), NOT Rb
        Rss: Steady-state interfacial resistance (Ω) - from EIS after polarization
            MUST be R_interface (Rct + Rsei), NOT Rb
        delta_V: Applied potential step (V)
        strict: If True (default), invalid inputs cause success=False.
                If False, attempts calculation with qc_pass=False and warnings.
    
    Returns:
        dict with:
        - t_li_plus: Transference number, or None if computation invalid
        - success: Whether computation succeeded
        - qc_pass: Whether all QC checks passed
        - qc_flags: List of specific QC issues
        - warnings: List of warning messages
        - params: Input parameters (including raw and effective values)
        - numerator: Formula numerator
        - denominator: Formula denominator
        - dV_eff0: Effective voltage at t=0 (delta_V - I0*R0)
        - dV_effss: Effective voltage at steady-state (delta_V - Iss*Rss)
        - v_drop_initial: Initial IR drop (I0*R0)
        - v_drop_ss: Steady-state IR drop (Iss*Rss)
        - current_abs_applied: Whether abs() was applied to currents
        - strict_used: Value of strict parameter
    """
    warnings: List[str] = []
    qc_flags: List[str] = []
    qc_pass = True
    success = True
    
    # Preserve raw values
    I0_raw = I0
    Iss_raw = Iss
    
    # Apply absolute value to currents (handle negative convention)
    current_abs_applied = False
    if I0 < 0 or Iss < 0:
        current_abs_applied = True
        if I0 < 0:
            warnings.append(f"I0 was negative ({I0:.2e} A); using absolute value")
        if Iss < 0:
            warnings.append(f"Iss was negative ({Iss:.2e} A); using absolute value")
    
    I0_eff = abs(I0)
    Iss_eff = abs(Iss)
    
    # Build params dict
    params = {
        "I0_A": I0_eff,
        "Iss_A": Iss_eff,
        "I0_raw_A": I0_raw,
        "Iss_raw_A": Iss_raw,
        "R0_ohm": R0,
        "Rss_ohm": Rss,
        "delta_V_V": delta_V
    }
    
    # CRITICAL WARNING #5: Resistance definition
    warnings.append(
        "IMPORTANT: R0/Rss must be interfacial resistance (Rct+Rsei from EIS semicircle), "
        "NOT bulk electrolyte Rb. Using Rb will severely bias tLi+ upward."
    )
    qc_flags.append("resistance_definition_assumed_interface")
    
    # Helper for failure return
    def _fail_result(t_val=None, num=None, den=None, dv0=None, dvss=None, vd0=None, vdss=None):
        return {
            "t_li_plus": t_val,
            "warnings": warnings,
            "params": params,
            "success": False,
            "qc_pass": False,
            "qc_flags": qc_flags,
            "numerator": num,
            "denominator": den,
            "dV_eff0": dv0,
            "dV_effss": dvss,
            "v_drop_initial": vd0,
            "v_drop_ss": vdss,
            "current_abs_applied": current_abs_applied,
            "strict_used": strict
        }
    
    # =========================================================================
    # Input Validation
    # =========================================================================
    
    # delta_V checks
    if delta_V <= 0:
        warnings.append("ΔV must be positive (applied potential step)")
        qc_flags.append("deltaV_nonpositive")
        return _fail_result()
    
    if delta_V > 0.10:
        qc_flags.append("deltaV_too_large")
        warnings.append(f"ΔV = {delta_V*1000:.1f} mV is very large (>100 mV); "
                       "Bruce-Vincent assumes small polarization (typically ≤10 mV)")
        qc_pass = False
        if strict:
            warnings.append("Strict mode: failing due to excessive ΔV")
            return _fail_result()
    elif delta_V > 0.02:
        warnings.append(f"ΔV = {delta_V*1000:.1f} mV may exceed linear regime "
                       "(recommended ≤20 mV for GPE)")
    
    # Current checks
    if I0_eff <= 0:
        warnings.append("I0 is zero after abs(); check measurement")
        qc_flags.append("I0_zero")
        return _fail_result()
    
    if Iss_eff <= 0:
        warnings.append("Iss is zero after abs(); check measurement")
        qc_flags.append("Iss_zero")
        return _fail_result()
    
    # Resistance checks
    if R0 <= 0:
        warnings.append("R0 must be positive (initial resistance)")
        qc_flags.append("R0_nonpositive")
        if strict:
            return _fail_result()
        else:
            qc_pass = False
            # Cannot compute with non-positive R
            return _fail_result()
    
    if Rss <= 0:
        warnings.append("Rss must be positive (steady-state resistance)")
        qc_flags.append("Rss_nonpositive")
        if strict:
            return _fail_result()
        else:
            qc_pass = False
            return _fail_result()
    
    # =========================================================================
    # Effective Voltage Computation
    # =========================================================================
    
    v_drop_initial = I0_eff * R0
    v_drop_ss = Iss_eff * Rss
    dV_eff0 = delta_V - v_drop_initial
    dV_effss = delta_V - v_drop_ss
    
    # Check effective voltages
    if dV_eff0 <= 0:
        warnings.append(f"Effective voltage at t=0 is non-positive: "
                       f"ΔV_eff0 = {delta_V:.4f} - {v_drop_initial:.4f} = {dV_eff0:.4f} V. "
                       "Initial IR drop exceeds applied voltage.")
        qc_flags.append("effective_voltage_nonpositive")
        qc_pass = False
        if strict:
            return _fail_result(dv0=dV_eff0, dvss=dV_effss, vd0=v_drop_initial, vdss=v_drop_ss)
        else:
            # Cannot compute valid tLi+ with non-positive effective voltage
            return _fail_result(dv0=dV_eff0, dvss=dV_effss, vd0=v_drop_initial, vdss=v_drop_ss)
    
    if dV_effss <= 0:
        warnings.append(f"Effective voltage at steady-state is non-positive: "
                       f"ΔV_effss = {delta_V:.4f} - {v_drop_ss:.4f} = {dV_effss:.4f} V. "
                       "Steady-state IR drop exceeds applied voltage.")
        qc_flags.append("effective_voltage_nonpositive")
        qc_pass = False
        if strict:
            return _fail_result(dv0=dV_eff0, dvss=dV_effss, vd0=v_drop_initial, vdss=v_drop_ss)
        else:
            return _fail_result(dv0=dV_eff0, dvss=dV_effss, vd0=v_drop_initial, vdss=v_drop_ss)
    
    # =========================================================================
    # Calculate tLi+
    # =========================================================================
    
    numerator = Iss_eff * dV_eff0
    denominator = I0_eff * dV_effss
    
    # Denominator check
    if abs(denominator) < 1e-15:
        warnings.append("Denominator near zero - numerical instability")
        qc_flags.append("denominator_near_zero")
        return _fail_result(num=numerator, den=denominator, 
                           dv0=dV_eff0, dvss=dV_effss, vd0=v_drop_initial, vdss=v_drop_ss)
    
    t = numerator / denominator
    
    # =========================================================================
    # Result QC Checks
    # =========================================================================
    
    # Check Iss >= I0 (unusual, suggests not at steady state)
    if Iss_eff >= I0_eff:
        warnings.append(f"Iss ({Iss_eff:.2e} A) ≥ I0 ({I0_eff:.2e} A): "
                       "This is unusual and may indicate measurement not at steady-state, "
                       "or I0 extracted during capacitive transient")
        qc_flags.append("iss_ge_i0")
        qc_pass = False
    
    # Result range checks
    if t < 0:
        qc_flags.append("t_negative")
        warnings.append(f"tLi+ = {t:.4f} < 0: Likely causes: wrong R assignment, "
                       "sign errors, or non-blocking electrodes")
        qc_pass = False
    elif t > 1:
        qc_flags.append("t_above_unity")
        warnings.append(f"tLi+ = {t:.4f} > 1: Likely causes: measurement not at steady state, "
                       "parasitic reactions, or incorrect Rss")
        qc_pass = False
    elif t < 0.1:
        warnings.append(f"tLi+ = {t:.4f} is unusually low for SPE (expected 0.2-0.6)")
    elif t > 0.8:
        warnings.append(f"tLi+ = {t:.4f} is unusually high for SPE (expected 0.2-0.6)")
    
    # Voltage drop warnings (original logic)
    if v_drop_initial > delta_V * 0.5:
        warnings.append(f"I0×R0 = {v_drop_initial:.4f} V is >50% of ΔV: "
                       "Large initial resistive drop")
    if v_drop_ss > delta_V * 0.5:
        warnings.append(f"Iss×Rss = {v_drop_ss:.4f} V is >50% of ΔV: "
                       "Large steady-state resistive drop")
    
    return {
        "t_li_plus": float(t),
        "warnings": warnings,
        "params": params,
        "success": True,
        "qc_pass": qc_pass,
        "qc_flags": qc_flags,
        "numerator": float(numerator),
        "denominator": float(denominator),
        "dV_eff0": float(dV_eff0),
        "dV_effss": float(dV_effss),
        "v_drop_initial": float(v_drop_initial),
        "v_drop_ss": float(v_drop_ss),
        "current_abs_applied": current_abs_applied,
        "strict_used": strict
    }
