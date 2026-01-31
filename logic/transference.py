"""
Transference number (tLi+) calculation using Bruce-Vincent method.

SIGN CONVENTION:
- Instruments often export negative current; formula uses magnitudes internally.
- Raw values preserved in output; abs() applied with `current_abs_applied` flag.

I0 EXTRACTION:
- Excludes initial capacitive charging spike (auto-detected via derivative)
- Uses median of early-time window after transient settling

Iss EXTRACTION:
- Detects steady-state via derivative analysis on tail
- If steady not detected, returns approximate Iss with qc_pass=False

STRICT MODE:
- strict=True (default): Fails hard on invalid inputs
- strict=False: Returns qc_pass=False with warnings, attempts computation where safe
"""
import numpy as np
from typing import Dict, Any, List, Literal, Tuple


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
    dt_safe = np.where(np.abs(dt) < 1e-12, 1e-12, dt)
    
    # Derivative (one fewer point than input)
    dy = np.diff(y_smooth)
    derivative = dy / dt_safe
    
    return derivative, dt_median


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
    tail_fraction: float = 0.2,
    tail_min_points: int = 20,
    ss_di_dt_tol_rel: float = 0.01,
    ss_min_consecutive: int = 10
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
        
        i0_window_fraction: Fraction of total time for I0 averaging window
        i0_min_points: Minimum points in I0 window
        
        tail_fraction: Fraction of data to consider as tail for Iss
        tail_min_points: Minimum points in tail
        ss_di_dt_tol_rel: Derivative threshold for steady-state detection
        ss_min_consecutive: Minimum consecutive points for steady-state
    
    Returns:
        dict with I0_A, Iss_A, t0_range_s, tss_range_s, and new QC fields:
        - transient_ignored_s: Time ignored for capacitive transient
        - i0_n_points: Points used for I0 calculation
        - iss_n_points: Points used for Iss calculation  
        - ss_detected: Whether true steady-state was detected
        - sorted_applied: Whether time-sorting was applied
        - n_raw: Raw data length
        - n_clean: Clean data length after filtering
        - qc_pass: Overall QC pass flag
        - warnings: List of warnings
    """
    warnings: List[str] = []
    n_raw = len(t_s)
    
    # =========================================================================
    # Preprocessing
    # =========================================================================
    
    # Validate input lengths
    if len(t_s) != len(i_a):
        return {
            "I0_A": None, "Iss_A": None,
            "t0_range_s": None, "tss_range_s": None,
            "transient_ignored_s": 0.0, "i0_n_points": 0, "iss_n_points": 0,
            "ss_detected": False, "sorted_applied": False,
            "n_raw": n_raw, "n_clean": 0, "qc_pass": False,
            "warnings": ["Input arrays have different lengths"]
        }
    
    if len(t_s) < 10:
        return {
            "I0_A": None, "Iss_A": None,
            "t0_range_s": None, "tss_range_s": None,
            "transient_ignored_s": 0.0, "i0_n_points": 0, "iss_n_points": 0,
            "ss_detected": False, "sorted_applied": False,
            "n_raw": n_raw, "n_clean": len(t_s), "qc_pass": False,
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
        t = t[finite_mask]
        i = i[finite_mask]
    
    if len(t) < 10:
        return {
            "I0_A": None, "Iss_A": None,
            "t0_range_s": None, "tss_range_s": None,
            "transient_ignored_s": 0.0, "i0_n_points": 0, "iss_n_points": 0,
            "ss_detected": False, "sorted_applied": False,
            "n_raw": n_raw, "n_clean": len(t), "qc_pass": False,
            "warnings": warnings + ["Insufficient clean data points"]
        }
    
    # Check time monotonicity and sort if needed
    sorted_applied = False
    dt_check = np.diff(t)
    if np.any(dt_check < 0):
        warnings.append("Time array not monotonically increasing; sorted by time")
        sort_idx = np.argsort(t)
        t = t[sort_idx]
        i = i[sort_idx]
        sorted_applied = True
    
    # Check for duplicate times
    if len(np.unique(t)) < len(t):
        warnings.append("Duplicate time values detected; kept original order")
    
    n_clean = len(t)
    t_total = t[-1] - t[0]
    eps = 1e-15
    
    # Work with absolute current
    i_abs = np.abs(i)
    i_scale = np.median(i_abs) + eps
    
    # =========================================================================
    # Transient Detection and I0 Extraction
    # =========================================================================
    
    transient_ignored_s = 0.0
    t_ignore_end = t[0]
    
    if transient_mode == "auto" and len(t) > 20:
        # Compute derivative on smoothed i_abs
        derivative, dt_median = _compute_derivative(t, i_abs, smooth_window=5)
        
        # Derivative threshold (relative to current scale)
        di_dt_threshold = di_dt_tol_rel * i_scale / max(dt_median, eps)
        
        # Find earliest index k where derivative stays small for next M points
        M = min(5, len(derivative) - 1)
        k_settled = None
        
        for k in range(2, len(derivative) - M):  # Start at 2 to skip very beginning
            # Check if derivative is small for next M points
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
        else:
            warnings.append("Capacitive transient not clearly detected; using conservative fallback")
            # Fallback: ignore min_ignore_s or 1% of total, whichever larger
            t_ignore_end = t[0] + max(min_ignore_s, t_total * 0.01)
            transient_ignored_s = t_ignore_end - t[0]
    else:
        # No transient mode: just apply min_ignore_s
        t_ignore_end = t[0] + min_ignore_s
        transient_ignored_s = min_ignore_s
    
    # I0 window: from t_ignore_end to t_ignore_end + (t_total * i0_window_fraction)
    t_i0_start = t_ignore_end
    t_i0_end_target = t_ignore_end + t_total * i0_window_fraction
    
    # Ensure minimum points
    i0_mask = (t >= t_i0_start) & (t <= t_i0_end_target)
    n_i0 = np.sum(i0_mask)
    
    if n_i0 < i0_min_points:
        # Extend window until we have enough points or hit 5% of data
        sorted_indices = np.argsort(t)
        start_idx = np.searchsorted(t[sorted_indices], t_i0_start)
        end_idx = min(start_idx + i0_min_points, int(len(t) * 0.05), len(t))
        if end_idx > start_idx:
            t_i0_end_target = t[sorted_indices[end_idx - 1]]
            i0_mask = (t >= t_i0_start) & (t <= t_i0_end_target)
            n_i0 = np.sum(i0_mask)
    
    # Compute I0
    if n_i0 > 0:
        I0 = float(np.median(i_abs[i0_mask]))
        t0_range = [float(t_i0_start), float(t_i0_end_target)]
    else:
        # Fallback to first available point after transient
        first_valid_idx = np.searchsorted(t, t_i0_start)
        if first_valid_idx < len(t):
            I0 = float(i_abs[first_valid_idx])
            t0_range = [float(t[first_valid_idx]), float(t[first_valid_idx])]
            n_i0 = 1
            warnings.append("I0 extracted from single point (insufficient data in window)")
        else:
            I0 = float(i_abs[0])
            t0_range = [float(t[0]), float(t[0])]
            n_i0 = 1
            warnings.append("I0 fallback to first point")
    
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
    
    # Steady-state detection
    ss_detected = False
    ss_segment_mask = None
    
    if len(t_tail) >= 10:
        # Compute derivative in tail
        deriv_tail, dt_median_tail = _compute_derivative(t_tail, i_tail, smooth_window=5)
        
        # Threshold
        ss_threshold = ss_di_dt_tol_rel * i_scale / max(dt_median_tail, eps)
        
        # Find segment where |deriv| is consistently small
        # Look for ss_min_consecutive consecutive points
        consecutive_count = 0
        ss_start_idx = None
        ss_end_idx = None
        
        for idx, d in enumerate(deriv_tail):
            if np.abs(d) <= ss_threshold:
                if ss_start_idx is None:
                    ss_start_idx = idx
                consecutive_count += 1
                if consecutive_count >= ss_min_consecutive:
                    ss_end_idx = idx + 1  # +1 because derivative is 1 shorter
                    ss_detected = True
                    break
            else:
                ss_start_idx = None
                consecutive_count = 0
        
        if ss_detected and ss_start_idx is not None:
            # Use that segment for Iss (adjust indices for original tail)
            ss_segment_mask = np.zeros(len(t_tail), dtype=bool)
            # Derivative index maps to interval between [i, i+1], so we include point i+1
            seg_start = ss_start_idx
            seg_end = min(ss_end_idx + 1, len(t_tail))
            ss_segment_mask[seg_start:seg_end] = True
    
    # Compute Iss
    qc_pass = True
    
    if ss_detected and ss_segment_mask is not None:
        Iss = float(np.median(i_tail[ss_segment_mask]))
        n_iss = int(np.sum(ss_segment_mask))
        tss_range = [float(t_tail[ss_segment_mask][0]), float(t_tail[ss_segment_mask][-1])]
    else:
        # Fallback: use last portion (legacy behavior) but set qc_pass=False
        n_fallback = max(int(n_tail * 0.5), min(10, n_tail))
        Iss = float(np.median(i_tail[-n_fallback:]))
        n_iss = n_fallback
        tss_range = [float(t_tail[-n_fallback]), float(t_tail[-1])]
        warnings.append("Steady-state not detected; Iss is approximate (last window median)")
        qc_pass = False
    
    return {
        "I0_A": I0,
        "Iss_A": Iss,
        "t0_range_s": t0_range,
        "tss_range_s": tss_range,
        "transient_ignored_s": float(transient_ignored_s),
        "i0_n_points": int(n_i0),
        "iss_n_points": int(n_iss),
        "ss_detected": ss_detected,
        "sorted_applied": sorted_applied,
        "n_raw": n_raw,
        "n_clean": n_clean,
        "qc_pass": qc_pass,
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
    
    Args:
        I0: Initial current (A) - from chronoamperometry after transient
        Iss: Steady-state current (A) - from chronoamperometry end
        R0: Initial interfacial resistance (Ω) - from EIS before polarization
        Rss: Steady-state interfacial resistance (Ω) - from EIS after polarization
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
