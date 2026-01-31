"""
Smart Window analysis module.

Handles electrochromic device metrics: ΔT, response time, coloration efficiency,
and cycle segmentation.

SCIPY OPTIONAL:
- Savitzky-Golay smoothing and peak detection require SciPy; degrade to numpy if missing.
- Integration uses np.trapz (numpy-only).

TIME ALIGNMENT:
- CA (electrochemistry) and optical transmittance may have trigger delays (1-5 s typical).
- align_ca_transmittance() can estimate and correct this lag via cross-correlation.

RESPONSE TIME:
- Uses plateau-based definition: T0/Tinf from segment endpoints, not raw endpoints.
- Returns None with warning if plateau not reached.

CE COMPUTATION:
- Computed ONLY for coloring segments (ΔOD > 0).
- Full color+bleach cycles are auto-split to avoid Q_net ~ 0 blowup.

BASELINE CORRECTION:
- Default is OFF. Never fit baseline to initial transient.
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any, Literal

# =============================================================================
# Optional SciPy import (safe for Streamlit Cloud without scipy)
# =============================================================================
SCIPY_AVAILABLE = False
savgol_filter = None
find_peaks = None

try:
    from scipy.signal import savgol_filter as _savgol, find_peaks as _find_peaks
    savgol_filter = _savgol
    find_peaks = _find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# Numpy-only Fallbacks
# =============================================================================

def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average for smoothing (numpy-only fallback)."""
    if window % 2 == 0:
        window += 1
    if len(arr) <= window:
        return arr.copy()
    
    kernel = np.ones(window) / window
    pad_left = window // 2
    pad_right = window - pad_left - 1
    arr_padded = np.concatenate([
        np.full(pad_left, arr[0]),
        arr,
        np.full(pad_right, arr[-1])
    ])
    return np.convolve(arr_padded, kernel, mode='valid')


def _optional_smooth(arr: np.ndarray, window: int = 11, polyorder: int = 2) -> Tuple[np.ndarray, List[str]]:
    """Apply Savitzky-Golay if available, else moving average."""
    warnings = []
    
    if window % 2 == 0:
        window += 1
    
    if len(arr) <= window:
        return arr.copy(), ["Insufficient points for smoothing"]
    
    if SCIPY_AVAILABLE and savgol_filter is not None:
        try:
            return savgol_filter(arr, window, polyorder), warnings
        except Exception as e:
            warnings.append(f"Savgol failed: {e}; using moving average")
    else:
        warnings.append("SciPy not installed; using numpy moving-average for smoothing")
    
    return _moving_average(arr, window), warnings


def _simple_find_extrema(arr: np.ndarray, prominence: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple extrema detector (numpy-only fallback for find_peaks).
    
    Returns (peak_indices, valley_indices).
    """
    if len(arr) < 3:
        return np.array([], dtype=int), np.array([], dtype=int)
    
    peaks = []
    valleys = []
    
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i-1] and arr[i] > arr[i+1]:
            # Check prominence
            left_min = np.min(arr[max(0, i-10):i])
            right_min = np.min(arr[i+1:min(len(arr), i+11)])
            prom = arr[i] - max(left_min, right_min)
            if prom >= prominence:
                peaks.append(i)
        elif arr[i] < arr[i-1] and arr[i] < arr[i+1]:
            left_max = np.max(arr[max(0, i-10):i])
            right_max = np.max(arr[i+1:min(len(arr), i+11)])
            prom = min(left_max, right_max) - arr[i]
            if prom >= prominence:
                valleys.append(i)
    
    return np.array(peaks, dtype=int), np.array(valleys, dtype=int)


# =============================================================================
# Time Alignment with Lag Estimation
# =============================================================================

def _estimate_time_lag(
    t_ca: np.ndarray, i_ca: np.ndarray,
    t_opt: np.ndarray, t_frac: np.ndarray,
    max_lag_s: float = 10.0,
    dt: float = 0.2,
    smooth_window: int = 11
) -> Tuple[float, float, bool, List[str]]:
    """
    Estimate time lag between CA and optical data using cross-correlation.
    
    Args:
        t_ca: CA time array (s)
        i_ca: CA current array (A)
        t_opt: Optical time array (s)
        t_frac: Transmittance array [0,1]
        max_lag_s: Maximum lag to search
        dt: Resampling interval
        smooth_window: Smoothing window for derivative
    
    Returns:
        (lag_s, correlation, confident, warnings)
        - lag_s: Estimated lag (positive = optical lags current)
        - correlation: Max correlation coefficient
        - confident: Whether correlation is above threshold
        - warnings: List of warnings
    """
    warnings = []
    
    # Build common time grid
    t_min = max(t_ca.min(), t_opt.min())
    t_max = min(t_ca.max(), t_opt.max())
    
    if t_max - t_min < max_lag_s * 2:
        warnings.append(f"Overlap too short for lag estimation ({t_max - t_min:.1f} s)")
        return 0.0, 0.0, False, warnings
    
    t_common = np.arange(t_min, t_max, dt)
    if len(t_common) < 20:
        warnings.append("Insufficient points for lag estimation")
        return 0.0, 0.0, False, warnings
    
    # Interpolate CA: use |I| as signal
    i_interp = np.interp(t_common, t_ca, np.abs(i_ca))
    i_smooth, _ = _optional_smooth(i_interp, smooth_window)
    
    # Interpolate optical: use -dT/dt (rate of darkening aligns with current)
    t_frac_interp = np.interp(t_common, t_opt, t_frac)
    t_frac_smooth, _ = _optional_smooth(t_frac_interp, smooth_window)
    
    # Compute derivative: -dT/dt
    dt_optical = -np.gradient(t_frac_smooth, dt)
    dt_optical_smooth, _ = _optional_smooth(dt_optical, smooth_window // 2 + 1)
    
    # Normalize signals
    if np.std(i_smooth) < 1e-12 or np.std(dt_optical_smooth) < 1e-12:
        warnings.append("Signal variance too low for correlation")
        return 0.0, 0.0, False, warnings
    
    i_norm = (i_smooth - np.mean(i_smooth)) / np.std(i_smooth)
    opt_norm = (dt_optical_smooth - np.mean(dt_optical_smooth)) / np.std(dt_optical_smooth)
    
    # Cross-correlation over lag range
    n_lags = int(max_lag_s / dt)
    lags = np.arange(-n_lags, n_lags + 1) * dt
    correlations = []
    
    for lag_samples in range(-n_lags, n_lags + 1):
        if lag_samples >= 0:
            i_seg = i_norm[lag_samples:]
            opt_seg = opt_norm[:len(i_seg)]
        else:
            opt_seg = opt_norm[-lag_samples:]
            i_seg = i_norm[:len(opt_seg)]
        
        if len(i_seg) < 10:
            correlations.append(0.0)
            continue
        
        corr = np.corrcoef(i_seg, opt_seg)[0, 1]
        correlations.append(corr if np.isfinite(corr) else 0.0)
    
    correlations = np.array(correlations)
    best_idx = np.argmax(correlations)
    best_lag = lags[best_idx]
    best_corr = correlations[best_idx]
    
    # Confidence check
    confident = best_corr > 0.3  # Reasonable threshold
    
    if not confident:
        warnings.append(f"Low correlation ({best_corr:.3f}); lag estimate may be unreliable")
    
    return float(best_lag), float(best_corr), confident, warnings


def align_ca_transmittance(
    ca_df: pd.DataFrame, 
    tt_df: pd.DataFrame,
    tolerance_s: float = 0.5,
    lag_mode: Literal["estimate", "manual", "none"] = "none",
    max_lag_s: float = 10.0,
    manual_lag_s: float = 0.0,
    direction: str = "optical_lags_current"
) -> Tuple[pd.DataFrame, Dict]:
    """
    Align chronoamperometry (I-t) and transmittance (T-t) data.
    
    Uses merge_asof for nearest-neighbor time matching.
    Optionally estimates and corrects trigger delay between instruments.
    
    Args:
        ca_df: DataFrame with columns t_s, i_a (and optional v_v)
        tt_df: DataFrame with columns t_s, t_frac
        tolerance_s: Maximum time difference for matching
        lag_mode: "estimate" (auto), "manual" (user-provided), "none" (skip)
        max_lag_s: Maximum lag to search in estimate mode
        manual_lag_s: User-provided lag in manual mode
        direction: Lag direction convention
    
    Returns:
        (merged_df, metadata)
    """
    warnings = []
    
    # Ensure sorted
    ca_sorted = ca_df.sort_values("t_s").copy()
    tt_sorted = tt_df.sort_values("t_s").copy()
    
    # Estimate or apply lag
    lag_s = 0.0
    lag_correlation = None
    lag_confident = True
    lag_mode_used = lag_mode
    
    if lag_mode == "estimate":
        lag_s, lag_correlation, lag_confident, lag_warnings = _estimate_time_lag(
            ca_sorted["t_s"].values, ca_sorted["i_a"].values,
            tt_sorted["t_s"].values, tt_sorted["t_frac"].values,
            max_lag_s=max_lag_s
        )
        warnings.extend(lag_warnings)
        
        if not lag_confident and abs(lag_s) > 0.5:
            warnings.append(f"Estimated lag {lag_s:.2f}s has low confidence; using 0")
            lag_s = 0.0
            lag_mode_used = "estimate_fallback_none"
    
    elif lag_mode == "manual":
        lag_s = manual_lag_s
    
    # Apply lag correction to optical time
    if abs(lag_s) > 1e-6:
        tt_sorted = tt_sorted.copy()
        tt_sorted["t_s"] = tt_sorted["t_s"] - lag_s  # Shift optical to align
    
    # Rename columns to avoid conflicts
    tt_sorted = tt_sorted.rename(columns={"t_s": "t_s_tt"})
    
    # Merge on time
    merged = pd.merge_asof(
        ca_sorted, tt_sorted,
        left_on="t_s", right_on="t_s_tt",
        direction="nearest",
        tolerance=tolerance_s
    )
    
    metadata = {
        "ca_time_range": [float(ca_df["t_s"].min()), float(ca_df["t_s"].max())],
        "tt_time_range": [float(tt_df["t_s"].min()), float(tt_df["t_s"].max())],
        "ca_points": len(ca_df),
        "tt_points": len(tt_df),
        "merged_points": len(merged),
        "tolerance_s": tolerance_s,
        "lag_mode": lag_mode,
        "lag_mode_used": lag_mode_used,
        "lag_s": lag_s,
        "lag_correlation": lag_correlation,
        "lag_confident": lag_confident,
        "warnings": warnings
    }
    
    # Drop the extra time column
    if "t_s_tt" in merged.columns:
        merged = merged.drop(columns=["t_s_tt"])
    
    return merged, metadata


# =============================================================================
# Basic Metrics
# =============================================================================

def compute_delta_t(t_bleached: float, t_colored: float) -> float:
    """
    Compute transmittance modulation.
    
    ΔT = T_bleached - T_colored (locked definition)
    
    Args:
        t_bleached: Transmittance in bleached state [0, 1]
        t_colored: Transmittance in colored state [0, 1]
    
    Returns:
        ΔT (always positive for valid EC device)
    """
    return t_bleached - t_colored


# =============================================================================
# Response Time (Plateau-Based)
# =============================================================================

def compute_response_time(
    t_s: np.ndarray, 
    t_frac: np.ndarray,
    threshold: float = 0.9,
    plateau_fraction: float = 0.1,
    smooth_window: int = 5,
    validate_plateau: bool = True,
    plateau_std_max: float = 0.02,
    plateau_slope_max: float = 0.01
) -> Dict[str, Any]:
    """
    Compute response time to reach threshold of full transition.
    
    Uses plateau-based definition:
    - T0 = median of first plateau_fraction of data
    - Tinf = median of last plateau_fraction of data
    - Target = T0 + threshold * (Tinf - T0)
    
    Args:
        t_s: Time array in seconds
        t_frac: Transmittance array [0, 1]
        threshold: Fraction of transition (0.9 = 90%, 0.95 = 95%)
        plateau_fraction: Fraction of data to use for plateau estimation
        smooth_window: Smoothing window for finding crossing
        validate_plateau: Whether to check if plateau was reached
        plateau_std_max: Maximum std for "good" plateau
        plateau_slope_max: Maximum |dT/dt| for "stable" plateau (1/s)
    
    Returns:
        dict with response_time_s, direction, qc_pass, reached_plateau, etc.
    """
    warnings = []
    
    if len(t_s) < 5:
        return {
            "response_time_s": None,
            "direction": None,
            "threshold": threshold,
            "qc_pass": False,
            "reached_plateau": False,
            "warnings": ["Insufficient data points"],
            "t_colored": None,
            "t_bleached": None,
            "delta_t": None
        }
    
    n_plateau = max(3, int(len(t_frac) * plateau_fraction))
    
    # Initial plateau (T0)
    t0_region = t_frac[:n_plateau]
    t0 = float(np.median(t0_region))
    t0_std = float(np.std(t0_region))
    
    # Final plateau (Tinf)
    tinf_region = t_frac[-n_plateau:]
    tinf = float(np.median(tinf_region))
    tinf_std = float(np.std(tinf_region))
    
    # Estimate slope in final region
    if len(t_s) >= n_plateau:
        time_final = t_s[-n_plateau:]
        dt_final = time_final[-1] - time_final[0]
        if dt_final > 0:
            tinf_slope = abs(tinf_region[-1] - tinf_region[0]) / dt_final
        else:
            tinf_slope = 0.0
    else:
        tinf_slope = 0.0
    
    # Plateau quality assessment
    plateau_quality = {
        "t0_std": t0_std,
        "tinf_std": tinf_std,
        "tinf_slope": tinf_slope
    }
    
    reached_plateau = (tinf_std <= plateau_std_max and tinf_slope <= plateau_slope_max)
    
    # Determine direction
    if tinf > t0:
        direction = "bleaching"
        t_colored = t0
        t_bleached = tinf
    else:
        direction = "coloring"
        t_bleached = t0
        t_colored = tinf
    
    delta_t = abs(tinf - t0)
    
    # Target value
    target = t0 + threshold * (tinf - t0)
    
    # Optionally smooth for crossing detection
    if smooth_window > 1:
        t_frac_smooth, _ = _optional_smooth(t_frac, smooth_window)
    else:
        t_frac_smooth = t_frac
    
    # Find first crossing
    response_time = None
    
    if direction == "bleaching":
        # T increases: find where T >= target
        mask = t_frac_smooth >= target
    else:
        # T decreases: find where T <= target
        mask = t_frac_smooth <= target
    
    if mask.any():
        idx = np.argmax(mask)
        
        # Linear interpolation for precision
        if idx > 0:
            t1, t2 = t_s[idx - 1], t_s[idx]
            v1, v2 = t_frac_smooth[idx - 1], t_frac_smooth[idx]
            
            if abs(v2 - v1) > 1e-10:
                # Interpolate to find exact crossing time
                t_cross = t1 + (target - v1) * (t2 - t1) / (v2 - v1)
                response_time = t_cross - t_s[0]
            else:
                response_time = t_s[idx] - t_s[0]
        else:
            response_time = t_s[idx] - t_s[0]
    
    # QC checks
    qc_pass = True
    
    if validate_plateau and not reached_plateau:
        warnings.append(
            f"Final plateau not reached (std={tinf_std:.4f}, slope={tinf_slope:.4f}/s). "
            "Response time may be unreliable."
        )
        qc_pass = False
        # Still return the value but flag it
    
    if response_time is None:
        warnings.append("Threshold not crossed within segment")
        qc_pass = False
    elif response_time < 0:
        warnings.append(f"Negative response time ({response_time:.2f} s); data may be inverted")
        qc_pass = False
    
    return {
        "response_time_s": float(response_time) if response_time is not None else None,
        "direction": direction,
        "threshold": threshold,
        "target_t": target,
        "t0": t0,
        "tinf": tinf,
        "t_colored": float(t_colored),
        "t_bleached": float(t_bleached),
        "delta_t": delta_t,
        "reached_plateau": reached_plateau,
        "plateau_quality": plateau_quality,
        "qc_pass": qc_pass,
        "warnings": warnings
    }


# =============================================================================
# Charge Density
# =============================================================================

def compute_charge_density(
    t_s: np.ndarray, 
    i_a: np.ndarray, 
    area_cm2: float
) -> Dict[str, Any]:
    """
    Compute charge density from current-time data.
    
    Q = ∫I dt / area
    
    Uses numpy trapz for integration (no scipy dependency).
    
    Args:
        t_s: Time array in seconds
        i_a: Current array in A
        area_cm2: Electrode area in cm²
    
    Returns:
        dict with q_c_cm2 (signed), q_abs_c_cm2, q_total_c, q_cumulative
    """
    if len(t_s) < 2:
        return {
            "q_c_cm2": 0.0,
            "q_abs_c_cm2": 0.0,
            "q_total_c": 0.0,
            "q_cumulative": np.array([0.0]),
            "error": "Insufficient data points"
        }
    
    # Integrate using numpy trapz
    q_total = float(np.trapz(i_a, t_s))
    q_density = q_total / area_cm2
    
    # Cumulative charge for plotting
    q_cumulative = np.zeros(len(t_s))
    for i in range(1, len(t_s)):
        q_cumulative[i] = q_cumulative[i-1] + np.trapz(i_a[i-1:i+1], t_s[i-1:i+1])
    
    return {
        "q_c_cm2": float(q_density),
        "q_abs_c_cm2": float(abs(q_density)),
        "q_total_c": q_total,
        "q_signed": float(q_density),
        "q_cumulative": q_cumulative
    }


# =============================================================================
# Step Classification
# =============================================================================

def classify_step(
    t_frac: np.ndarray,
    i_a: Optional[np.ndarray] = None,
    t_s: Optional[np.ndarray] = None,
    area_cm2: float = 1.0,
    use_charge_sign: bool = False,
    od_threshold: float = 0.01
) -> Dict[str, Any]:
    """
    Classify a segment as coloring, bleaching, or unknown.
    
    Args:
        t_frac: Transmittance array [0, 1]
        i_a: Current array (optional, for charge sign corroboration)
        t_s: Time array (optional, for charge calculation)
        area_cm2: Electrode area
        use_charge_sign: Whether to corroborate with charge sign
        od_threshold: Minimum |ΔOD| to classify (avoid noise)
    
    Returns:
        dict with step_type, delta_od, confidence, etc.
    """
    if len(t_frac) < 2:
        return {
            "step_type": "unknown",
            "delta_od": 0.0,
            "confidence": "low",
            "warnings": ["Insufficient data"]
        }
    
    t_start = float(np.median(t_frac[:max(1, len(t_frac)//10)]))
    t_end = float(np.median(t_frac[-max(1, len(t_frac)//10):]))
    
    # Avoid log(0)
    t_start = max(t_start, 1e-6)
    t_end = max(t_end, 1e-6)
    
    # ΔOD = log10(T_initial / T_final) = log10(T_start) - log10(T_end)
    # Positive ΔOD means OD increased (coloring)
    delta_od = float(np.log10(t_start / t_end))
    
    # Classify by ΔOD
    if abs(delta_od) < od_threshold:
        step_type = "unknown"
        confidence = "low"
    elif delta_od > 0:
        step_type = "coloring"
        confidence = "high" if delta_od > 0.1 else "medium"
    else:
        step_type = "bleaching"
        confidence = "high" if delta_od < -0.1 else "medium"
    
    result = {
        "step_type": step_type,
        "delta_od": delta_od,
        "t_start": t_start,
        "t_end": t_end,
        "confidence": confidence,
        "warnings": []
    }
    
    # Corroborate with charge sign if requested
    if use_charge_sign and i_a is not None and t_s is not None:
        q_result = compute_charge_density(t_s, i_a, area_cm2)
        q_signed = q_result.get("q_signed", 0.0)
        
        # Typically: coloring = cathodic (negative Q), bleaching = anodic (positive Q)
        # But this depends on device; just check consistency
        result["q_signed"] = q_signed
        
        if step_type == "coloring" and q_signed > 0:
            result["warnings"].append(
                "ΔOD suggests coloring but Q is positive (anodic); check device polarity"
            )
            result["confidence"] = "low"
        elif step_type == "bleaching" and q_signed < 0:
            result["warnings"].append(
                "ΔOD suggests bleaching but Q is negative (cathodic); check device polarity"
            )
            result["confidence"] = "low"
    
    return result


# =============================================================================
# Coloration Efficiency
# =============================================================================

def compute_coloration_efficiency(
    t_bleached: float, 
    t_colored: float,
    q_c_cm2: float,
    step_type: Optional[str] = None,
    q_min_threshold: float = 1e-6,
    force_compute: bool = False
) -> Dict[str, Any]:
    """
    Compute coloration efficiency.
    
    CE = ΔOD / |Q|
    ΔOD = log10(Tb / Tc)  -- base-10 locked
    
    By default, CE is only computed for coloring segments.
    
    Args:
        t_bleached: Transmittance in bleached state [0, 1]
        t_colored: Transmittance in colored state [0, 1]
        q_c_cm2: Charge density in C/cm² (use absolute value internally)
        step_type: "coloring", "bleaching", or None (auto-detect)
        q_min_threshold: Minimum |Q| to compute CE (prevents blowup)
        force_compute: If True, compute CE even for bleaching segments
    
    Returns:
        dict with ce_cm2_c (CE in cm²/C), delta_od, or ce_skipped_reason
    """
    errors = []
    warnings = []
    
    # Validate inputs
    if t_colored <= 0:
        errors.append("T_colored must be > 0 for log calculation")
        return {
            "ce_cm2_c": None,
            "delta_od": None,
            "errors": errors,
            "warnings": warnings,
            "ce_skipped_reason": "invalid_t_colored"
        }
    
    if t_bleached <= 0:
        errors.append("T_bleached must be > 0")
        return {
            "ce_cm2_c": None,
            "delta_od": None,
            "errors": errors,
            "warnings": warnings,
            "ce_skipped_reason": "invalid_t_bleached"
        }
    
    # Compute ΔOD
    delta_od = np.log10(t_bleached / t_colored)
    
    # Auto-detect step type if not provided
    if step_type is None:
        step_type = "coloring" if delta_od > 0.01 else ("bleaching" if delta_od < -0.01 else "unknown")
    
    # Check if we should compute CE
    if step_type == "bleaching" and not force_compute:
        return {
            "ce_cm2_c": None,
            "delta_od": float(delta_od),
            "step_type": step_type,
            "errors": errors,
            "warnings": ["CE not computed for bleaching segments (ΔOD < 0)"],
            "ce_skipped_reason": "bleaching_segment"
        }
    
    if step_type == "unknown" and not force_compute:
        return {
            "ce_cm2_c": None,
            "delta_od": float(delta_od),
            "step_type": step_type,
            "errors": errors,
            "warnings": ["CE not computed: segment type unclear (|ΔOD| < 0.01)"],
            "ce_skipped_reason": "unknown_segment"
        }
    
    # Check charge magnitude
    if abs(q_c_cm2) < q_min_threshold:
        warnings.append(f"Charge density near zero ({abs(q_c_cm2):.2e} C/cm²)")
        return {
            "ce_cm2_c": None,
            "delta_od": float(delta_od),
            "step_type": step_type,
            "errors": errors,
            "warnings": warnings,
            "ce_skipped_reason": "near_zero_charge",
            "q_abs_c_cm2": abs(q_c_cm2)
        }
    
    # Compute CE
    ce = abs(delta_od) / abs(q_c_cm2)
    
    # QC checks for CE value
    if ce < 10:
        warnings.append(f"CE = {ce:.1f} cm²/C is low for typical EC materials (< 10)")
    elif ce > 500:
        warnings.append(f"CE = {ce:.1f} cm²/C is unusually high (> 500)")
    
    if t_bleached <= t_colored:
        warnings.append("T_bleached ≤ T_colored (unusual for coloring efficiency definition)")
    
    return {
        "ce_cm2_c": float(ce),
        "delta_od": float(delta_od),
        "log_base": 10,
        "step_type": step_type,
        "q_used_c_cm2": abs(q_c_cm2),
        "errors": errors,
        "warnings": warnings,
        "ce_skipped_reason": None
    }


# =============================================================================
# Baseline Correction (Safe)
# =============================================================================

def baseline_correct_current(
    t_s: np.ndarray, 
    i_a: np.ndarray,
    baseline_mode: Literal["none", "offset_tail", "offset_head", "linear_tail"] = "none",
    tail_fraction: float = 0.1,
    head_fraction: float = 0.1,
    linear_r2_min: float = 0.8
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply baseline correction to current data.
    
    Default is OFF (mode="none") to avoid corrupting faradaic signal.
    
    Modes:
        - "none": No correction (safest default)
        - "offset_tail": Subtract median of last tail_fraction (removes DC offset)
        - "offset_head": Subtract median of first head_fraction
        - "linear_tail": Fit linear to tail only; reject if R² < threshold
    
    Args:
        t_s: Time array
        i_a: Current array
        baseline_mode: Correction mode
        tail_fraction: Fraction of data for tail window
        head_fraction: Fraction of data for head window
        linear_r2_min: Minimum R² for linear correction
    
    Returns:
        (corrected_current, metadata)
    """
    metadata = {
        "mode": baseline_mode,
        "offset_applied": 0.0,
        "slope": None,
        "r_squared": None,
        "qc_pass": True,
        "warnings": []
    }
    
    if baseline_mode == "none":
        return i_a.copy(), metadata
    
    n = len(i_a)
    
    if baseline_mode == "offset_tail":
        n_tail = max(3, int(n * tail_fraction))
        offset = float(np.median(i_a[-n_tail:]))
        metadata["offset_applied"] = offset
        return i_a - offset, metadata
    
    elif baseline_mode == "offset_head":
        n_head = max(3, int(n * head_fraction))
        offset = float(np.median(i_a[:n_head]))
        metadata["offset_applied"] = offset
        return i_a - offset, metadata
    
    elif baseline_mode == "linear_tail":
        n_tail = max(5, int(n * tail_fraction))
        
        t_tail = t_s[-n_tail:]
        i_tail = i_a[-n_tail:]
        
        try:
            coeffs = np.polyfit(t_tail, i_tail, 1)
            slope, intercept = coeffs[0], coeffs[1]
            
            # Calculate R²
            i_fit = np.polyval(coeffs, t_tail)
            ss_res = np.sum((i_tail - i_fit) ** 2)
            ss_tot = np.sum((i_tail - np.mean(i_tail)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
            
            metadata["slope"] = float(slope)
            metadata["r_squared"] = float(r_squared)
            
            if r_squared < linear_r2_min:
                metadata["warnings"].append(
                    f"Linear tail R² = {r_squared:.3f} < {linear_r2_min}; correction not applied"
                )
                metadata["qc_pass"] = False
                return i_a.copy(), metadata
            
            # Apply correction
            baseline = np.polyval(coeffs, t_s)
            metadata["offset_applied"] = float(intercept)
            
            return i_a - baseline, metadata
            
        except Exception as e:
            metadata["warnings"].append(f"Linear fit failed: {e}")
            metadata["qc_pass"] = False
            return i_a.copy(), metadata
    
    else:
        metadata["warnings"].append(f"Unknown baseline_mode: {baseline_mode}")
        metadata["qc_pass"] = False
        return i_a.copy(), metadata


# =============================================================================
# Cycle Segmentation
# =============================================================================

def segment_cycles_by_voltage(
    v: np.ndarray, 
    t_s: np.ndarray,
    v_threshold: float = 0.1
) -> List[Dict]:
    """
    Detect cycle boundaries from voltage step changes.
    
    Args:
        v: Voltage array in V
        t_s: Time array in s
        v_threshold: Minimum voltage change to detect as step
    
    Returns:
        List of cycle dicts with start_idx, end_idx, start_time, end_time, v_start, v_end
    """
    if len(v) < 2:
        return []
    
    dv = np.diff(v)
    step_indices = np.where(np.abs(dv) > v_threshold)[0]
    
    boundaries = [0] + list(step_indices + 1) + [len(v)]
    
    cycles = []
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        if end_idx > start_idx:
            cycles.append({
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "start_time_s": float(t_s[start_idx]),
                "end_time_s": float(t_s[end_idx - 1]),
                "v_start": float(v[start_idx]),
                "v_end": float(v[end_idx - 1])
            })
    
    return cycles


def segment_cycles_by_transmittance(
    t_frac: np.ndarray, 
    t_s: np.ndarray,
    smooth_window: int = 11,
    prominence: float = 0.05
) -> List[Dict]:
    """
    Detect cycle boundaries from transmittance peaks/valleys.
    
    Uses smoothed signal to avoid noise triggers.
    Falls back to numpy-only extrema detection if SciPy unavailable.
    
    Args:
        t_frac: Transmittance array [0, 1]
        t_s: Time array in s
        smooth_window: Smoothing window (must be odd)
        prominence: Minimum prominence for peak detection
    
    Returns:
        List of cycle dicts
    """
    if len(t_frac) < smooth_window:
        return [{
            "start_idx": 0, 
            "end_idx": len(t_frac), 
            "start_time_s": float(t_s[0]), 
            "end_time_s": float(t_s[-1])
        }]
    
    # Ensure odd window
    if smooth_window % 2 == 0:
        smooth_window += 1
    
    # Smooth the signal
    t_smooth, _ = _optional_smooth(t_frac, smooth_window)
    
    # Find peaks and valleys
    if SCIPY_AVAILABLE and find_peaks is not None:
        peaks, _ = find_peaks(t_smooth, prominence=prominence)
        valleys, _ = find_peaks(-t_smooth, prominence=prominence)
    else:
        peaks, valleys = _simple_find_extrema(t_smooth, prominence)
    
    # Combine and sort extrema
    extrema = sorted(np.concatenate([peaks, valleys]))
    
    if len(extrema) < 2:
        return [{
            "start_idx": 0, 
            "end_idx": len(t_frac),
            "start_time_s": float(t_s[0]), 
            "end_time_s": float(t_s[-1])
        }]
    
    # Create half-cycles between extrema
    boundaries = [0] + list(extrema) + [len(t_frac)]
    
    cycles = []
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        if end_idx > start_idx:
            t_start = t_frac[start_idx]
            t_end = t_frac[min(end_idx - 1, len(t_frac) - 1)]
            cycles.append({
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "start_time_s": float(t_s[start_idx]),
                "end_time_s": float(t_s[min(end_idx - 1, len(t_s) - 1)]),
                "t_start": float(t_start),
                "t_end": float(t_end),
                "is_coloring": t_end < t_start
            })
    
    return cycles


def split_cycle_by_transmittance_extrema(
    t_frac: np.ndarray,
    i_a: np.ndarray, 
    t_s: np.ndarray,
    smooth_window: int = 11
) -> List[Dict]:
    """
    Split a full color+bleach cycle at the transmittance extremum.
    
    Use this when a segment spans both coloring and bleaching.
    
    Returns:
        List of 1 or 2 sub-segments with their data slices
    """
    if len(t_frac) < 5:
        return [{
            "start_idx": 0,
            "end_idx": len(t_frac),
            "t_frac": t_frac,
            "i_a": i_a,
            "t_s": t_s,
            "was_split": False
        }]
    
    # Smooth and find extremum
    t_smooth, _ = _optional_smooth(t_frac, smooth_window)
    
    # Find min and max
    idx_min = np.argmin(t_smooth)
    idx_max = np.argmax(t_smooth)
    
    # Determine if we should split
    # If extremum is not at endpoints, we have a full cycle
    split_idx = None
    if idx_min > 0 and idx_min < len(t_frac) - 1:
        # Minimum is internal -> this is the colored extremum
        split_idx = idx_min
    elif idx_max > 0 and idx_max < len(t_frac) - 1:
        # Maximum is internal -> this is the bleached extremum
        split_idx = idx_max
    
    if split_idx is None or split_idx < 3 or split_idx > len(t_frac) - 3:
        return [{
            "start_idx": 0,
            "end_idx": len(t_frac),
            "t_frac": t_frac,
            "i_a": i_a,
            "t_s": t_s,
            "was_split": False
        }]
    
    # Split into two segments
    return [
        {
            "start_idx": 0,
            "end_idx": split_idx + 1,
            "t_frac": t_frac[:split_idx + 1],
            "i_a": i_a[:split_idx + 1],
            "t_s": t_s[:split_idx + 1],
            "was_split": True
        },
        {
            "start_idx": split_idx,
            "end_idx": len(t_frac),
            "t_frac": t_frac[split_idx:],
            "i_a": i_a[split_idx:],
            "t_s": t_s[split_idx:],
            "was_split": True
        }
    ]


# =============================================================================
# Cycling Metrics (with QC)
# =============================================================================

def compute_cycling_metrics(
    cycles: List[Dict], 
    t_frac: np.ndarray, 
    i_a: np.ndarray, 
    t_s: np.ndarray,
    area_cm2: float,
    response_threshold: float = 0.9,
    validate_plateau: bool = True,
    auto_split_full_cycles: bool = True
) -> pd.DataFrame:
    """
    Compute metrics for each cycle with comprehensive QC.
    
    Args:
        cycles: List of cycle dicts from segmentation
        t_frac: Full transmittance array
        i_a: Full current array
        t_s: Full time array
        area_cm2: Electrode area
        response_threshold: Threshold for response time (0.9 = 90%)
        validate_plateau: Check if plateau reached for response time
        auto_split_full_cycles: Split color+bleach cycles for correct CE
    
    Returns:
        DataFrame with columns: cycle_num, segment_type, delta_t, ce_cm2_c, 
        q_c_cm2, response_time_s, qc_pass, reached_plateau, warnings
    """
    results = []
    
    for i, cycle in enumerate(cycles):
        start_idx = cycle["start_idx"]
        end_idx = cycle["end_idx"]
        
        if end_idx <= start_idx:
            continue
        
        t_cycle = t_frac[start_idx:end_idx]
        i_cycle = i_a[start_idx:end_idx]
        time_cycle = t_s[start_idx:end_idx]
        
        # Check if this looks like a full color+bleach cycle
        subsegments = [{"t_frac": t_cycle, "i_a": i_cycle, "t_s": time_cycle, 
                        "was_split": False, "start_idx": 0, "end_idx": len(t_cycle)}]
        
        if auto_split_full_cycles:
            subsegments = split_cycle_by_transmittance_extrema(t_cycle, i_cycle, time_cycle)
        
        for j, seg in enumerate(subsegments):
            seg_t_frac = seg["t_frac"]
            seg_i_a = seg["i_a"]
            seg_t_s = seg["t_s"]
            
            # Classify step
            step_info = classify_step(seg_t_frac, seg_i_a, seg_t_s, area_cm2)
            segment_type = step_info["step_type"]
            
            # ΔT
            t_max = float(np.max(seg_t_frac))
            t_min = float(np.min(seg_t_frac))
            delta_t = t_max - t_min
            
            # Charge
            q_result = compute_charge_density(seg_t_s, seg_i_a, area_cm2)
            q = q_result.get("q_abs_c_cm2", 0)
            q_signed = q_result.get("q_signed", 0)
            
            # CE (only for coloring)
            ce_result = compute_coloration_efficiency(
                t_max, t_min, q, 
                step_type=segment_type
            )
            ce = ce_result.get("ce_cm2_c")
            ce_skipped = ce_result.get("ce_skipped_reason")
            
            # Response time
            rt_result = compute_response_time(
                seg_t_s, seg_t_frac, 
                threshold=response_threshold,
                validate_plateau=validate_plateau
            )
            rt = rt_result.get("response_time_s")
            reached_plateau = rt_result.get("reached_plateau", False)
            rt_qc = rt_result.get("qc_pass", True)
            
            # Aggregate warnings
            all_warnings = (
                step_info.get("warnings", []) + 
                ce_result.get("warnings", []) + 
                ce_result.get("errors", []) +
                rt_result.get("warnings", [])
            )
            
            # Overall QC
            qc_pass = rt_qc and (segment_type != "unknown") and (
                segment_type == "bleaching" or ce is not None
            )
            
            # Build result row
            cycle_num = i + 1
            if len(subsegments) > 1:
                cycle_label = f"{cycle_num}.{j+1}"
            else:
                cycle_label = str(cycle_num)
            
            results.append({
                "cycle_num": i + 1,
                "cycle_label": cycle_label,
                "segment_type": segment_type,
                "delta_t": delta_t,
                "ce_cm2_c": ce,
                "ce_skipped_reason": ce_skipped,
                "q_c_cm2": q,
                "q_signed_c_cm2": q_signed,
                "response_time_s": rt,
                "reached_plateau": reached_plateau,
                "segment_was_split": seg.get("was_split", False),
                "qc_pass": qc_pass,
                "warnings": "; ".join(all_warnings) if all_warnings else None
            })
    
    df = pd.DataFrame(results)
    
    # Add summary stats as attributes (accessible via df.attrs)
    if len(df) > 0:
        df.attrs["n_cycles"] = len(df)
        df.attrs["n_valid"] = int(df["qc_pass"].sum())
        df.attrs["n_coloring"] = int((df["segment_type"] == "coloring").sum())
        df.attrs["n_bleaching"] = int((df["segment_type"] == "bleaching").sum())
        df.attrs["pct_valid"] = float(df["qc_pass"].mean() * 100)
    
    return df
