"""
Smart Window analysis module.

Handles electrochromic device metrics: ΔT, response time, coloration efficiency,
and cycle segmentation.
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any
from scipy.signal import savgol_filter, find_peaks
from scipy.integrate import cumulative_trapezoid


def align_ca_transmittance(ca_df: pd.DataFrame, tt_df: pd.DataFrame,
                           tolerance_s: float = 0.5) -> Tuple[pd.DataFrame, Dict]:
    """
    Align chronoamperometry (I-t) and transmittance (T-t) data.
    
    Uses merge_asof for nearest-neighbor time matching.
    Preserves original timebase info in metadata.
    
    Args:
        ca_df: DataFrame with columns t_s, i_a (and optional v_v)
        tt_df: DataFrame with columns t_s, t_frac
        tolerance_s: Maximum time difference for matching
    
    Returns:
        (merged_df, metadata)
    """
    # Ensure sorted
    ca_sorted = ca_df.sort_values("t_s").copy()
    tt_sorted = tt_df.sort_values("t_s").copy()
    
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
        "tolerance_s": tolerance_s
    }
    
    # Drop the extra time column
    if "t_s_tt" in merged.columns:
        merged = merged.drop(columns=["t_s_tt"])
    
    return merged, metadata


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


def compute_response_time(t_s: np.ndarray, t_frac: np.ndarray,
                          threshold: float = 0.9) -> Dict[str, Any]:
    """
    Compute response time to reach threshold of full transition.
    
    Args:
        t_s: Time array in seconds
        t_frac: Transmittance array [0, 1]
        threshold: Fraction of transition (0.9 = 90%, 0.95 = 95%)
    
    Returns:
        dict with coloring_time_s, bleaching_time_s, t_colored, t_bleached
    """
    t_start = t_frac[0]
    t_end = t_frac[-1]
    
    # Determine direction
    if t_end > t_start:
        # Bleaching (getting more transparent)
        t_colored = t_start
        t_bleached = t_end
        direction = "bleaching"
        target = t_start + (t_end - t_start) * threshold
        mask = t_frac >= target
    else:
        # Coloring (getting more opaque)
        t_bleached = t_start
        t_colored = t_end
        direction = "coloring"
        target = t_start + (t_end - t_start) * threshold  # Note: this goes down
        mask = t_frac <= target
    
    response_time = None
    if mask.any():
        idx = np.argmax(mask)
        response_time = t_s[idx] - t_s[0]
    
    return {
        "response_time_s": response_time,
        "direction": direction,
        "threshold": threshold,
        "t_colored": float(min(t_start, t_end)),
        "t_bleached": float(max(t_start, t_end)),
        "delta_t": abs(t_end - t_start)
    }


def compute_charge_density(t_s: np.ndarray, i_a: np.ndarray, 
                           area_cm2: float) -> Dict[str, Any]:
    """
    Compute charge density from current-time data.
    
    Q = ∫I dt / area
    
    Args:
        t_s: Time array in seconds
        i_a: Current array in A
        area_cm2: Electrode area in cm²
    
    Returns:
        dict with q_c_cm2 (charge density in C/cm²), q_total_c
    """
    # Integrate current over time
    if len(t_s) < 2:
        return {"q_c_cm2": 0, "q_total_c": 0, "error": "Insufficient data points"}
    
    q_cumulative = cumulative_trapezoid(i_a, t_s, initial=0)
    q_total = q_cumulative[-1]
    q_density = q_total / area_cm2
    
    return {
        "q_c_cm2": float(q_density),
        "q_total_c": float(q_total),
        "q_abs_c_cm2": float(abs(q_density)),
        "q_cumulative": q_cumulative
    }


def compute_coloration_efficiency(t_bleached: float, t_colored: float,
                                   q_c_cm2: float) -> Dict[str, Any]:
    """
    Compute coloration efficiency.
    
    CE = ΔOD / |Q|
    ΔOD = log10(Tb / Tc)  -- base-10 locked
    
    Args:
        t_bleached: Transmittance in bleached state [0, 1]
        t_colored: Transmittance in colored state [0, 1]
        q_c_cm2: Charge density in C/cm² (use absolute value internally)
    
    Returns:
        dict with ce_cm2_c (CE in cm²/C), delta_od
    """
    errors = []
    
    if t_colored <= 0:
        errors.append("T_colored must be > 0 for log calculation")
        return {"ce_cm2_c": None, "delta_od": None, "errors": errors}
    
    if t_bleached <= 0:
        errors.append("T_bleached must be > 0")
        return {"ce_cm2_c": None, "delta_od": None, "errors": errors}
    
    if t_bleached <= t_colored:
        errors.append("T_bleached should be > T_colored")
    
    delta_od = np.log10(t_bleached / t_colored)
    
    if abs(q_c_cm2) < 1e-12:
        errors.append("Charge density near zero")
        return {"ce_cm2_c": None, "delta_od": float(delta_od), "errors": errors}
    
    ce = delta_od / abs(q_c_cm2)
    
    # QC checks
    warnings = []
    if ce < 10:
        warnings.append(f"CE = {ce:.1f} cm²/C is low for typical EC materials (< 10)")
    elif ce > 500:
        warnings.append(f"CE = {ce:.1f} cm²/C is unusually high (> 500)")
    
    return {
        "ce_cm2_c": float(ce),
        "delta_od": float(delta_od),
        "log_base": 10,
        "errors": errors,
        "warnings": warnings
    }


def segment_cycles_by_voltage(v: np.ndarray, t_s: np.ndarray,
                              v_threshold: float = 0.1) -> List[Dict]:
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


def segment_cycles_by_transmittance(t_frac: np.ndarray, t_s: np.ndarray,
                                     smooth_window: int = 11,
                                     prominence: float = 0.05) -> List[Dict]:
    """
    Detect cycle boundaries from transmittance peaks/valleys.
    
    Uses smoothed signal to avoid noise triggers.
    
    Args:
        t_frac: Transmittance array [0, 1]
        t_s: Time array in s
        smooth_window: Savitzky-Golay window (must be odd)
        prominence: Minimum prominence for peak detection
    
    Returns:
        List of cycle dicts
    """
    if len(t_frac) < smooth_window:
        return [{"start_idx": 0, "end_idx": len(t_frac), 
                 "start_time_s": float(t_s[0]), "end_time_s": float(t_s[-1])}]
    
    # Ensure odd window
    if smooth_window % 2 == 0:
        smooth_window += 1
    
    # Smooth the signal
    t_smooth = savgol_filter(t_frac, smooth_window, 2)
    
    # Find peaks (bleached states) and valleys (colored states)
    peaks, _ = find_peaks(t_smooth, prominence=prominence)
    valleys, _ = find_peaks(-t_smooth, prominence=prominence)
    
    # Combine and sort extrema
    extrema = sorted(np.concatenate([peaks, valleys]))
    
    if len(extrema) < 2:
        return [{"start_idx": 0, "end_idx": len(t_frac),
                 "start_time_s": float(t_s[0]), "end_time_s": float(t_s[-1])}]
    
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


def compute_cycling_metrics(cycles: List[Dict], t_frac: np.ndarray, 
                            i_a: np.ndarray, t_s: np.ndarray,
                            area_cm2: float) -> pd.DataFrame:
    """
    Compute metrics for each cycle.
    
    Returns DataFrame with columns: cycle_num, delta_t, ce, q_c_cm2, response_time_s
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
        
        # ΔT
        t_max = np.max(t_cycle)
        t_min = np.min(t_cycle)
        delta_t = t_max - t_min
        
        # Charge
        q_result = compute_charge_density(time_cycle, i_cycle, area_cm2)
        q = q_result.get("q_abs_c_cm2", 0)
        
        # CE
        ce_result = compute_coloration_efficiency(t_max, t_min, q)
        ce = ce_result.get("ce_cm2_c")
        
        # Response time
        rt_result = compute_response_time(time_cycle, t_cycle)
        rt = rt_result.get("response_time_s")
        
        results.append({
            "cycle_num": i + 1,
            "delta_t": delta_t,
            "ce_cm2_c": ce,
            "q_c_cm2": q,
            "response_time_s": rt
        })
    
    return pd.DataFrame(results)


def baseline_correct_current(t_s: np.ndarray, i_a: np.ndarray,
                             baseline_fraction: float = 0.1) -> np.ndarray:
    """
    Apply baseline correction to current data.
    
    Uses linear fit to initial portion to remove drift.
    
    Args:
        t_s: Time array
        i_a: Current array
        baseline_fraction: Fraction of data to use for baseline
    
    Returns:
        Corrected current array
    """
    n_baseline = max(3, int(len(i_a) * baseline_fraction))
    
    # Fit baseline to initial portion
    coeffs = np.polyfit(t_s[:n_baseline], i_a[:n_baseline], 1)
    baseline = np.polyval(coeffs, t_s)
    
    return i_a - baseline
