"""
Temperature-dependent conductivity fits.

Provides Arrhenius and VFT fitting with:
- Principled model comparison (AICc/BIC, not raw R²)
- Log-space fitting to avoid high-σ bias
- Automatic temperature unit detection (°C vs K)
- Honest QC diagnostics

Scientific notes:
- Arrhenius: ln(σ) = ln(A) - Ea/(R·T)  [2 parameters: ln(A), Ea]
- VFT: ln(σ) = ln(A) - B/(T - T₀)       [3 parameters: ln(A), B, T₀]

VFT fitting is done in log-space to give equal relative weight to all
data points. Linear-scale fitting overweights high-T points where σ is large.

Model comparison uses AICc (corrected AIC for small samples) and BIC,
which penalize model complexity. Raw R² comparison is biased toward VFT.
"""
import warnings
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Literal

from utils import deps


# Physical constants
R_GAS = 8.314  # J/(mol·K) - Gas constant
KB = 8.617e-5  # eV/K - Boltzmann constant in eV
EV_TO_J_MOL = 96485  # 1 eV = 96485 J/mol


# =============================================================================
# Input Sanitization and Temperature Handling
# =============================================================================

def _sanitize_inputs(
    temps: np.ndarray,
    sigmas: np.ndarray,
    merge_duplicates: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Sanitize temperature-conductivity input arrays.
    
    Args:
        temps: Temperature array
        sigmas: Conductivity array
        merge_duplicates: If True, merge duplicate temperatures by median ln(σ)
    
    Returns:
        Tuple of (clean_temps, clean_sigmas, metadata)
        
    Metadata contains:
        - n_original: original point count
        - n_dropped_nonfinite: points dropped due to NaN/inf
        - n_dropped_nonpositive_sigma: points dropped due to σ ≤ 0
        - n_merged: points merged due to duplicate T
        - n_final: final point count
    """
    # Convert to 1D float arrays
    temps = np.atleast_1d(np.asarray(temps, dtype=np.float64)).ravel()
    sigmas = np.atleast_1d(np.asarray(sigmas, dtype=np.float64)).ravel()
    
    if len(temps) != len(sigmas):
        raise ValueError(f"Length mismatch: temps ({len(temps)}) vs sigmas ({len(sigmas)})")
    
    n_original = len(temps)
    
    # Drop non-finite values
    finite_mask = np.isfinite(temps) & np.isfinite(sigmas)
    n_dropped_nonfinite = n_original - np.sum(finite_mask)
    temps = temps[finite_mask]
    sigmas = sigmas[finite_mask]
    
    # Drop non-positive sigma
    positive_mask = sigmas > 0
    n_dropped_nonpositive = len(sigmas) - np.sum(positive_mask)
    temps = temps[positive_mask]
    sigmas = sigmas[positive_mask]
    
    n_merged = 0
    
    # Merge duplicate temperatures
    if merge_duplicates and len(temps) > 0:
        unique_temps = np.unique(temps)
        if len(unique_temps) < len(temps):
            # Group by temperature and take median of ln(sigma)
            merged_temps = []
            merged_sigmas = []
            for t in unique_temps:
                mask = temps == t
                ln_sigma_median = np.median(np.log(sigmas[mask]))
                merged_temps.append(t)
                merged_sigmas.append(np.exp(ln_sigma_median))
            
            n_merged = len(temps) - len(unique_temps)
            temps = np.array(merged_temps)
            sigmas = np.array(merged_sigmas)
    
    # Sort by temperature
    sort_idx = np.argsort(temps)
    temps = temps[sort_idx]
    sigmas = sigmas[sort_idx]
    
    metadata = {
        "n_original": n_original,
        "n_dropped_nonfinite": int(n_dropped_nonfinite),
        "n_dropped_nonpositive_sigma": int(n_dropped_nonpositive),
        "n_merged": int(n_merged),
        "n_final": len(temps)
    }
    
    return temps, sigmas, metadata


def _handle_temp_unit(
    temps: np.ndarray,
    temp_unit: Literal["K", "C", "auto"] = "auto"
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Handle temperature unit conversion and validation.
    
    Args:
        temps: Temperature array (after sanitization, but before unit conversion)
        temp_unit: "K" (Kelvin), "C" (Celsius), or "auto" (auto-detect)
    
    Returns:
        Tuple of (temps_K, metadata)
        
    Metadata contains:
        - temp_unit_input: user-specified unit
        - temp_unit_inferred: what auto-detection determined (if auto)
        - converted_to_K: whether conversion was applied
        - temp_range_K: (min, max) in Kelvin
        - warnings: list of warning messages
    """
    if len(temps) == 0:
        return temps, {
            "temp_unit_input": temp_unit,
            "temp_unit_inferred": None,
            "converted_to_K": False,
            "temp_range_K": (None, None),
            "warnings": []
        }
    
    temp_warnings = []
    converted = False
    inferred = None
    
    if temp_unit == "C":
        # Explicit Celsius
        temps_k = temps + 273.15
        converted = True
        inferred = "C"
    elif temp_unit == "K":
        # Explicit Kelvin
        temps_k = temps.copy()
        inferred = "K"
        # Sanity check: typical electrolyte measurements 200-450 K
        if np.any(temps < 100):
            temp_warnings.append(
                "WARNING: Some temperatures < 100 K with temp_unit='K'. "
                "Did you mean Celsius?"
            )
    else:
        # Auto-detect
        median_t = np.median(temps)
        max_t = np.max(temps)
        min_t = np.min(temps)

        if 150 <= min_t <= 250 and 150 <= max_t <= 250:
            temp_warnings.append(
                "AUTO-DETECT AMBIGUOUS: Temperatures in the 150–250 range could "
                "be Kelvin or Celsius. Please pass temp_unit='K' or 'C' explicitly."
            )
        
        if median_t < 150 and max_t < 250:
            # Strongly suspect Celsius (typical lab range: 20-80°C)
            temps_k = temps + 273.15
            converted = True
            inferred = "C"
            temp_warnings.append(
                f"AUTO-CONVERSION: Input temperatures (median={median_t:.1f}, "
                f"max={max_t:.1f}) appear to be Celsius. Converted to Kelvin. "
                "Set temp_unit='K' to override."
            )
        elif min_t < 0:
            # Negative values - likely Celsius
            temps_k = temps + 273.15
            converted = True
            inferred = "C"
            temp_warnings.append(
                f"AUTO-CONVERSION: Negative temperature ({min_t:.1f}) detected. "
                "Assuming Celsius and converting to Kelvin."
            )
        else:
            # Assume Kelvin
            temps_k = temps.copy()
            inferred = "K"
    
    # Validate: all temperatures must be > 0 K after conversion
    if np.any(temps_k <= 0):
        raise ValueError(
            f"Invalid temperatures: values ≤ 0 K after conversion. "
            f"Range: [{temps_k.min():.2f}, {temps_k.max():.2f}] K"
        )
    
    metadata = {
        "temp_unit_input": temp_unit,
        "temp_unit_inferred": inferred,
        "converted_to_K": converted,
        "temp_range_K": (float(temps_k.min()), float(temps_k.max())),
        "warnings": temp_warnings
    }
    
    return temps_k, metadata


# =============================================================================
# Information Criteria for Model Comparison
# =============================================================================

def _compute_rss(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    """Residual sum of squares."""
    return float(np.sum((y_obs - y_pred) ** 2))


def _compute_rmse(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_obs - y_pred) ** 2)))


def _compute_r2(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R²)."""
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - y_obs.mean()) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1 - ss_res / ss_tot)


def _compute_aic(n: int, k: int, rss: float) -> float:
    """
    Akaike Information Criterion.
    
    AIC = n * ln(RSS/n) + 2k
    
    Args:
        n: number of data points
        k: number of fitted parameters
        rss: residual sum of squares (in ln-space)
    """
    if n <= 0:
        return float('inf')
    rss_safe = max(float(rss), np.finfo(float).tiny)
    return n * np.log(rss_safe / n) + 2 * k


def _compute_aicc(n: int, k: int, rss: float) -> float:
    """
    Corrected AIC for small sample sizes.
    
    AICc = AIC + (2k(k+1)) / (n - k - 1)
    
    Returns inf if n <= k + 1 (insufficient data).
    """
    if n <= k + 1:
        return float('inf')
    aic = _compute_aic(n, k, rss)
    if not np.isfinite(aic):
        return float('inf')
    correction = (2 * k * (k + 1)) / (n - k - 1)
    return aic + correction


def _compute_bic(n: int, k: int, rss: float) -> float:
    """
    Bayesian Information Criterion.
    
    BIC = n * ln(RSS/n) + k * ln(n)
    """
    if n <= 0:
        return float('inf')
    rss_safe = max(float(rss), np.finfo(float).tiny)
    return n * np.log(rss_safe / n) + k * np.log(n)


def _interpret_delta_ic(delta: float) -> str:
    """
    Interpret ΔAICc or ΔBIC for model selection.
    
    Following Burnham & Anderson (2002):
    - 0-2: essentially equivalent
    - 2-4: weak support for better model
    - 4-7: moderate support
    - >10: strong support
    """
    delta = abs(delta)
    if delta <= 2:
        return "inconclusive"
    elif delta <= 4:
        return "weak"
    elif delta <= 7:
        return "moderate"
    else:
        return "strong"


# =============================================================================
# QC Flags
# =============================================================================

def _generate_qc_flags(
    n_points: int,
    k_params: int,
    temp_range_k: Tuple[float, float],
    temp_converted: bool,
    t0_k: Optional[float] = None,
    model_type: str = "arrhenius"
) -> Tuple[bool, List[str]]:
    """
    Generate QC flags based on data quality and fit validity.
    
    Returns:
        Tuple of (qc_pass, qc_flags)
    """
    qc_flags = []
    
    # Check minimum points
    if n_points <= k_params + 1:
        qc_flags.append("too_few_points_for_model")
    
    # Check temperature span
    if temp_range_k[0] is not None and temp_range_k[1] is not None:
        span = temp_range_k[1] - temp_range_k[0]
        if span < 30:
            qc_flags.append("narrow_temperature_span")
    
    # Temperature unit warning
    if temp_converted:
        qc_flags.append("temp_unit_auto_converted")
    
    # VFT-specific checks
    if model_type == "vft" and t0_k is not None and temp_range_k[0] is not None:
        t_min = temp_range_k[0]
        if t_min - t0_k < 10:
            qc_flags.append("T0_close_to_Tmin")
        if n_points <= 4:
            qc_flags.append("vft_overfit_risk")
    
    # Determine overall pass
    critical_flags = {"too_few_points_for_model", "T0_close_to_Tmin"}
    qc_pass = not any(f in critical_flags for f in qc_flags)
    
    return qc_pass, qc_flags


# =============================================================================
# Arrhenius Fit (numpy-only)
# =============================================================================

def arrhenius_fit(
    temps_k: np.ndarray,
    sigmas: np.ndarray,
    temp_unit: Literal["K", "C", "auto"] = "auto"
) -> Dict[str, Any]:
    """
    Fit Arrhenius equation to conductivity data.
    
    ln(σ) = ln(A) - Ea/(R·T)
    
    This is a linear fit in (1/T, ln(σ)) space using numpy only.
    
    Args:
        temps_k: Temperature array (Kelvin, Celsius, or auto-detect)
        sigmas: Conductivity array in S/cm
        temp_unit: "K", "C", or "auto" (default: auto-detect)
    
    Returns:
        dict with:
        - success: bool
        - ln_A, A_s_cm: pre-exponential factor
        - ea_kj_mol, ea_ev: activation energy
        - r_squared: R² on ln(σ) scale (primary metric)
        - r2_ln: same as r_squared (explicit name)
        - rmse_ln: RMSE in ln(σ) space
        - aic, aicc, bic: information criteria (k=2)
        - slope: fit slope for compatibility
        - fit_type, equation: metadata
        - temp_metadata: temperature handling info
        - sanitize_metadata: input sanitization info
        - qc_pass, qc_flags: quality control
        - warnings: list of warning messages
    """
    result_warnings = []
    
    # Sanitize inputs
    try:
        temps_clean, sigmas_clean, sanitize_meta = _sanitize_inputs(
            temps_k, sigmas, merge_duplicates=True
        )
    except Exception as e:
        return {"success": False, "error": f"Input sanitization failed: {e}"}
    
    if sanitize_meta["n_dropped_nonfinite"] > 0:
        result_warnings.append(
            f"Dropped {sanitize_meta['n_dropped_nonfinite']} non-finite points"
        )
    if sanitize_meta["n_dropped_nonpositive_sigma"] > 0:
        result_warnings.append(
            f"Dropped {sanitize_meta['n_dropped_nonpositive_sigma']} points with σ ≤ 0"
        )
    if sanitize_meta["n_merged"] > 0:
        result_warnings.append(
            f"Merged {sanitize_meta['n_merged']} duplicate temperatures (median ln(σ))"
        )
    
    n = len(temps_clean)
    if n < 2:
        return {
            "success": False,
            "error": f"Need at least 2 data points (got {n} after sanitization)",
            "sanitize_metadata": sanitize_meta
        }
    
    # Handle temperature units
    try:
        temps_k_final, temp_meta = _handle_temp_unit(temps_clean, temp_unit)
    except Exception as e:
        return {"success": False, "error": str(e), "sanitize_metadata": sanitize_meta}
    
    result_warnings.extend(temp_meta["warnings"])
    
    # Fit: ln(σ) = intercept + slope * (1/T)
    inv_T = 1.0 / temps_k_final
    ln_sigma = np.log(sigmas_clean)
    
    try:
        slope, intercept = np.polyfit(inv_T, ln_sigma, 1)
    except Exception as e:
        return {
            "success": False,
            "error": f"Linear fit failed: {e}",
            "sanitize_metadata": sanitize_meta,
            "temp_metadata": temp_meta
        }
    
    # Predict and compute metrics
    ln_sigma_pred = slope * inv_T + intercept
    
    r2_ln = _compute_r2(ln_sigma, ln_sigma_pred)
    rmse_ln = _compute_rmse(ln_sigma, ln_sigma_pred)
    rss = _compute_rss(ln_sigma, ln_sigma_pred)
    
    # Information criteria (k=2: slope and intercept)
    k = 2
    aic = _compute_aic(n, k, rss)
    aicc = _compute_aicc(n, k, rss)
    bic = _compute_bic(n, k, rss)
    
    # Activation energy: Ea = -slope * R
    ea_j_mol = -slope * R_GAS
    ea_kj_mol = ea_j_mol / 1000
    ea_ev = ea_j_mol / EV_TO_J_MOL
    
    # Pre-exponential factor
    A = np.exp(intercept)
    
    # QC flags
    qc_pass, qc_flags = _generate_qc_flags(
        n_points=n,
        k_params=k,
        temp_range_k=temp_meta["temp_range_K"],
        temp_converted=temp_meta["converted_to_K"],
        model_type="arrhenius"
    )
    
    return {
        "success": True,
        # Primary results (backward compatible)
        "ln_A": float(intercept),
        "A_s_cm": float(A),
        "ea_kj_mol": float(ea_kj_mol),
        "ea_ev": float(ea_ev),
        "r_squared": float(r2_ln),  # backward compatible key
        "slope": float(slope),
        "fit_type": "Arrhenius",
        "equation": "ln(σ) = ln(A) - Ea/(R·T)",
        # New metrics
        "r2_ln": float(r2_ln),
        "rmse_ln": float(rmse_ln),
        "aic": float(aic),
        "aicc": float(aicc),
        "bic": float(bic),
        "n_points": n,
        "k_params": k,
        # Metadata
        "temp_metadata": temp_meta,
        "sanitize_metadata": sanitize_meta,
        # QC
        "qc_pass": qc_pass,
        "qc_flags": qc_flags,
        "warnings": result_warnings
    }


# =============================================================================
# VFT Fit (requires SciPy)
# =============================================================================

def vft_fit(
    temps_k: np.ndarray,
    sigmas: np.ndarray,
    T0_init: Optional[float] = None,
    temp_unit: Literal["K", "C", "auto"] = "auto",
    vft_prefactor: Literal["standard", "T^-1", "T^-0.5"] = "standard"
) -> Dict[str, Any]:
    """
    Fit VFT (Vogel-Fulcher-Tammann) equation to conductivity data.
    
    Fitting is done in log-space to avoid overweighting high-σ points:
    
    Standard:  ln(σ) = ln(A) - B/(T - T₀)
    T^-1:      ln(σ) = ln(A) - ln(T) - B/(T - T₀)
    T^-0.5:    ln(σ) = ln(A) - 0.5*ln(T) - B/(T - T₀)
    
    Note: VFT does NOT yield a single constant Ea.
    For Ea, use apparent_ea_vft() at a specific temperature.
    
    Args:
        temps_k: Temperature array (Kelvin, Celsius, or auto-detect)
        sigmas: Conductivity array in S/cm
        T0_init: Initial guess for T0 (default: Tmin - 50)
        temp_unit: "K", "C", or "auto" (default: auto-detect)
        vft_prefactor: Prefactor model - "standard", "T^-1", or "T^-0.5"
    
    Returns:
        dict with A, B_K, T0_K, r_squared, metrics, QC flags
    """
    # Check SciPy availability (optional dependency)
    try:
        scipy_ok = deps.require_scipy(impact="VFT temperature fitting disabled.")
    except Exception as e:
        dep = deps.check_dependency("scipy")
        error_msg = dep.get("error") or f"SciPy check failed: {e}"
        return {
            "success": False,
            "error": error_msg,
            "scipy_required": True,
            "scipy_status": dep.get("category", "unknown"),
        }
    if not scipy_ok:
        dep = deps.check_dependency("scipy")
        error_msg = dep.get("error") or (
            "VFT fitting requires SciPy (pip install scipy). "
            "Arrhenius fit (numpy-only) is still available."
        )
        return {
            "success": False,
            "error": error_msg,
            "scipy_required": True,
            "scipy_status": dep.get("category", "unknown"),
        }
    try:
        from scipy.optimize import curve_fit
    except Exception as e:
        dep = deps.check_dependency("scipy")
        error_msg = dep.get("error") or f"SciPy import failed: {e}"
        return {
            "success": False,
            "error": error_msg,
            "scipy_required": True,
            "scipy_status": dep.get("category", "unknown"),
        }
    
    result_warnings = []
    
    # Sanitize inputs
    try:
        temps_clean, sigmas_clean, sanitize_meta = _sanitize_inputs(
            temps_k, sigmas, merge_duplicates=True
        )
    except Exception as e:
        return {"success": False, "error": f"Input sanitization failed: {e}"}
    
    if sanitize_meta["n_dropped_nonfinite"] > 0:
        result_warnings.append(
            f"Dropped {sanitize_meta['n_dropped_nonfinite']} non-finite points"
        )
    if sanitize_meta["n_dropped_nonpositive_sigma"] > 0:
        result_warnings.append(
            f"Dropped {sanitize_meta['n_dropped_nonpositive_sigma']} points with σ ≤ 0"
        )
    if sanitize_meta["n_merged"] > 0:
        result_warnings.append(
            f"Merged {sanitize_meta['n_merged']} duplicate temperatures (median ln(σ))"
        )
    
    n = len(temps_clean)
    if n < 3:
        return {
            "success": False,
            "error": f"VFT needs at least 3 data points (got {n} after sanitization)",
            "sanitize_metadata": sanitize_meta
        }
    
    # Handle temperature units
    try:
        temps_k_final, temp_meta = _handle_temp_unit(temps_clean, temp_unit)
    except Exception as e:
        return {"success": False, "error": str(e), "sanitize_metadata": sanitize_meta}
    
    result_warnings.extend(temp_meta["warnings"])

    # Tighten validity: enforce finite, positive T and σ after conversion
    valid_mask = (
        np.isfinite(temps_k_final)
        & np.isfinite(sigmas_clean)
        & (temps_k_final > 0)
        & (sigmas_clean > 0)
    )
    if not np.all(valid_mask):
        dropped = int(np.sum(~valid_mask))
        temps_k_final = temps_k_final[valid_mask]
        sigmas_clean = sigmas_clean[valid_mask]
        result_warnings.append(
            f"Dropped {dropped} invalid points after temperature conversion"
        )
        n = len(temps_k_final)
        if n < 3:
            return {
                "success": False,
                "error": f"VFT needs at least 3 data points (got {n} after filtering)",
                "sanitize_metadata": sanitize_meta,
                "temp_metadata": temp_meta,
                "warnings": result_warnings
            }
        temp_meta["temp_range_K"] = (float(temps_k_final.min()), float(temps_k_final.max()))

    T_min = temps_k_final.min()

    # Observed ln(sigma)
    ln_sigma_obs = np.log(sigmas_clean)
    
    # Define VFT model in log-space based on prefactor choice
    if vft_prefactor == "standard":
        # ln(σ) = lnA - B/(T - T0)
        def vft_ln_model(T, lnA, B, T0):
            return lnA - B / (T - T0)
        k = 3  # parameters: lnA, B, T0
        equation = "ln(σ) = ln(A) - B/(T - T₀)"
    elif vft_prefactor == "T^-1":
        # ln(σ) = lnA - ln(T) - B/(T - T0)
        def vft_ln_model(T, lnA, B, T0):
            return lnA - np.log(T) - B / (T - T0)
        k = 3
        equation = "ln(σ) = ln(A) - ln(T) - B/(T - T₀)"
    elif vft_prefactor == "T^-0.5":
        # ln(σ) = lnA - 0.5*ln(T) - B/(T - T0)
        def vft_ln_model(T, lnA, B, T0):
            return lnA - 0.5 * np.log(T) - B / (T - T0)
        k = 3
        equation = "ln(σ) = ln(A) - 0.5·ln(T) - B/(T - T₀)"
    else:
        return {"success": False, "error": f"Unknown vft_prefactor: {vft_prefactor}"}
    
    # Initial guesses
    if T0_init is None:
        T0_init_guess = T_min - 50  # Typical: T0 is 30-50 K below Tg
    else:
        T0_init_guess = T0_init

    ln_sigma_min = float(ln_sigma_obs.min())
    ln_sigma_max = float(ln_sigma_obs.max())
    ln_sigma_span = ln_sigma_max - ln_sigma_min
    margin = max(5.0, 0.25 * ln_sigma_span)

    lnA_lower = min(-20.0, ln_sigma_min - margin)
    lnA_upper = max(20.0, ln_sigma_max + margin)

    lnA_init = ln_sigma_max + 2.0  # A should be larger than max observed σ
    lnA_init = float(np.clip(lnA_init, lnA_lower, lnA_upper))

    B_lower = 1.0
    B_upper = 50000.0
    B_init = float(np.clip(1000.0, B_lower, B_upper))

    T0_lower = 0.0
    tiny_margin = 1e-6
    T0_upper = min(max(50.0, T_min - 5.0), T_min - tiny_margin)
    if T0_upper <= T0_lower:
        return {
            "success": False,
            "error": (
                f"Invalid T0 bounds: T0_upper={T0_upper:.4g} ≤ T0_lower={T0_lower:.4g}. "
                "Insufficient temperature span or invalid temperatures."
            ),
            "sanitize_metadata": sanitize_meta,
            "temp_metadata": temp_meta,
            "warnings": result_warnings
        }
    T0_init_guess = float(np.clip(T0_init_guess, T0_lower, T0_upper))
    
    try:
        bounds = (
            [lnA_lower, B_lower, T0_lower],
            [lnA_upper, B_upper, T0_upper]
        )
        
        popt, pcov = curve_fit(
            vft_ln_model,
            temps_k_final,
            ln_sigma_obs,
            p0=[lnA_init, B_init, T0_init_guess],
            bounds=bounds,
            maxfev=10000
        )
        
        lnA, B, T0 = popt
        A = np.exp(lnA)

        def _hit_bound(value: float, bound: float) -> bool:
            tol = 1e-8 + 1e-6 * max(1.0, abs(bound))
            return abs(value - bound) <= tol

        bounds_hit = []
        if _hit_bound(lnA, lnA_lower):
            bounds_hit.append("lnA at lower bound")
        elif _hit_bound(lnA, lnA_upper):
            bounds_hit.append("lnA at upper bound")
        if _hit_bound(B, B_lower):
            bounds_hit.append("B at lower bound")
        elif _hit_bound(B, B_upper):
            bounds_hit.append("B at upper bound")
        if _hit_bound(T0, T0_lower):
            bounds_hit.append("T0 at lower bound")
        elif _hit_bound(T0, T0_upper):
            bounds_hit.append("T0 at upper bound")
        if bounds_hit:
            result_warnings.append(
                "Fit hit parameter bounds: " + ", ".join(bounds_hit)
            )
        
        # Predictions in ln-space
        ln_sigma_pred = vft_ln_model(temps_k_final, lnA, B, T0)
        
        # Metrics in ln-space (primary)
        r2_ln = _compute_r2(ln_sigma_obs, ln_sigma_pred)
        rmse_ln = _compute_rmse(ln_sigma_obs, ln_sigma_pred)
        rss = _compute_rss(ln_sigma_obs, ln_sigma_pred)
        
        # Information criteria
        aic = _compute_aic(n, k, rss)
        aicc = _compute_aicc(n, k, rss)
        bic = _compute_bic(n, k, rss)
        
        # Also compute R² in linear σ space for reference (but document it's misleading)
        sigma_pred = np.exp(ln_sigma_pred)
        r2_sigma = _compute_r2(sigmas_clean, sigma_pred)
        
        # QC flags
        qc_pass, qc_flags = _generate_qc_flags(
            n_points=n,
            k_params=k,
            temp_range_k=temp_meta["temp_range_K"],
            temp_converted=temp_meta["converted_to_K"],
            t0_k=T0,
            model_type="vft"
        )
        
        # Warn about AICc validity
        if n <= k + 1:
            result_warnings.append(
                f"AICc invalid: n={n} ≤ k+1={k+1}. Use BIC or get more data."
            )
        
        return {
            "success": True,
            # Primary results (backward compatible)
            "A_s_cm": float(A),
            "B_K": float(B),
            "T0_K": float(T0),
            "r_squared": float(r2_ln),  # backward compatible, now in ln-space
            "fit_type": "VFT",
            "equation": equation,
            "note": "VFT does NOT yield constant Ea. Use apparent_ea_vft() for Ea(T).",
            # New detailed results
            "ln_A": float(lnA),
            "r2_ln": float(r2_ln),
            "r2_sigma": float(r2_sigma),  # for reference only
            "rmse_ln": float(rmse_ln),
            "aic": float(aic),
            "aicc": float(aicc),
            "bic": float(bic),
            "n_points": n,
            "k_params": k,
            "vft_prefactor": vft_prefactor,
            "bounds_used": {
                "lnA": (float(lnA_lower), float(lnA_upper)),
                "B": (float(B_lower), float(B_upper)),
                "T0": (float(T0_lower), float(T0_upper)),
            },
            # Metadata
            "temp_metadata": temp_meta,
            "sanitize_metadata": sanitize_meta,
            # QC
            "qc_pass": qc_pass,
            "qc_flags": qc_flags,
            "warnings": result_warnings,
            # Note about r2_sigma
            "r2_sigma_note": "R² in linear σ space is misleading due to high-T overweighting. Use r2_ln."
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"VFT curve_fit failed: {e}",
            "sanitize_metadata": sanitize_meta,
            "temp_metadata": temp_meta,
            "warnings": result_warnings
        }


def apparent_ea_vft(
    B_K: float,
    T0_K: float,
    T_ref_K: float,
    vft_prefactor: Literal["standard", "T^-1", "T^-0.5"] = "standard"
) -> Dict[str, Any]:
    """
    Compute apparent activation energy at a reference temperature for VFT.

    Definition:
        Ea = -R * d ln(σ) / d(1/T) = R * T^2 * d ln(σ) / dT

    Prefactor-aware forms:
        standard: ln(σ) = ln(A) - B/(T - T0)
        T^-1:     ln(σ) = ln(A) - ln(T) - B/(T - T0)
        T^-0.5:   ln(σ) = ln(A) - 0.5*ln(T) - B/(T - T0)

    This is temperature-dependent! Always report with the temperature.

    Args:
        B_K: VFT B parameter in Kelvin
        T0_K: VFT T0 parameter in Kelvin
        T_ref_K: Reference temperature in Kelvin
        vft_prefactor: Prefactor model ("standard", "T^-1", "T^-0.5")

    Returns:
        dict with ea_apparent_kj_mol, ea_apparent_ev, at_temp_K
    """
    if T_ref_K <= T0_K:
        return {"error": f"T_ref ({T_ref_K} K) must be > T0 ({T0_K} K)"}

    base = R_GAS * B_K * (T_ref_K ** 2) / ((T_ref_K - T0_K) ** 2)
    if vft_prefactor == "standard":
        ea_j_mol = base
    elif vft_prefactor == "T^-1":
        ea_j_mol = base - R_GAS * T_ref_K
    elif vft_prefactor == "T^-0.5":
        ea_j_mol = base - 0.5 * R_GAS * T_ref_K
    else:
        return {"error": f"Unknown vft_prefactor: {vft_prefactor}"}

    ea_kj_mol = ea_j_mol / 1000
    ea_ev = ea_j_mol / EV_TO_J_MOL
    
    return {
        "ea_apparent_kj_mol": float(ea_kj_mol),
        "ea_apparent_ev": float(ea_ev),
        "at_temp_K": float(T_ref_K),
        "at_temp_C": float(T_ref_K - 273.15),
        "note": "This Ea is temperature-dependent (VFT behavior)",
        "vft_prefactor": vft_prefactor
    }


# =============================================================================
# Model Comparison
# =============================================================================

def compare_fits(
    temps_k: np.ndarray,
    sigmas: np.ndarray,
    temp_unit: Literal["K", "C", "auto"] = "auto",
    vft_prefactor: Literal["standard", "T^-1", "T^-0.5"] = "standard"
) -> Dict[str, Any]:
    """
    Fit both Arrhenius and VFT and compare using information criteria.
    
    Model comparison is based on AICc (preferred for small n) and BIC,
    NOT raw R² which is biased toward the model with more parameters.
    
    Args:
        temps_k: Temperature array
        sigmas: Conductivity array in S/cm
        temp_unit: "K", "C", or "auto"
        vft_prefactor: VFT prefactor model
    
    Returns:
        dict with:
        - arrhenius: full Arrhenius fit result
        - vft: full VFT fit result
        - best_model: "Arrhenius", "VFT", or "inconclusive"
        - delta_aicc: AICc(VFT) - AICc(Arrhenius)
        - delta_bic: BIC(VFT) - BIC(Arrhenius)
        - support_strength: "inconclusive", "weak", "moderate", "strong"
        - recommendation: human-readable recommendation string
        - explanation: detailed explanation for UI display
    """
    scipy_available = deps.is_available("scipy")
    arr_result = arrhenius_fit(temps_k, sigmas, temp_unit=temp_unit)
    vft_result = vft_fit(temps_k, sigmas, temp_unit=temp_unit, vft_prefactor=vft_prefactor)
    
    comparison = {
        "arrhenius": arr_result,
        "vft": vft_result
    }
    
    # Initialize defaults
    best_model = "inconclusive"
    delta_aicc = None
    delta_bic = None
    support_strength = "inconclusive"
    recommendation = ""
    explanation = ""
    
    if arr_result.get("success") and vft_result.get("success"):
        # Both fits succeeded - compare using information criteria
        arr_aicc = arr_result.get("aicc", float('inf'))
        vft_aicc = vft_result.get("aicc", float('inf'))
        arr_bic = arr_result.get("bic", float('inf'))
        vft_bic = vft_result.get("bic", float('inf'))
        
        # ΔIC = VFT - Arrhenius (negative means VFT is better)
        delta_aicc = vft_aicc - arr_aicc
        delta_bic = vft_bic - arr_bic
        
        # Check if AICc is valid
        aicc_valid = np.isfinite(arr_aicc) and np.isfinite(vft_aicc)
        bic_valid = np.isfinite(arr_bic) and np.isfinite(vft_bic)
        
        if aicc_valid:
            support_strength = _interpret_delta_ic(delta_aicc)
            
            if abs(delta_aicc) <= 2:
                best_model = "inconclusive"
                recommendation = "Models are statistically equivalent"
                explanation = (
                    f"ΔAICc = {delta_aicc:.1f} (|Δ| ≤ 2 means inconclusive). "
                    "Both Arrhenius and VFT fit the data equally well. "
                    "Arrhenius is simpler; use VFT only if there's physical reason."
                )
            elif delta_aicc < -2:
                # VFT is better (lower AICc)
                best_model = "VFT"
                recommendation = f"VFT preferred ({support_strength} support)"
                explanation = (
                    f"ΔAICc = {delta_aicc:.1f}: VFT has lower AICc. "
                    f"This suggests {support_strength} evidence for VFT over Arrhenius. "
                    "Indicates glassy dynamics / fragile behavior."
                )
            else:
                # Arrhenius is better (lower AICc)
                best_model = "Arrhenius"
                recommendation = f"Arrhenius preferred ({support_strength} support)"
                explanation = (
                    f"ΔAICc = {delta_aicc:.1f}: Arrhenius has lower AICc. "
                    f"This suggests {support_strength} evidence that the simpler "
                    "Arrhenius model is sufficient. May indicate crystalline SPE."
                )
        elif bic_valid:
            # Fall back to BIC
            support_strength = _interpret_delta_ic(delta_bic)
            if abs(delta_bic) <= 2:
                best_model = "inconclusive"
                recommendation = "Models are statistically equivalent (BIC)"
            elif delta_bic < -2:
                best_model = "VFT"
                recommendation = f"VFT preferred (BIC: {support_strength} support)"
            else:
                best_model = "Arrhenius"
                recommendation = f"Arrhenius preferred (BIC: {support_strength} support)"
            explanation = f"AICc invalid (too few points); using BIC. ΔBIC = {delta_bic:.1f}"
        else:
            recommendation = "Cannot compare: information criteria invalid"
            explanation = "Insufficient data for reliable model comparison."
        
        # Add R² for reference (but note it shouldn't drive decisions)
        comparison["r2_comparison_note"] = (
            "R² values are shown for reference only. Model selection is based on "
            "AICc/BIC which penalize complexity. VFT naturally has higher R² due "
            "to having more parameters."
        )
        
    elif arr_result.get("success"):
        best_model = "Arrhenius"
        recommendation = "Only Arrhenius fit succeeded"
        if not scipy_available or vft_result.get("scipy_required"):
            explanation = "VFT fitting requires SciPy (not installed or broken)."
        else:
            explanation = f"VFT fit failed: {vft_result.get('error', 'unknown error')}"
            
    elif vft_result.get("success"):
        best_model = "VFT"
        recommendation = "Only VFT fit succeeded"
        explanation = f"Arrhenius fit failed: {arr_result.get('error', 'unknown error')}"
        
    else:
        recommendation = "Neither fit succeeded"
        explanation = (
            f"Arrhenius error: {arr_result.get('error', 'unknown')}. "
            f"VFT error: {vft_result.get('error', 'unknown')}"
        )
    
    comparison.update({
        "best_model": best_model,
        "delta_aicc": float(delta_aicc) if delta_aicc is not None else None,
        "delta_bic": float(delta_bic) if delta_bic is not None else None,
        "support_strength": support_strength,
        "recommendation": recommendation,
        "explanation": explanation,
        # Backward compatibility
        "scipy_available": scipy_available
    })
    
    return comparison


# =============================================================================
# Curve Generation for Plotting
# =============================================================================

def generate_fit_curves(
    temps_k: np.ndarray,
    arr_result: Dict,
    vft_result: Dict,
    n_points: int = 100
) -> Dict[str, np.ndarray]:
    """
    Generate smooth curves for plotting fitted models.
    
    Returns arrays for plotting: T_fit, sigma_arr_fit, sigma_vft_fit
    """
    # Handle temperature range - use the metadata if available
    if arr_result.get("success") and "temp_metadata" in arr_result:
        t_range = arr_result["temp_metadata"].get("temp_range_K")
        if t_range and t_range[0] is not None:
            T_min, T_max = t_range
        else:
            T_min, T_max = temps_k.min(), temps_k.max()
    else:
        T_min, T_max = temps_k.min(), temps_k.max()
    
    T_range = np.linspace(T_min, T_max, n_points)
    
    result = {"T_K": T_range}
    
    # Arrhenius curve
    if arr_result.get("success"):
        ln_A = arr_result["ln_A"]
        slope = arr_result["slope"]
        ln_sigma_arr = ln_A + slope * (1.0 / T_range)
        result["sigma_arrhenius"] = np.exp(ln_sigma_arr)
    
    # VFT curve
    if vft_result.get("success"):
        lnA = vft_result.get("ln_A", np.log(vft_result["A_s_cm"]))
        B = vft_result["B_K"]
        T0 = vft_result["T0_K"]
        prefactor = vft_result.get("vft_prefactor", "standard")
        
        # Avoid division by zero near T0
        safe_mask = T_range > T0 + 1
        T_safe = T_range[safe_mask]
        
        if prefactor == "standard":
            ln_sigma_vft = lnA - B / (T_safe - T0)
        elif prefactor == "T^-1":
            ln_sigma_vft = lnA - np.log(T_safe) - B / (T_safe - T0)
        elif prefactor == "T^-0.5":
            ln_sigma_vft = lnA - 0.5 * np.log(T_safe) - B / (T_safe - T0)
        else:
            ln_sigma_vft = lnA - B / (T_safe - T0)
        
        # Create full array with NaN for unsafe region
        sigma_vft_full = np.full(n_points, np.nan)
        sigma_vft_full[safe_mask] = np.exp(ln_sigma_vft)
        result["sigma_vft"] = sigma_vft_full
    
    return result
