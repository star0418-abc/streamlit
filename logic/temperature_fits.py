"""
Temperature-dependent conductivity fits.

Provides Arrhenius and VFT fitting with proper Ea handling.
"""
import numpy as np
from typing import Dict, List, Any, Optional
from scipy.optimize import curve_fit


# Physical constants
R_GAS = 8.314  # J/(mol·K) - Gas constant
KB = 8.617e-5  # eV/K - Boltzmann constant in eV


def arrhenius_fit(temps_k: np.ndarray, sigmas: np.ndarray) -> Dict[str, Any]:
    """
    Fit Arrhenius equation to conductivity data.
    
    ln(σ) = ln(A) - Ea/(R·T)
    
    Args:
        temps_k: Temperature array in Kelvin
        sigmas: Conductivity array in S/cm
    
    Returns:
        dict with ln_A, ea_kj_mol, ea_ev, r_squared, fit_params
    """
    if len(temps_k) < 2:
        return {"success": False, "error": "Need at least 2 data points"}
    
    if (sigmas <= 0).any():
        return {"success": False, "error": "All conductivities must be positive"}
    
    inv_T = 1.0 / temps_k
    ln_sigma = np.log(sigmas)
    
    # Linear fit: ln(σ) = intercept + slope * (1/T)
    try:
        slope, intercept = np.polyfit(inv_T, ln_sigma, 1)
    except Exception as e:
        return {"success": False, "error": str(e)}
    
    # Ea = -slope * R
    ea_j_mol = -slope * R_GAS
    ea_kj_mol = ea_j_mol / 1000
    ea_ev = ea_j_mol / 96485  # 1 eV = 96485 J/mol
    
    # R² calculation
    ln_sigma_fit = slope * inv_T + intercept
    ss_res = np.sum((ln_sigma - ln_sigma_fit) ** 2)
    ss_tot = np.sum((ln_sigma - ln_sigma.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    # Pre-exponential factor
    A = np.exp(intercept)
    
    return {
        "success": True,
        "ln_A": float(intercept),
        "A_s_cm": float(A),
        "ea_kj_mol": float(ea_kj_mol),
        "ea_ev": float(ea_ev),
        "r_squared": float(r_squared),
        "slope": float(slope),
        "fit_type": "Arrhenius",
        "equation": "ln(σ) = ln(A) - Ea/(R·T)"
    }


def vft_fit(temps_k: np.ndarray, sigmas: np.ndarray, 
            T0_init: Optional[float] = None) -> Dict[str, Any]:
    """
    Fit VFT (Vogel-Fulcher-Tammann) equation to conductivity data.
    
    σ = A × exp(-B / (T - T0))
    
    Note: VFT does NOT yield a single constant Ea.
    For Ea, use apparent_ea_vft() at a specific temperature.
    
    Args:
        temps_k: Temperature array in Kelvin
        sigmas: Conductivity array in S/cm
        T0_init: Initial guess for T0 (default: Tmin - 50)
    
    Returns:
        dict with A, B_K, T0_K, r_squared
    """
    if len(temps_k) < 3:
        return {"success": False, "error": "VFT needs at least 3 data points"}
    
    if (sigmas <= 0).any():
        return {"success": False, "error": "All conductivities must be positive"}
    
    def vft_func(T, A, B, T0):
        return A * np.exp(-B / (T - T0))
    
    # Initial guesses
    T_min = temps_k.min()
    if T0_init is None:
        T0_init = T_min - 50  # Typical: T0 is 30-50 K below Tg
    
    A_init = sigmas.max()
    B_init = 1000  # Typical B values: 500-2000 K
    
    try:
        # Bounds to ensure physical validity
        bounds = (
            [1e-10, 100, 50],           # Lower: A > 0, B > 100, T0 > 50 K
            [1e3, 5000, T_min - 5]       # Upper: reasonable limits, T0 < Tmin - 5
        )
        
        popt, pcov = curve_fit(
            vft_func, temps_k, sigmas,
            p0=[A_init, B_init, T0_init],
            bounds=bounds,
            maxfev=10000
        )
        
        A, B, T0 = popt
        
        # Calculate R²
        sigma_fit = vft_func(temps_k, A, B, T0)
        ss_res = np.sum((sigmas - sigma_fit) ** 2)
        ss_tot = np.sum((sigmas - sigmas.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        return {
            "success": True,
            "A_s_cm": float(A),
            "B_K": float(B),
            "T0_K": float(T0),
            "r_squared": float(r_squared),
            "fit_type": "VFT",
            "equation": "σ = A × exp(-B / (T - T0))",
            "note": "VFT does NOT yield constant Ea. Use apparent_ea_vft() for Ea(T)."
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def apparent_ea_vft(B_K: float, T0_K: float, T_ref_K: float) -> Dict[str, float]:
    """
    Compute apparent activation energy at a reference temperature for VFT.
    
    Ea_app(T) = R × B × T² / (T - T0)²
    
    This is temperature-dependent! Always report with the temperature.
    
    Args:
        B_K: VFT B parameter in Kelvin
        T0_K: VFT T0 parameter in Kelvin
        T_ref_K: Reference temperature in Kelvin
    
    Returns:
        dict with ea_apparent_kj_mol, ea_apparent_ev, at_temp_K
    """
    if T_ref_K <= T0_K:
        return {"error": f"T_ref ({T_ref_K} K) must be > T0 ({T0_K} K)"}
    
    ea_j_mol = R_GAS * B_K * (T_ref_K ** 2) / ((T_ref_K - T0_K) ** 2)
    ea_kj_mol = ea_j_mol / 1000
    ea_ev = ea_j_mol / 96485
    
    return {
        "ea_apparent_kj_mol": float(ea_kj_mol),
        "ea_apparent_ev": float(ea_ev),
        "at_temp_K": float(T_ref_K),
        "at_temp_C": float(T_ref_K - 273.15),
        "note": "This Ea is temperature-dependent (VFT behavior)"
    }


def compare_fits(temps_k: np.ndarray, sigmas: np.ndarray) -> Dict[str, Any]:
    """
    Fit both Arrhenius and VFT and compare.
    
    Returns both fit results and a recommendation.
    """
    arr_result = arrhenius_fit(temps_k, sigmas)
    vft_result = vft_fit(temps_k, sigmas)
    
    recommendation = None
    
    if arr_result.get("success") and vft_result.get("success"):
        arr_r2 = arr_result["r_squared"]
        vft_r2 = vft_result["r_squared"]
        
        if vft_r2 - arr_r2 > 0.02:
            recommendation = "VFT provides significantly better fit"
        elif arr_r2 - vft_r2 > 0.02:
            recommendation = "Arrhenius provides better fit (may indicate crystalline SPE)"
        else:
            recommendation = "Both fits are comparable"
    elif arr_result.get("success"):
        recommendation = "Only Arrhenius fit succeeded"
    elif vft_result.get("success"):
        recommendation = "Only VFT fit succeeded"
    else:
        recommendation = "Neither fit succeeded"
    
    return {
        "arrhenius": arr_result,
        "vft": vft_result,
        "recommendation": recommendation
    }


def generate_fit_curves(temps_k: np.ndarray, arr_result: Dict, vft_result: Dict,
                        n_points: int = 100) -> Dict[str, np.ndarray]:
    """
    Generate smooth curves for plotting fitted models.
    
    Returns arrays for plotting: T_fit, sigma_arr_fit, sigma_vft_fit
    """
    T_range = np.linspace(temps_k.min(), temps_k.max(), n_points)
    
    result = {"T_K": T_range}
    
    # Arrhenius curve
    if arr_result.get("success"):
        ln_A = arr_result["ln_A"]
        slope = arr_result["slope"]
        ln_sigma_arr = ln_A + slope * (1.0 / T_range)
        result["sigma_arrhenius"] = np.exp(ln_sigma_arr)
    
    # VFT curve
    if vft_result.get("success"):
        A = vft_result["A_s_cm"]
        B = vft_result["B_K"]
        T0 = vft_result["T0_K"]
        result["sigma_vft"] = A * np.exp(-B / (T_range - T0))
    
    return result
