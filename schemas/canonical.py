"""
Canonical DataFrame schemas for all measurement types.

All imported data is normalized to these internal schemas before computation.
Units are enforced and documented for traceability.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

# =============================================================================
# EIS Schema: Electrochemical Impedance Spectroscopy
# =============================================================================
EIS_SCHEMA = {
    "freq_hz": float,      # Frequency in Hz (required)
    "z_re_ohm": float,     # Real impedance in Ω (required)
    "z_im_ohm": float,     # Imaginary impedance in Ω, positive = capacitive (required)
    "temp_k": float,       # Temperature in K (optional)
}

EIS_REQUIRED = ["freq_hz", "z_re_ohm", "z_im_ohm"]
EIS_OPTIONAL = ["temp_k"]

# =============================================================================
# LSV Schema: Linear Sweep Voltammetry
# =============================================================================
LSV_SCHEMA = {
    "e_v": float,          # Potential in V vs reference (required)
    "j_ma_cm2": float,     # Current density in mA/cm² (required)
}

LSV_REQUIRED = ["e_v", "j_ma_cm2"]
LSV_OPTIONAL = []

# Metadata stored separately
LSV_METADATA = {
    "area_cm2": float,     # Electrode area in cm²
    "reference": str,      # Reference electrode type
    "scan_rate_mv_s": float,  # Scan rate in mV/s
}

# =============================================================================
# CA Schema: Chronoamperometry
# =============================================================================
CA_SCHEMA = {
    "t_s": float,          # Time in seconds (required)
    "i_a": float,          # Current in A (required)
    "v_v": float,          # Voltage in V (optional)
}

CA_REQUIRED = ["t_s", "i_a"]
CA_OPTIONAL = ["v_v"]

# =============================================================================
# Transmittance Schema: Optical transmittance vs time
# =============================================================================
TRANSMITTANCE_SCHEMA = {
    "t_s": float,          # Time in seconds (required)
    "t_frac": float,       # Transmittance as fraction [0, 1] (required)
}

TRANSMITTANCE_REQUIRED = ["t_s", "t_frac"]
TRANSMITTANCE_OPTIONAL = []

# Metadata stored separately
TRANSMITTANCE_METADATA = {
    "wavelength_nm": float,  # Wavelength in nm, or -1 for Tvis
}

# =============================================================================
# Unit Conversion Factors
# =============================================================================
UNIT_CONVERSIONS = {
    # Frequency
    "khz_to_hz": 1000.0,
    "mhz_to_hz": 1e6,
    
    # Impedance
    "kohm_to_ohm": 1000.0,
    "mohm_to_ohm": 1e6,
    
    # Current
    "ma_to_a": 1e-3,
    "ua_to_a": 1e-6,
    "na_to_a": 1e-9,
    
    # Current density
    "a_cm2_to_ma_cm2": 1000.0,
    "ua_cm2_to_ma_cm2": 1e-3,
    
    # Potential
    "mv_to_v": 1e-3,
    
    # Length
    "mm_to_cm": 0.1,
    "um_to_cm": 1e-4,
    "m_to_cm": 100.0,
    
    # Area
    "mm2_to_cm2": 0.01,
    "m2_to_cm2": 1e4,
    
    # Temperature
    "c_to_k": lambda x: x + 273.15,
    "k_to_k": lambda x: x,
    
    # Transmittance (percentage to fraction)
    "percent_to_frac": 0.01,
}


@dataclass
class ColumnMapping:
    """Stores the mapping from original columns to canonical schema."""
    original_columns: List[str]
    mapped_to: List[str]
    unit_conversions: Dict[str, Dict[str, Any]]
    skip_rows: int = 0
    delimiter: str = ","
    encoding: str = "utf-8"
    
    def to_dict(self) -> dict:
        return {
            "original_columns": self.original_columns,
            "mapped_to": self.mapped_to,
            "unit_conversions": self.unit_conversions,
            "skip_rows": self.skip_rows,
            "delimiter": self.delimiter,
            "encoding": self.encoding,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "ColumnMapping":
        return cls(**d)


def get_schema_for_type(measurement_type: str) -> tuple:
    """Get schema, required, and optional columns for a measurement type."""
    schemas = {
        "EIS": (EIS_SCHEMA, EIS_REQUIRED, EIS_OPTIONAL),
        "LSV": (LSV_SCHEMA, LSV_REQUIRED, LSV_OPTIONAL),
        "CA": (CA_SCHEMA, CA_REQUIRED, CA_OPTIONAL),
        "Transmittance": (TRANSMITTANCE_SCHEMA, TRANSMITTANCE_REQUIRED, TRANSMITTANCE_OPTIONAL),
    }
    return schemas.get(measurement_type, (None, None, None))
