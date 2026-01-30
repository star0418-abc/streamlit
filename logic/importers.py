"""
Import layer for CSV/TXT files with column mapping and unit conversion.

Supports auto-detection of file format, preview, and normalization to canonical schemas.
"""
import pandas as pd
import numpy as np
import re
import hashlib
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import chardet

from schemas.canonical import (
    EIS_SCHEMA, EIS_REQUIRED,
    LSV_SCHEMA, LSV_REQUIRED,
    CA_SCHEMA, CA_REQUIRED,
    TRANSMITTANCE_SCHEMA, TRANSMITTANCE_REQUIRED,
    UNIT_CONVERSIONS, ColumnMapping, get_schema_for_type
)


def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _is_numeric(s: str) -> bool:
    """Check if a string represents a numeric value."""
    try:
        float(s.replace(",", "."))
        return True
    except (ValueError, AttributeError):
        return False


def detect_file_format(file_path: str) -> Dict[str, Any]:
    """
    Auto-detect file encoding, delimiter, and header row.
    
    Returns:
        dict with keys: encoding, delimiter, header_row, skip_rows
    """
    # Detect encoding
    with open(file_path, 'rb') as f:
        raw = f.read(10000)
        detected = chardet.detect(raw)
        encoding = detected.get('encoding', 'utf-8') or 'utf-8'
    
    # Read first 30 lines
    with open(file_path, 'r', encoding=encoding, errors='replace') as f:
        lines = []
        for _ in range(30):
            line = f.readline()
            if not line:
                break
            lines.append(line)
    
    if not lines:
        return {"encoding": encoding, "delimiter": ",", "header_row": 0, "skip_rows": 0}
    
    # Detect delimiter by counting occurrences
    delimiters = ['\t', ',', ';', ' ']
    delimiter_counts = {d: sum(l.count(d) for l in lines) for d in delimiters}
    delimiter = max(delimiters, key=lambda d: delimiter_counts[d])
    
    # Find header row (first row with mostly non-numeric values)
    header_row = 0
    for i, line in enumerate(lines):
        parts = [p.strip() for p in line.strip().split(delimiter) if p.strip()]
        if len(parts) < 2:
            continue
        non_numeric = sum(1 for p in parts if not _is_numeric(p))
        if non_numeric >= len(parts) // 2:
            header_row = i
            break
    
    return {
        "encoding": encoding,
        "delimiter": delimiter,
        "header_row": header_row,
        "skip_rows": header_row
    }


def preview_file(file_path: str, nrows: int = 10) -> Tuple[pd.DataFrame, Dict]:
    """
    Preview a file and return DataFrame + detected format.
    
    Returns:
        (preview_df, format_dict)
    """
    fmt = detect_file_format(file_path)
    
    try:
        df = pd.read_csv(
            file_path,
            encoding=fmt["encoding"],
            sep=fmt["delimiter"],
            skiprows=fmt["skip_rows"],
            nrows=nrows
        )
    except Exception as e:
        # Fallback: try with different settings
        df = pd.read_csv(file_path, nrows=nrows)
    
    return df, fmt


def extract_temp_from_filename(filename: str) -> Optional[float]:
    """
    Extract temperature from filename using regex patterns.
    
    Supported patterns:
    - 25C, 60C -> Kelvin
    - 298K, 333K -> Kelvin
    - RT -> 298.15 K
    
    Returns temperature in Kelvin or None.
    """
    patterns = [
        # Match numbers followed by C (case insensitive), not followed by other letters
        (r'[-_]?(\d+)\s*[Cc](?![a-zA-Z])', lambda m: float(m.group(1)) + 273.15),
        # Match numbers followed by K (case insensitive), not followed by other letters
        (r'[-_]?(\d{2,3})\s*[Kk](?![a-zA-Z])', lambda m: float(m.group(1))),
        # Match RT (room temperature)
        (r'(?:^|[-_])RT(?:[-_]|$)', lambda m: 298.15),
    ]
    
    for pattern, converter in patterns:
        match = re.search(pattern, filename)
        if match:
            return converter(match)
    return None


def apply_unit_conversion(value: float, conversion_key: str) -> float:
    """Apply a unit conversion by key."""
    conv = UNIT_CONVERSIONS.get(conversion_key)
    if conv is None:
        return value
    if callable(conv):
        return conv(value)
    return value * conv


def normalize_to_schema(df: pd.DataFrame, mapping: ColumnMapping, 
                        measurement_type: str) -> pd.DataFrame:
    """
    Normalize a DataFrame to the canonical schema using the provided mapping.
    
    Args:
        df: Input DataFrame with original column names
        mapping: ColumnMapping with column mappings and unit conversions
        measurement_type: One of "EIS", "LSV", "CA", "Transmittance"
    
    Returns:
        DataFrame with canonical column names and converted units
    """
    schema, required, optional = get_schema_for_type(measurement_type)
    if schema is None:
        raise ValueError(f"Unknown measurement type: {measurement_type}")
    
    result = pd.DataFrame()
    
    # Map and convert columns
    for orig_col, target_col in zip(mapping.original_columns, mapping.mapped_to):
        if target_col and orig_col in df.columns:
            values = df[orig_col].values.astype(float)
            
            # Apply unit conversion if specified
            if target_col in mapping.unit_conversions:
                conv_info = mapping.unit_conversions[target_col]
                conv_key = conv_info.get("key")
                if conv_key:
                    conv = UNIT_CONVERSIONS.get(conv_key)
                    if conv is not None:
                        if callable(conv):
                            values = np.array([conv(v) for v in values])
                        else:
                            values = values * conv
            
            result[target_col] = values
    
    # Validate required columns
    missing = [col for col in required if col not in result.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return result


def normalize_transmittance(t_values: np.ndarray) -> np.ndarray:
    """
    Normalize transmittance to [0, 1] range.
    
    Handles both 0-100% and 0-1 formats.
    """
    t_max = np.nanmax(t_values)
    if t_max > 1.5:  # Likely percentage (0-100)
        return t_values / 100.0
    return t_values


def load_and_normalize(file_path: str, mapping: ColumnMapping,
                       measurement_type: str) -> pd.DataFrame:
    """
    Load a file and normalize to canonical schema.
    
    This is the main entry point for importing data.
    """
    fmt = {
        "encoding": mapping.encoding,
        "delimiter": mapping.delimiter,
        "skip_rows": mapping.skip_rows
    }
    
    df = pd.read_csv(
        file_path,
        encoding=fmt["encoding"],
        sep=fmt["delimiter"],
        skiprows=fmt["skip_rows"]
    )
    
    result = normalize_to_schema(df, mapping, measurement_type)
    
    # Special handling for transmittance normalization
    if measurement_type == "Transmittance" and "t_frac" in result.columns:
        result["t_frac"] = normalize_transmittance(result["t_frac"].values)
    
    return result


def suggest_column_mapping(df_columns: List[str], measurement_type: str) -> Dict[str, str]:
    """
    Suggest column mappings based on common column name patterns.
    
    Returns dict: {original_column: canonical_column}
    """
    schema, required, optional = get_schema_for_type(measurement_type)
    if schema is None:
        return {}
    
    # Common patterns for each canonical column
    patterns = {
        # EIS
        "freq_hz": [r"freq", r"f\s*\(?\s*hz", r"frequency"],
        "z_re_ohm": [r"z['\s]*re", r"z'", r"re\s*\(?\s*z", r"real"],
        "z_im_ohm": [r"z['\s]*im", r"z''", r"-?z''", r"im\s*\(?\s*z", r"imag"],
        "temp_k": [r"temp", r"t\s*\(?\s*k\)?", r"temperature"],
        
        # LSV
        "e_v": [r"e\s*\(?\s*v\)?", r"potential", r"voltage", r"ewe"],
        "j_ma_cm2": [r"j\s*\(?\s*ma", r"current\s*dens", r"i/a"],
        
        # CA
        "t_s": [r"time", r"t\s*\(?\s*s\)?", r"t/s"],
        "i_a": [r"i\s*\(?\s*a\)?", r"current", r"<i>"],
        "v_v": [r"v\s*\(?\s*v\)?", r"voltage", r"ewe"],
        
        # Transmittance
        "t_frac": [r"t\s*\(?\s*%\)?", r"trans", r"t\s*vis", r"%t"],
    }
    
    suggestions = {}
    used_targets = set()
    
    for orig_col in df_columns:
        col_lower = orig_col.lower().strip()
        for target_col, pattern_list in patterns.items():
            if target_col in schema and target_col not in used_targets:
                for pattern in pattern_list:
                    if re.search(pattern, col_lower, re.IGNORECASE):
                        suggestions[orig_col] = target_col
                        used_targets.add(target_col)
                        break
                if orig_col in suggestions:
                    break
    
    return suggestions


def validate_imported_data(df: pd.DataFrame, measurement_type: str) -> List[str]:
    """
    Validate imported data and return list of warnings/errors.
    """
    warnings = []
    
    if measurement_type == "EIS":
        if "freq_hz" in df.columns:
            if (df["freq_hz"] <= 0).any():
                warnings.append("Frequency contains zero or negative values")
            if df["freq_hz"].max() > 1e8:
                warnings.append("Frequency > 100 MHz - check units")
        
        if "z_re_ohm" in df.columns and "z_im_ohm" in df.columns:
            # Check for typical Nyquist plot shape
            if (df["z_im_ohm"] < 0).all():
                warnings.append("All Z_im negative - may need sign flip for capacitive convention")
    
    elif measurement_type == "Transmittance":
        if "t_frac" in df.columns:
            if (df["t_frac"] < 0).any() or (df["t_frac"] > 1).any():
                warnings.append("Transmittance values outside [0, 1] after normalization")
    
    elif measurement_type == "LSV":
        if "j_ma_cm2" in df.columns:
            if np.abs(df["j_ma_cm2"]).max() > 1000:
                warnings.append("Current density > 1 A/cm² - check units")
    
    return warnings
