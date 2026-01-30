"""
Import Data Page - Upload and map data files to canonical schemas.
"""
import streamlit as st
import pandas as pd
import tempfile
import os
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from logic.importers import (
    detect_file_format, preview_file, suggest_column_mapping,
    load_and_normalize, validate_imported_data, compute_file_hash
)
from schemas.canonical import ColumnMapping, get_schema_for_type
from database.db import create_measurement
from utils.i18n import t, init_language, language_selector

# Initialize language
init_language()

st.set_page_config(page_title=t("import.page_title"), page_icon="📊", layout="wide")

# Sidebar language selector
with st.sidebar:
    language_selector()

st.title(t("import.title"))
st.markdown(t("import.subtitle"))

# Measurement type selection
measurement_type = st.selectbox(
    t("import.select_type"),
    ["EIS", "LSV", "CA", "Transmittance"],
    help=t("import.select_type_help")
)

schema, required_cols, optional_cols = get_schema_for_type(measurement_type)

st.info(t("import.required_cols", cols=", ".join(required_cols)))
if optional_cols:
    st.caption(t("import.optional_cols", cols=", ".join(optional_cols)))

# File upload
uploaded_file = st.file_uploader(
    t("import.upload_file"),
    type=["csv", "txt", "dat"],
    help=t("import.upload_help")
)

if uploaded_file is not None:
    # Save to temp file for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    try:
        # Detect format and preview
        st.subheader(t("import.file_preview"))
        preview_df, fmt = preview_file(tmp_path, nrows=10)
        
        col1, col2 = st.columns(2)
        with col1:
            st.caption(t("import.detected_encoding", encoding=fmt['encoding']))
            st.caption(t("import.detected_delimiter", delimiter=repr(fmt['delimiter'])))
        with col2:
            st.caption(t("import.header_row", row=fmt['header_row']))
            st.caption(t("import.columns_found", count=len(preview_df.columns)))
        
        st.dataframe(preview_df, use_container_width=True)
        
        # Column mapping
        st.subheader(t("import.column_mapping"))
        st.markdown(t("import.column_mapping_desc"))
        
        # Get suggestions
        suggestions = suggest_column_mapping(list(preview_df.columns), measurement_type)
        
        # Create mapping UI
        mapping_cols = st.columns(2)
        column_map = {}
        
        all_cols = required_cols + optional_cols
        
        for i, target_col in enumerate(all_cols):
            with mapping_cols[i % 2]:
                # Find suggested source column
                suggested = None
                for orig, tgt in suggestions.items():
                    if tgt == target_col:
                        suggested = orig
                        break
                
                options = [t("import.not_mapped")] + list(preview_df.columns)
                default_idx = 0
                if suggested and suggested in options:
                    default_idx = options.index(suggested)
                
                is_required = target_col in required_cols
                label = f"**{target_col}**" if is_required else target_col
                
                selected = st.selectbox(
                    label,
                    options,
                    index=default_idx,
                    key=f"map_{target_col}"
                )
                
                if selected != t("import.not_mapped"):
                    column_map[selected] = target_col
        
        # Unit conversions
        st.subheader(t("import.unit_conversions"))
        unit_conversions = {}
        
        with st.expander(t("import.configure_units")):
            for orig_col, target_col in column_map.items():
                conversion_options = {
                    "freq_hz": ["Hz (no conversion)", "kHz → Hz", "MHz → Hz"],
                    "z_re_ohm": ["Ω (no conversion)", "kΩ → Ω", "MΩ → Ω"],
                    "z_im_ohm": ["Ω (no conversion)", "kΩ → Ω", "MΩ → Ω"],
                    "e_v": ["V (no conversion)", "mV → V"],
                    "j_ma_cm2": ["mA/cm² (no conversion)", "A/cm² → mA/cm²", "µA/cm² → mA/cm²"],
                    "i_a": ["A (no conversion)", "mA → A", "µA → A"],
                    "t_s": ["s (no conversion)", "ms → s", "min → s"],
                    "t_frac": ["fraction [0-1] (no conversion)", "% [0-100] → fraction"],
                }
                
                if target_col in conversion_options:
                    conv = st.selectbox(
                        f"{orig_col} → {target_col}",
                        conversion_options[target_col],
                        key=f"conv_{target_col}"
                    )
                    
                    conv_key = None
                    if "kHz" in conv:
                        conv_key = "khz_to_hz"
                    elif "MHz" in conv:
                        conv_key = "mhz_to_hz"
                    elif "kΩ" in conv:
                        conv_key = "kohm_to_ohm"
                    elif "MΩ" in conv:
                        conv_key = "mohm_to_ohm"
                    elif "mV" in conv:
                        conv_key = "mv_to_v"
                    elif "A/cm² → mA" in conv:
                        conv_key = "a_cm2_to_ma_cm2"
                    elif "µA/cm² → mA" in conv:
                        conv_key = "ua_cm2_to_ma_cm2"
                    elif "mA → A" in conv:
                        conv_key = "ma_to_a"
                    elif "µA → A" in conv:
                        conv_key = "ua_to_a"
                    elif "% [0-100]" in conv:
                        conv_key = "percent_to_frac"
                    
                    if conv_key:
                        unit_conversions[target_col] = {"key": conv_key}
        
        # Validate and import
        st.subheader(t("import.import_data"))
        
        # Check required columns
        mapped_targets = set(column_map.values())
        missing_required = [col for col in required_cols if col not in mapped_targets]
        
        if missing_required:
            st.error(t("import.missing_required", cols=", ".join(missing_required)))
        else:
            if st.button(t("import.btn_import"), type="primary"):
                try:
                    # Create mapping object
                    mapping = ColumnMapping(
                        original_columns=list(column_map.keys()),
                        mapped_to=list(column_map.values()),
                        unit_conversions=unit_conversions,
                        skip_rows=fmt["skip_rows"],
                        delimiter=fmt["delimiter"],
                        encoding=fmt["encoding"]
                    )
                    
                    # Load and normalize
                    normalized_df = load_and_normalize(tmp_path, mapping, measurement_type)
                    
                    # Validate
                    warnings = validate_imported_data(normalized_df, measurement_type)
                    
                    if warnings:
                        for w in warnings:
                            st.warning(w)
                    
                    st.success(t("import.import_success", count=len(normalized_df)))
                    
                    # Show normalized data
                    st.subheader(t("import.normalized_data"))
                    st.dataframe(normalized_df, use_container_width=True)
                    
                    # Store in session state for other pages
                    file_hash = compute_file_hash(tmp_path)
                    
                    st.session_state["imported_data"] = {
                        "df": normalized_df,
                        "type": measurement_type,
                        "filename": uploaded_file.name,
                        "file_hash": file_hash,
                        "mapping": mapping.to_dict()
                    }
                    
                    st.info(t("import.data_stored"))
                    
                    # Option to save to database
                    st.subheader(t("import.save_optional"))
                    
                    with st.form("save_form"):
                        sample_code = st.text_input(t("import.sample_code"))
                        notes = st.text_area(t("common.notes"))
                        
                        if st.form_submit_button(t("import.btn_save_raw")):
                            measurement_id = create_measurement(
                                measurement_type=measurement_type,
                                raw_file_path=uploaded_file.name,
                                raw_file_hash=file_hash,
                                import_mapping=mapping.to_dict(),
                                software_version="0.1.0"
                            )
                            st.success(t("import.saved_id", id=measurement_id))
                
                except Exception as e:
                    st.error(t("import.import_failed", error=str(e)))
    
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass
else:
    st.info(t("import.upload_hint"))
