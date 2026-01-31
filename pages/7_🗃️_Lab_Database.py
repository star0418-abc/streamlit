"""
Lab Database Page - Manage recipes, batches, and samples.
"""
import streamlit as st
import pandas as pd
from datetime import date
from pathlib import Path

# Idempotent path setup (avoids duplicate insertions on reruns)
import sys
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Guard database import - may fail on Cloud if init fails
try:
    from database.db import (
        create_recipe, get_recipe, list_recipes,
        create_batch, get_batch, list_batches,
        create_sample, get_sample, list_samples,
        list_measurements
    )
    DB_AVAILABLE = True
except Exception as _db_err:
    DB_AVAILABLE = False
    # Define stubs that return empty data
    create_recipe = create_batch = create_sample = None
    get_recipe = get_batch = get_sample = lambda x: None
    list_recipes = list_batches = list_samples = list_measurements = lambda *a, **kw: []
    _db_error_msg = str(_db_err)

from utils.i18n import t, init_language, language_selector

# Initialize language
init_language()

st.set_page_config(page_title=t("database.page_title"), page_icon="🗃️", layout="wide")

# Sidebar language selector
with st.sidebar:
    language_selector()

st.title(t("database.title"))
st.markdown(t("database.subtitle"))

# Show error if database is not available
if not DB_AVAILABLE:
    st.error("❌ 数据库不可用 / Database not available")
    st.warning(f"详情 / Details: {_db_error_msg}")
    st.info(
        "💡 在 Streamlit Cloud 上，数据库使用临时存储。首次访问时会自动初始化。\n\n"
        "On Streamlit Cloud, the database uses temporary storage and will be initialized on first access."
    )
    st.stop()


tab1, tab2, tab3, tab4 = st.tabs([
    t("database.tab_recipes"), 
    t("database.tab_batches"), 
    t("database.tab_samples"), 
    t("database.tab_measurements")
])

# =============================================================================
# Recipes Tab
# =============================================================================
with tab1:
    st.subheader(t("database.recipes"))
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(t("database.create_recipe"))
        
        with st.form("new_recipe"):
            recipe_name = st.text_input(t("database.recipe_name"))
            recipe_desc = st.text_area(t("database.description"))
            
            st.markdown(t("database.components_json"))
            components_str = st.text_area(
                t("database.components"),
                value='{\n  "salt": {"LiTFSI": {"wt_pct": 20}},\n  "polymer": {"PEO": {"wt_pct": 60}},\n  "plasticizer": {"PC": {"wt_pct": 20}}\n}',
                height=150
            )
            
            if st.form_submit_button(t("database.btn_create_recipe")):
                try:
                    import json
                    components = json.loads(components_str)
                    recipe_id = create_recipe(recipe_name, components, recipe_desc)
                    st.success(t("database.created_recipe", id=recipe_id))
                    st.rerun()
                except Exception as e:
                    st.error(f"{t('common.error')}: {e}")
    
    with col2:
        st.markdown(t("database.existing_recipes"))
        recipes = list_recipes()
        
        if recipes:
            recipes_df = pd.DataFrame(recipes)
            st.dataframe(recipes_df, use_container_width=True)
        else:
            st.info(t("database.no_recipes"))

# =============================================================================
# Batches Tab
# =============================================================================
with tab2:
    st.subheader(t("database.batches"))
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(t("database.create_batch"))
        
        recipes = list_recipes()
        recipe_options = {f"{r['id']}: {r['name']}": r['id'] for r in recipes}
        
        with st.form("new_batch"):
            selected_recipe = st.selectbox(t("database.recipe"), list(recipe_options.keys()) or [t("database.no_recipes")])
            operator = st.text_input(t("database.operator"))
            batch_date = st.date_input(t("database.date"), value=date.today())
            process_notes = st.text_area(t("database.process_notes"))
            
            if st.form_submit_button(t("database.btn_create_batch")):
                if selected_recipe and selected_recipe != t("database.no_recipes"):
                    recipe_id = recipe_options[selected_recipe]
                    batch_id = create_batch(
                        recipe_id=recipe_id,
                        operator=operator,
                        batch_date=str(batch_date),
                        notes=process_notes
                    )
                    st.success(t("database.created_batch", id=batch_id))
                    st.rerun()
                else:
                    st.error(t("database.select_recipe"))
    
    with col2:
        st.markdown(t("database.existing_batches"))
        batches = list_batches()
        
        if batches:
            batches_df = pd.DataFrame(batches)
            st.dataframe(batches_df, use_container_width=True)
        else:
            st.info(t("database.no_batches"))

# =============================================================================
# Samples Tab
# =============================================================================
with tab3:
    st.subheader(t("database.samples"))
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(t("database.create_sample"))
        
        batches = list_batches()
        batch_options = {f"{b['id']}: {b['batch_date']}": b['id'] for b in batches}
        
        with st.form("new_sample"):
            selected_batch = st.selectbox(t("database.batch"), list(batch_options.keys()) or [t("database.no_batches")])
            sample_code = st.text_input(t("database.sample_code"), placeholder=t("database.sample_code_placeholder"))
            thickness = st.number_input(t("database.thickness"), value=0.01, format="%.4f")
            area = st.number_input(t("database.area"), value=1.0, format="%.4f")
            intended_test = st.selectbox(t("database.intended_test"), ["EIS", "LSV", "CA", "Cycling", "Other"])
            sample_notes = st.text_area(t("common.notes"))
            
            if st.form_submit_button(t("database.btn_create_sample")):
                if selected_batch and selected_batch != t("database.no_batches"):
                    batch_id = batch_options[selected_batch]
                    try:
                        sample_id = create_sample(
                            batch_id=batch_id,
                            sample_code=sample_code,
                            thickness_cm=thickness,
                            area_cm2=area,
                            intended_test=intended_test,
                            notes=sample_notes
                        )
                        st.success(t("database.created_sample", id=sample_id))
                        st.rerun()
                    except Exception as e:
                        st.error(f"{t('common.error')}: {e}")
                else:
                    st.error(t("database.select_batch"))
    
    with col2:
        st.markdown(t("database.existing_samples"))
        samples = list_samples()
        
        if samples:
            samples_df = pd.DataFrame(samples)
            st.dataframe(samples_df, use_container_width=True)
        else:
            st.info(t("database.no_samples"))

# =============================================================================
# Measurements Tab
# =============================================================================
with tab4:
    st.subheader(t("database.measurements"))
    
    measurements = list_measurements()
    
    if measurements:
        meas_df = pd.DataFrame(measurements)
        st.dataframe(meas_df, use_container_width=True)
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            type_filter = st.selectbox(t("database.filter_by_type"), [t("database.filter_all")] + list(meas_df["measurement_type"].unique()))
        
        if type_filter != t("database.filter_all"):
            filtered = list_measurements(measurement_type=type_filter)
            st.dataframe(pd.DataFrame(filtered), use_container_width=True)
    else:
        st.info(t("database.no_measurements"))
