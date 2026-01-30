"""
Reports Page - Export measurement reports.
"""
import streamlit as st
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db import list_measurements, get_measurement
from utils.i18n import t, init_language, language_selector, get_current_language

# Initialize language
init_language()

st.set_page_config(page_title=t("reports.page_title"), page_icon="📝", layout="wide")

# Sidebar language selector
with st.sidebar:
    language_selector()

st.title(t("reports.title"))
st.markdown(t("reports.subtitle"))

# List available measurements
measurements = list_measurements()

if measurements:
    st.subheader(t("reports.select_measurement"))
    
    # Create selection table
    meas_df = pd.DataFrame(measurements)
    meas_df["select"] = False
    
    selected_id = st.selectbox(
        t("reports.measurement_id"),
        options=[m["id"] for m in measurements],
        format_func=lambda x: f"ID {x}: {next((m['measurement_type'] for m in measurements if m['id']==x), '')} - {next((m['raw_file_path'] for m in measurements if m['id']==x), '')}"
    )
    
    if selected_id:
        # Fetch full measurement
        meas = get_measurement(selected_id)
        
        if meas:
            st.subheader(t("reports.details"))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(t("reports.id", id=meas['id']))
                st.markdown(t("reports.type", type=meas['measurement_type']))
                st.markdown(t("reports.file", file=meas['raw_file_path']))
                st.markdown(t("reports.hash", hash=meas['raw_file_hash'][:16]))
            
            with col2:
                st.markdown(t("reports.created", time=meas['created_at']))
                st.markdown(t("reports.revision", rev=meas.get('revision', 1)))
                st.markdown(t("reports.software", ver=meas.get('software_version', 'N/A')))
            
            # Parameters
            if meas.get("params"):
                st.subheader(t("reports.parameters"))
                st.json(meas["params"])
            
            # Results
            if meas.get("results"):
                st.subheader(t("reports.results"))
                st.json(meas["results"])
            
            # Import mapping
            if meas.get("import_mapping"):
                with st.expander(t("reports.import_mapping")):
                    st.json(meas["import_mapping"])
            
            # Export options
            st.subheader(t("reports.export_report"))
            
            col1, col2 = st.columns(2)
            
            # Get current language for report generation
            lang = get_current_language()
            
            with col1:
                if st.button(t("reports.btn_json")):
                    report = {
                        "report_generated": datetime.now().isoformat(),
                        "measurement": meas,
                        "software_version": "0.1.0"
                    }
                    
                    json_str = json.dumps(report, indent=2, default=str)
                    
                    st.download_button(
                        label=t("reports.download_json"),
                        data=json_str,
                        file_name=f"report_measurement_{selected_id}.json",
                        mime="application/json"
                    )
            
            with col2:
                if st.button(t("reports.btn_markdown")):
                    # Use translated report title based on current language
                    if lang == "zh-CN":
                        md_report = f"""# 测量报告

## 元数据
- **ID：** {meas['id']}
- **类型：** {meas['measurement_type']}
- **文件：** {meas['raw_file_path']}
- **哈希：** {meas['raw_file_hash']}
- **创建时间：** {meas['created_at']}
- **软件：** {meas.get('software_version', 'N/A')}

## 参数
```json
{json.dumps(meas.get('params', {}), indent=2)}
```

## 结果
```json
{json.dumps(meas.get('results', {}), indent=2)}
```

---
*报告生成时间：{datetime.now().isoformat()}*
"""
                    else:
                        md_report = f"""# Measurement Report

## Metadata
- **ID:** {meas['id']}
- **Type:** {meas['measurement_type']}
- **File:** {meas['raw_file_path']}
- **Hash:** {meas['raw_file_hash']}
- **Created:** {meas['created_at']}
- **Software:** {meas.get('software_version', 'N/A')}

## Parameters
```json
{json.dumps(meas.get('params', {}), indent=2)}
```

## Results
```json
{json.dumps(meas.get('results', {}), indent=2)}
```

---
*Report generated: {datetime.now().isoformat()}*
"""
                    
                    st.download_button(
                        label=t("reports.download_markdown"),
                        data=md_report,
                        file_name=f"report_measurement_{selected_id}.md",
                        mime="text/markdown"
                    )

else:
    st.info(t("reports.no_measurements"))
    
    st.markdown(t("reports.to_generate"))

# Batch report
st.subheader(t("reports.batch_report"))
st.info(t("reports.batch_coming"))
