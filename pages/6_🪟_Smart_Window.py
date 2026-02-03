"""
Smart Window Page - Electrochromic device analysis.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Check plotly availability before importing
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    make_subplots = None

# Idempotent path setup (avoids duplicate insertions on reruns)
import sys
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from logic.smart_window import (
    align_ca_transmittance, compute_delta_t, compute_response_time,
    compute_charge_density, compute_coloration_efficiency,
    segment_cycles_by_voltage, segment_cycles_by_transmittance,
    compute_cycling_metrics, classify_step
)
from utils.i18n import t, init_language, language_selector
from utils.ui_header import render_top_banner

# Initialize language
init_language()

st.set_page_config(page_title=t("smart_window.page_title"), page_icon="🪟", layout="wide")

# Render top banner
render_top_banner()

# Sidebar language selector
with st.sidebar:
    language_selector()

# Check plotly availability
if not PLOTLY_AVAILABLE:
    st.error("❌ **plotly 未安装 / plotly is not installed**")
    st.code("pip install plotly", language="bash")
    st.info(
        "💡 建议运行以下命令安装所有依赖：\n\n"
        "Recommended: install all dependencies:\n\n"
        "```\npip install -r requirements.txt\n```"
    )
    st.stop()

st.title(t("smart_window.title"))
st.markdown(t("smart_window.subtitle"))

# Initialize session state
if "ca_data" not in st.session_state:
    st.session_state["ca_data"] = None
if "tt_data" not in st.session_state:
    st.session_state["tt_data"] = None

st.subheader(t("smart_window.data_input"))

col1, col2 = st.columns(2)

with col1:
    st.markdown(t("smart_window.ca_data"))
    
    imported = st.session_state.get("imported_data")
    if imported and imported.get("type") == "CA":
        st.success(t("smart_window.ca_loaded", filename=imported['filename']))
        if st.button(t("smart_window.btn_use_ca")):
            st.session_state["ca_data"] = imported["df"]
            st.rerun()
    
    if st.session_state["ca_data"] is not None:
        st.info(t("smart_window.ca_points", count=len(st.session_state['ca_data'])))

with col2:
    st.markdown(t("smart_window.tt_data"))
    
    if imported and imported.get("type") == "Transmittance":
        st.success(t("smart_window.tt_loaded", filename=imported['filename']))
        if st.button(t("smart_window.btn_use_tt")):
            st.session_state["tt_data"] = imported["df"]
            st.rerun()
    
    if st.session_state["tt_data"] is not None:
        st.info(t("smart_window.tt_points", count=len(st.session_state['tt_data'])))

# Demo mode with synthetic data
with st.expander(t("smart_window.demo_data")):
    if st.button(t("smart_window.btn_load_demo")):
        # Generate synthetic EC data
        t_arr = np.linspace(0, 600, 1000)
        
        # Simulated voltage steps
        v = np.where((t_arr % 120) < 60, 1.5, -0.5)
        
        # Simulated current response
        i = 0.001 * np.exp(-t_arr % 60 / 10) * np.sign(np.diff(v, prepend=v[0]) + 0.1)
        
        # Simulated transmittance
        T = 0.3 + 0.4 * (1 - np.exp(-((t_arr % 120) / 15))) * ((t_arr % 120) < 60).astype(float)
        T += 0.4 * np.exp(-((t_arr % 120 - 60) / 15)) * ((t_arr % 120) >= 60).astype(float)
        
        st.session_state["ca_data"] = pd.DataFrame({"t_s": t_arr, "i_a": i, "v_v": v})
        st.session_state["tt_data"] = pd.DataFrame({"t_s": t_arr, "t_frac": T})
        st.success(t("smart_window.demo_loaded"))
        st.rerun()

# Analysis
ca_data = st.session_state["ca_data"]
tt_data = st.session_state["tt_data"]

if ca_data is not None and tt_data is not None:
    st.subheader(t("smart_window.parameters"))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        area = st.number_input(t("smart_window.active_area"), value=1.0, min_value=0.01)
    
    with col2:
        response_threshold = st.selectbox(t("smart_window.response_threshold"), [0.90, 0.95], index=0)
    
    with col3:
        time_tolerance = st.number_input(t("smart_window.time_tolerance"), value=0.5, min_value=0.01)
    
    # Advanced alignment options
    with st.expander("🔧 高级对齐设置 / Advanced Alignment"):
        lag_mode = st.selectbox(
            "时间对齐模式 / Time Alignment Mode",
            ["none", "estimate", "manual"],
            index=0,
            help="estimate: 自动估计CA与光学数据的触发延迟\n"
                 "manual: 手动输入延迟\n"
                 "none: 不做延迟校正"
        )
        
        manual_lag_s = 0.0
        max_lag_s = 10.0
        lag_signal_mode = "coloring_only"  # default
        
        if lag_mode == "manual":
            manual_lag_s = st.number_input(
                "手动延迟 / Manual Lag (s)", 
                value=0.0, 
                min_value=-20.0, 
                max_value=20.0,
                help="正值表示光学数据滞后于CA数据 / Positive = optical lags CA"
            )
        elif lag_mode == "estimate":
            max_lag_s = st.number_input(
                "最大搜索延迟 / Max Lag Search (s)", 
                value=10.0, 
                min_value=1.0, 
                max_value=30.0
            )
            lag_signal_mode = st.selectbox(
                "相关信号模式 / Correlation Signal Mode",
                ["coloring_only", "bleaching_only", "max_abs_corr", "full_cycle"],
                index=0,
                help="coloring_only (推荐): 仅使用着色段估计延迟，避免符号抵消\n"
                     "bleaching_only: 仅使用褪色段\n"
                     "max_abs_corr: 两段都尝试，选相关性最高的\n"
                     "full_cycle: 旧方法，可能导致相关性抵消"
            )
        
        st.caption(
            "💡 **延迟约定**: 正延迟 = 光学信号滞后于电化学信号，"
            "校正时将从光学时间中减去此值。"
        )
    
    # Baseline correction for leakage current (GPE)
    with st.expander("⚡ 漏电流校正 / Leakage Current Correction (GPE)"):
        st.markdown(
            "**问题**: GPE系统存在漏电流，导致Q随测试时间增长，"
            "CE在不同测试时长间不可比。\n\n"
            "**Problem**: In GPE systems, leakage current causes Q to grow with time "
            "even after optical plateau, making CE non-comparable across test durations."
        )
        
        baseline_mode = st.selectbox(
            "基线校正模式 / Baseline Correction Mode",
            ["none", "offset_tail", "offset_head", "offset_both"],
            index=0,
            help="none: 不校正 (液态电解质)\n"
                 "offset_tail (推荐GPE): 用尾部中值作为漏电流基线\n"
                 "offset_head: 用头部中值\n"
                 "offset_both: 头尾平均"
        )
        
        tail_fraction = 0.2
        if baseline_mode != "none":
            tail_fraction = st.slider(
                "尾部比例 / Tail Fraction", 
                min_value=0.1, 
                max_value=0.4, 
                value=0.2,
                step=0.05,
                help="用于估计漏电流的数据尾部比例"
            )
            st.info(
                "✅ 启用漏电流校正后，将同时输出原始和校正后的Q/CE值。"
            )
        else:
            st.warning(
                "⚠️ **GPE用户注意**: 如果数据包含显著漏电流，建议选择 `offset_tail` 模式。"
            )
    
    # Align data with new parameters
    merged_df, align_meta = align_ca_transmittance(
        ca_data, tt_data, 
        tolerance_s=time_tolerance,
        lag_mode=lag_mode,
        max_lag_s=max_lag_s,
        manual_lag_s=manual_lag_s,
        lag_signal_mode=lag_signal_mode
    )
    
    st.caption(t("smart_window.aligned_points", 
                merged=align_meta['merged_points'], 
                ca=align_meta['ca_points'], 
                tt=align_meta['tt_points']))
    
    # Show lag info if applicable
    if lag_mode != "none":
        lag_s = align_meta.get('lag_s', 0)
        lag_conf = align_meta.get('lag_confident', True)
        if lag_mode == "estimate":
            lag_corr = align_meta.get('lag_correlation', 0)
            conf_icon = "✅" if lag_conf else "⚠️"
            st.caption(f"估计延迟 / Estimated lag: {lag_s:.2f} s (r={lag_corr:.3f}) {conf_icon}")
            
            # Show which segment was used
            est_meta = align_meta.get("align_meta", {}).get("lag_estimation", {})
            if est_meta:
                seg_used = est_meta.get("segment_used", "unknown")
                col_frac = est_meta.get("coloring_fraction", 0)
                bl_frac = est_meta.get("bleaching_fraction", 0)
                st.caption(
                    f"使用段 / Segment used: {seg_used} "
                    f"(着色:{col_frac:.0%}, 褪色:{bl_frac:.0%})"
                )
        else:
            st.caption(f"手动延迟 / Manual lag: {lag_s:.2f} s")
        
        # Show warnings
        for w in align_meta.get('warnings', []):
            st.warning(w)
    
    # Overall metrics
    st.subheader(t("smart_window.overall_metrics"))
    
    if st.button(t("smart_window.btn_calc_metrics"), type="primary"):
        t_s = merged_df["t_s"].values
        i_a = merged_df["i_a"].values
        t_frac = merged_df["t_frac"].values
        
        # Remove NaN values from merged data
        valid_mask = ~np.isnan(t_frac)
        t_s = t_s[valid_mask]
        i_a = i_a[valid_mask]
        t_frac = t_frac[valid_mask]
        
        if len(t_s) < 5:
            st.error("数据点不足 / Insufficient data points after alignment")
        else:
            # Classify step type
            step_info = classify_step(t_frac, i_a, t_s, area)
            segment_type = step_info["step_type"]
            
            # Basic metrics
            t_max = float(np.nanmax(t_frac))
            t_min = float(np.nanmin(t_frac))
            delta_t = compute_delta_t(t_max, t_min)
            
            # Charge with optional baseline correction
            q_result = compute_charge_density(
                t_s, i_a, area,
                baseline_mode=baseline_mode,
                tail_fraction=tail_fraction
            )
            
            # CE (with step type and raw Q for comparison)
            ce_result = compute_coloration_efficiency(
                t_max, t_min, 
                q_result["q_abs_c_cm2"],
                q_c_cm2_raw=q_result.get("q_abs_c_cm2_raw") if baseline_mode != "none" else None,
                step_type=segment_type,
                baseline_mode_used=baseline_mode
            )
            
            # Response time (plateau-based)
            rt_result = compute_response_time(
                t_s, t_frac, 
                threshold=response_threshold,
                validate_plateau=True
            )
            
            # Display
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("ΔT", f"{delta_t:.3f}")
                st.caption(f"({delta_t*100:.1f}%)")
            
            with col2:
                st.metric("T_bleached", f"{t_max:.3f}")
            
            with col3:
                st.metric("T_colored", f"{t_min:.3f}")
            
            with col4:
                ce = ce_result.get("ce_cm2_c")
                ce_raw = ce_result.get("ce_cm2_c_raw")
                if ce is not None:
                    label = "CE" if baseline_mode == "none" else "CE (校正)"
                    st.metric(label, f"{ce:.1f} cm²/C")
                    if ce_raw is not None and baseline_mode != "none":
                        st.caption(f"原始: {ce_raw:.1f}")
                else:
                    reason = ce_result.get("ce_skipped_reason", "N/A")
                    st.metric("CE", "—")
                    st.caption(f"({reason})")
            
            with col5:
                rt = rt_result.get("response_time_s")
                if rt is not None:
                    st.metric(f"t{int(response_threshold*100)}", f"{rt:.1f} s")
                    if not rt_result.get("reached_plateau", True):
                        st.caption("⚠️ 未达稳态")
                else:
                    st.metric(f"t{int(response_threshold*100)}", "—")
            
            # Segment type indicator
            type_emoji = {"coloring": "🔵", "bleaching": "⚪", "unknown": "❓"}
            st.caption(f"段类型 / Segment: {type_emoji.get(segment_type, '❓')} {segment_type}")
            
            # QC Warnings
            all_warnings = (
                q_result.get("warnings", []) +
                ce_result.get("warnings", []) + 
                ce_result.get("errors", []) +
                rt_result.get("warnings", []) +
                step_info.get("warnings", [])
            )
            for w in all_warnings:
                st.warning(w)
            
            st.caption(f"ΔOD = {ce_result.get('delta_od', 0):.4f} (log₁₀ base)")
            
            # Show Q with baseline info
            q_text = f"Q = {q_result['q_abs_c_cm2']:.4f} C/cm²"
            if baseline_mode != "none":
                q_raw = q_result.get('q_abs_c_cm2_raw', 0)
                i_baseline = q_result.get('i_baseline_A', 0)
                q_text += f" (原始: {q_raw:.4f}, 漏电流: {i_baseline:.2e} A)"
            else:
                q_text += f" (signed: {q_result.get('q_signed', 0):.4f})"
            st.caption(q_text)
            
            # Plateau quality info
            pq = rt_result.get("plateau_quality", {})
            if pq:
                st.caption(
                    f"平台质量 / Plateau: std={pq.get('tinf_std', 0):.4f}, "
                    f"slope={pq.get('tinf_slope', 0):.4f}/s"
                )
            
            # Store results
            st.session_state["sw_result"] = {
                "delta_t": delta_t,
                "t_bleached": t_max,
                "t_colored": t_min,
                "ce": ce,
                "q_c_cm2": q_result["q_abs_c_cm2"],
                "response_time_s": rt,
                "segment_type": segment_type,
                "qc_pass": rt_result.get("qc_pass", True)
            }
    
    # Visualization
    st.subheader(t("smart_window.dual_axis_plot"))
    
    t_s = merged_df["t_s"].values
    i_a = merged_df["i_a"].values
    t_frac = merged_df["t_frac"].values
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Transmittance
    fig.add_trace(
        go.Scatter(x=t_s, y=t_frac * 100, name="T (%)", line=dict(color="blue")),
        secondary_y=False
    )
    
    # Current density
    j_ma_cm2 = i_a * 1000 / area
    fig.add_trace(
        go.Scatter(x=t_s, y=j_ma_cm2, name="j (mA/cm²)", line=dict(color="red")),
        secondary_y=True
    )
    
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="T (%)", secondary_y=False)
    fig.update_yaxes(title_text="j (mA/cm²)", secondary_y=True)
    fig.update_layout(height=400)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Cycle segmentation
    st.subheader(t("smart_window.cycle_analysis"))
    
    seg_method = st.radio(t("smart_window.seg_method"), 
                          [t("smart_window.seg_voltage"), t("smart_window.seg_transmittance")])
    
    if st.button(t("smart_window.btn_segment")):
        # Remove NaN before segmentation
        valid_mask = ~np.isnan(t_frac)
        t_s_valid = t_s[valid_mask]
        i_a_valid = i_a[valid_mask]
        t_frac_valid = t_frac[valid_mask]
        
        if seg_method == t("smart_window.seg_voltage") and "v_v" in merged_df.columns:
            v_v_valid = merged_df["v_v"].values[valid_mask]
            cycles = segment_cycles_by_voltage(v_v_valid, t_s_valid)
        else:
            cycles = segment_cycles_by_transmittance(t_frac_valid, t_s_valid)
        
        if cycles:
            st.success(t("smart_window.found_segments", count=len(cycles)))
            
            # Compute per-cycle metrics with QC and baseline correction
            cycle_df = compute_cycling_metrics(
                cycles, t_frac_valid, i_a_valid, t_s_valid, area,
                response_threshold=response_threshold,
                validate_plateau=True,
                auto_split_full_cycles=True,
                baseline_mode=baseline_mode,
                tail_fraction=tail_fraction
            )
            
            if len(cycle_df) > 0:
                # Display summary
                n_valid = cycle_df.attrs.get("n_valid", 0)
                n_total = cycle_df.attrs.get("n_cycles", len(cycle_df))
                pct_valid = cycle_df.attrs.get("pct_valid", 0)
                
                st.caption(f"有效周期 / Valid cycles: {n_valid}/{n_total} ({pct_valid:.0f}%)")
                
                # Show baseline mode if applied
                if baseline_mode != "none":
                    st.caption(f"📊 基线校正模式 / Baseline mode: {baseline_mode}")
                
                # Color code by segment type
                def highlight_type(row):
                    if row.get("segment_type") == "coloring":
                        return ["background-color: #e3f2fd"] * len(row)
                    elif row.get("segment_type") == "bleaching":
                        return ["background-color: #fff3e0"] * len(row)
                    return [""] * len(row)
                
                # Select columns to display (add raw columns if baseline correction applied)
                display_cols = [
                    "cycle_label", "segment_type", "delta_t", "ce_cm2_c", 
                    "q_c_cm2", "response_time_s", "reached_plateau", "qc_pass"
                ]
                if baseline_mode != "none":
                    display_cols.insert(4, "ce_cm2_c_raw")
                    display_cols.insert(6, "q_c_cm2_raw")
                display_cols = [c for c in display_cols if c in cycle_df.columns]
                
                st.dataframe(
                    cycle_df[display_cols].style.apply(highlight_type, axis=1),
                    use_container_width=True
                )
                
                # Show warnings if any
                warnings_col = cycle_df.get("warnings")
                if warnings_col is not None:
                    for idx, w in warnings_col.items():
                        if w:
                            st.warning(f"Cycle {cycle_df.loc[idx, 'cycle_label']}: {w}")
                
                # Retention plot
                if len(cycle_df) > 1:
                    st.subheader(t("smart_window.cycling_retention"))
                    
                    fig_ret = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # ΔT retention
                    fig_ret.add_trace(
                        go.Scatter(
                            x=cycle_df["cycle_label"],
                            y=cycle_df["delta_t"] * 100,
                            mode="markers+lines",
                            name="ΔT (%)",
                            marker=dict(color="blue")
                        ),
                        secondary_y=False
                    )
                    
                    # CE (if available, only for coloring)
                    coloring_mask = cycle_df["segment_type"] == "coloring"
                    if coloring_mask.any() and cycle_df.loc[coloring_mask, "ce_cm2_c"].notna().any():
                        fig_ret.add_trace(
                            go.Scatter(
                                x=cycle_df.loc[coloring_mask, "cycle_label"],
                                y=cycle_df.loc[coloring_mask, "ce_cm2_c"],
                                mode="markers+lines",
                                name="CE (cm²/C)",
                                marker=dict(color="green")
                            ),
                            secondary_y=True
                        )
                    
                    fig_ret.update_layout(
                        xaxis_title="Cycle",
                        height=350
                    )
                    fig_ret.update_yaxes(title_text="ΔT (%)", secondary_y=False)
                    fig_ret.update_yaxes(title_text="CE (cm²/C)", secondary_y=True)
                    
                    st.plotly_chart(fig_ret, use_container_width=True)
        else:
            st.warning(t("smart_window.segment_failed"))

else:
    st.info(t("smart_window.load_both"))
    st.markdown(t("smart_window.steps"))
