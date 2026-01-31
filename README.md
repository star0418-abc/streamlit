# GPE Lab - Gel Polymer Electrolyte & Smart Window Analysis Platform

A multi-page Streamlit application for standardizing gel polymer electrolyte (GPE) research workflows and electrochromic smart window analysis.

## Prerequisites

- **Python 3.8+** (3.10 recommended)
- **Anaconda or Miniconda** (recommended for Windows)

## Quick Start

### Option 1: Windows Launcher (Recommended)

1. Edit `run_streamlit.bat` and set your conda environment name
2. Double-click `run_streamlit.bat`

### Option 2: Manual Setup

```bash
# Create and activate conda environment (optional but recommended)
conda create -n gpe_lab python=3.10
conda activate gpe_lab

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Cloud Deployment (Streamlit Community Cloud)

The app is designed to run on Streamlit Community Cloud with these considerations:

### Key Cloud Behaviors

| Feature | Local | Cloud |
|---------|-------|-------|
| Database path | `data/lab.db` | `/tmp/lab.db` |
| Data persistence | Permanent | Ephemeral (resets on restart) |
| File writes | Project folder | `/tmp` only |

### Deploy Steps

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and select `app.py` as entry point
4. Deploy

### Configuration Requirements

- **Do NOT** add `server.port` or `server.address` to `.streamlit/config.toml`
- Keep only UI/theme settings in config.toml if needed
- All dependencies must be in `requirements.txt`

### Troubleshooting Cloud Issues

If you see "Oh no. Error running app." on Cloud:

1. Check the logs in your Streamlit Cloud dashboard
2. Open the app and expand "🔧 环境检查 / Environment Check" for diagnostics
3. Common issues:
   - Missing dependency in `requirements.txt`
   - Import-time errors in pages (check for missing packages)
   - Path issues (don't use hardcoded Windows paths like `D:\`)

## Troubleshooting

### "No module named 'plotly'"

This error occurs when plotly is not installed. Fix with:

```bash
pip install plotly
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

### "No module named 'scipy'"

```bash
pip install scipy
```

### Conda not found (Windows)

1. Open **Anaconda Prompt** instead of regular Command Prompt
2. Or add Anaconda to PATH during installation

### App crashes on startup

1. Check Python version: `python --version` (need 3.8+)
2. Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`
3. Check for error messages in terminal

## Features

### Module A: GPE Electrochem Calculator
- **Conductivity from EIS**: Interactive Nyquist plot with Rb extraction → σ = L/(Rb×S)
- **Temperature Fits**: Arrhenius (Ea in kJ/mol) and VFT with apparent Ea clarification
- **Transference Number**: Bruce-Vincent method with validation warnings
- **Stability Window**: LSV onset detection with configurable thresholds

### Module B: Smart Window Analysis
- **Transmittance Metrics**: ΔT, response time (90%/95%), coloration efficiency
- **Data Alignment**: Automatic CA + transmittance time-series merge
- **Cycle Segmentation**: Voltage-based or peak detection with manual adjustment
- **Cycling Retention**: ΔT and CE vs cycle number plots

### Module C: Lab Database
- **Recipe Management**: Components, ratios, process parameters
- **Batch & Sample Tracking**: Full traceability from recipe to measurement
- **Analytics**: Composition vs property trends, similar recipe search

### 更新报告（Changelog）
- **📋 Update Report**: View version history and recent changes

## Language / 语言

The app defaults to **Chinese (简体中文)** with an **English** toggle available.

### Switching Language

1. Open the app
2. In the sidebar, find the language selector: **语言 / Language**
3. Choose:
   - **中文（默认）** — Chinese (default)
   - **English** — English

The language setting is preserved across all pages during your session.

### Adding or Modifying Translations

Translation files are located in the `i18n/` directory:

```
i18n/
├── zh-CN.json   # Chinese (simplified) - primary language
└── en.json      # English - fallback language
```

#### Key Naming Convention

Keys are organized hierarchically by page/section:

```json
{
  "common": { "app_name": "...", "save_to_db": "..." },
  "sidebar": { "language_label": "..." },
  "home": { "title": "...", "subtitle": "..." },
  "import": { "page_title": "...", "btn_import": "..." },
  "eis": { "page_title": "...", "nyquist_plot": "..." },
  ...
}
```

#### To Add a New Translation Key

1. Add the key to **both** `zh-CN.json` and `en.json` with appropriate values
2. Use in code via: `from utils.i18n import t` then `t("section.key")`

#### Format Strings

Use `{name}` placeholders for dynamic values:

```json
"import_success": "✅ Successfully imported {count} rows"
```

In code:
```python
t("import.import_success", count=100)  # → "✅ Successfully imported 100 rows"
```

#### Fallback Behavior

- If a key is missing in the current language, English is used as fallback
- If missing in both, the raw key is returned and a developer warning is logged

## Project Structure

```
d:\cal\
├── app.py                    # Entry point
├── run_streamlit.bat         # Windows launcher
├── requirements.txt          # Python dependencies
├── pages/                    # Streamlit pages
├── logic/                    # Pure computation modules
├── database/                 # SQLite layer (WAL mode, Cloud-aware)
├── schemas/                  # Canonical DataFrame schemas
├── i18n/                     # Translation files (zh-CN.json, en.json)
├── utils/                    # Utility modules including i18n.py, deps.py
└── data/                     # Database + raw files (local only)
```

## Data Schemas

All imported data is normalized to canonical internal schemas:

| Measurement | Columns | Units |
|-------------|---------|-------|
| EIS | freq, z_re, z_im | Hz, Ω, Ω |
| LSV | e, j | V, mA/cm² |
| CA | t, i, v | s, A, V |
| Transmittance | t, t_frac | s, [0-1] |

## EIS Analysis Notes

### SciPy Optional

The EIS module (`logic/eis.py`) works without SciPy for basic Rb extraction methods:
- `find_hf_intercept_direct()` — Direct HF intercept detection
- `estimate_rb_intercept_linear()` — Linear extrapolation from HF band

Advanced fitting (`fit_simple_rc`) requires SciPy. If missing, it returns `{success: False, error: "SciPy not installed"}`.

### Rb Extraction Assumptions (SS/GPE/SS Cells)

- **HF inductive artifacts** (>100 kHz from lead inductance) are automatically excluded before Rb estimation
- **Sign convention**: Internal computation uses capacitive-negative Z_im; Nyquist plots display -Z_im (positive upper half)
- **Frequency-based HF selection**: Uses top 1 decade (freq ≥ fmax/10), not point-count fractions

### RC Fitting Caveats

`fit_simple_rc()` fits ONLY the HF semicircle (Rs + R||C). It is NOT valid for:
- Full-spectrum Randles+Warburg analysis
- Diffusion-dominated spectra (blocking electrodes)

For quantitative equivalent circuit analysis, use specialized software (ZView, Relaxis, EC-Lab Z Fit).

## LSV Analysis Notes

### SciPy Optional

LSV smoothing (`logic/lsv.py`) uses Savitzky-Golay filter from SciPy. If SciPy is missing:
- Smoothing degrades to numpy moving-average (still functional)
- A warning is included in the result dict

### Onset Detection Methods

| Method | Parameter | Best For |
|--------|-----------|----------|
| `threshold` | `onset_method="threshold"` | Clean data with sharp faradaic onset |
| `tangent` | `onset_method="tangent"` | GPE with baseline drift/current creep |

**Threshold method** (default):
- Detects when |j| exceeds `threshold_ma_cm2`
- Requires `min_consecutive=3` points above threshold (avoids single-spike noise)
- Applies linear interpolation for sub-datapoint precision

**Tangent method** (recommended for GPE):
- Fits constant/linear baseline from first 10% of sweep (median-based, robust)
- Identifies rising region via max derivative on smoothed current
- Fits tangent line around max derivative
- Onset = intersection of baseline and tangent lines

### CV Auto-Segmentation

If the potential array is non-monotonic (e.g., CV data uploaded instead of LSV):
- The longest monotonic segment is automatically extracted
- A warning is included describing what was done
- The data is NOT sorted (preserves time order for correct onset logic)

### Baseline Correction

The baseline is computed from the **first portion** of the sweep (first 10% or 20 points, whichever larger):
- Default: constant baseline = median(j) in baseline region
- Linear baseline used only if R² > 0.7 and doesn't create large negative artifacts
- This is safer than the old global low-current mask approach for GPE data

### Direction/Sign Sanity

- `direction="oxidation"` expects positive current rise
- `direction="reduction"` expects negative current drop
- If the data doesn't match the expected sign, the function warns and returns `onset_v=None` (safe default)

## Transference Number (tLi+) Notes

### Bruce-Vincent Method

The module (`logic/transference.py`) implements the Bruce-Vincent/Evans method:

```
tLi+ = Iss × (ΔV - I0×R0) / [I0 × (ΔV - Iss×Rss)]
```

This assumes **small polarization** in the linear regime (typically ΔV ≤ 10-20 mV).

### Strict vs Lenient Mode

| Parameter | `strict=True` (default) | `strict=False` |
|-----------|------------------------|----------------|
| Invalid ΔV (≤0) | `success=False` | `success=False` |
| Large ΔV (>100 mV) | `success=False` | `qc_pass=False` + warning |
| Effective voltage ≤0 | `success=False` | `qc_pass=False`, `t_li_plus=None` |
| Invalid R (≤0) | `success=False` | `qc_pass=False`, `t_li_plus=None` |

### Current Sign Convention

Instruments often export negative current by convention. The formula requires magnitudes:
- `I0_eff = abs(I0)`, `Iss_eff = abs(Iss)`
- Raw values preserved in output: `I0_raw_A`, `Iss_raw_A`
- Flag `current_abs_applied=True` indicates sign correction occurred

### Effective Polarization Voltage

The corrected driving voltages are:
- `dV_eff0 = ΔV - I0×R0` (at t=0)
- `dV_effss = ΔV - Iss×Rss` (at steady-state)

If either is ≤0, the computation is physically invalid (IR drop exceeds applied voltage).

### I0 Extraction (Capacitive Transient Handling)

The initial current I0 is extracted **after** capacitive settling, not from the first 1% of time blindly:

1. **Transient detection** (`transient_mode="auto"`):
   - Computes `|dI/dt|` on smoothed current
   - Finds earliest point where derivative stays small for 5 consecutive readings
   - Ignores this initial capacitive spike region

2. **I0 window**: First 1% of time after transient, minimum 5 points
3. **I0 = median(|I|)** in window (robust to outliers)

Metadata returned: `transient_ignored_s`, `i0_n_points`

### Iss Extraction (Steady-State Detection)

Steady-state current Iss is validated, not assumed from "last 10%":

1. **Tail region**: Last 20% of time (configurable)
2. **Steady detection**: Finds segment where `|dI/dt|` is consistently small (≥10 consecutive points)
3. If steady found: `ss_detected=True`, `Iss = median` of that segment
4. If NOT found: `ss_detected=False`, `qc_pass=False`, uses last-window median with warning

### QC Flags

The `qc_flags` list contains structured issue identifiers:

| Flag | Meaning |
|------|---------|
| `deltaV_too_large` | ΔV > 100 mV (non-linear regime) |
| `effective_voltage_nonpositive` | IR drop exceeds ΔV |
| `iss_ge_i0` | Iss ≥ I0 (unusual, not at steady-state) |
| `t_negative` | tLi+ < 0 (physics error) |
| `t_above_unity` | tLi+ > 1 (parasitic reactions or not steady) |
| `denominator_near_zero` | Numerical instability |

### When `t_li_plus=None`

The function returns `t_li_plus=None` (and `success=False`) when:
- ΔV ≤ 0
- R0 or Rss ≤ 0
- Effective voltage (ΔV - I×R) ≤ 0
- Denominator near zero

In lenient mode (`strict=False`), large ΔV still attempts computation but sets `qc_pass=False`.

### Recommended Experimental Constraints

For reliable Bruce-Vincent measurements:
- **Small ΔV**: 10 mV typical, warn above 20 mV, reject above 100 mV
- **Sufficient polarization time**: Until dI/dt ≈ 0 (check `ss_detected` flag)
- **Symmetric cells**: Li/GPE/Li or equivalent blocking configuration
- **Matched EIS**: R0 from EIS before polarization, Rss from EIS after

## Temperature Fits Notes

### Why Log-Space Fitting (VFT)

The VFT equation is fitted in log-space: `ln(σ) = ln(A) - B/(T - T₀)`.

**Problem with linear-scale fitting**: When fitting `σ = A × exp(-B/(T-T₀))` directly, high-temperature data points with large σ values dominate the residuals, biasing T₀ and B. Low-temperature points (critical for T₀ determination) are effectively ignored.

**Solution**: Log-space fitting gives equal **relative** weight to all data points, which is appropriate for conductivity spanning orders of magnitude.

### Why AICc/BIC Instead of R²

The previous approach comparing `vft_r2 - arr_r2 > 0.02` is biased:

| Metric | Arrhenius (k=2) | VFT (k=3) | Issue |
|--------|-----------------|-----------|-------|
| R² | Lower | Higher | VFT always wins with more params |
| AICc | Fair | Fair | Penalizes extra parameter |
| BIC | Fair | Fair | Stronger penalty for complexity |

**Recommendation rules (ΔAICc = VFT - Arrhenius)**:
- |Δ| ≤ 2: Inconclusive (use simpler Arrhenius)
- 2-4: Weak evidence
- 4-7: Moderate evidence
- >10: Strong evidence

### Temperature Unit Handling

| Input Detected | Action |
|----------------|--------|
| median(T) < 150 AND max(T) < 250 | Auto-convert °C → K with warning |
| Any T < 0 | Auto-convert °C → K with warning |
| Otherwise | Assume Kelvin |

Override with `temp_unit="K"` or `temp_unit="C"` to skip auto-detection.

### VFT Prefactor Options

| Option | Equation | Use Case |
|--------|----------|----------|
| `standard` | ln(σ) = ln(A) - B/(T-T₀) | Default, most common |
| `T^-1` | ln(σ) = ln(A) - ln(T) - B/(T-T₀) | Theoretical models |
| `T^-0.5` | ln(σ) = ln(A) - 0.5·ln(T) - B/(T-T₀) | Some polymer systems |

All three have the same number of fitted parameters (k=3).

### SciPy Dependency

| Function | SciPy Required | Notes |
|----------|----------------|-------|
| `arrhenius_fit()` | No | Pure numpy linear fit |
| `vft_fit()` | Yes | Nonlinear curve_fit |
| `compare_fits()` | Partial | Arrhenius works; VFT returns `success=False` |

On Streamlit Cloud without SciPy: Arrhenius fully functional, VFT returns clear error message.

### QC Flags

| Flag | Meaning |
|------|---------|
| `too_few_points_for_model` | n ≤ k+1 (AICc invalid) |
| `narrow_temperature_span` | ΔT < 30 K (inconclusive fits) |
| `temp_unit_auto_converted` | Celsius was auto-detected |
| `T0_close_to_Tmin` | Tmin - T₀ < 10 K (unstable VFT) |
| `vft_overfit_risk` | n ≤ 4 for VFT |

## Traceability

Every computed result stores:
- Raw file path + SHA256 hash
- Import column mapping
- All parameter values (L, S, thresholds, etc.)
- Software version
- Computed metrics + plot references

## 更新报告 / Changelog

The app includes a changelog page (**📋 更新报告**) showing version history.

### Adding New Entries

Edit `data/changelog_zh.md` and append entries in this format:

```markdown
## YYYY-MM-DD vX.Y.Z

### 新增功能
- **功能名称**：功能描述

### 修复
- 修复内容描述

### 改进
- 改进内容描述
```

### Entry Guidelines

- Use Chinese for all content
- Date format: `YYYY-MM-DD`
- Version format: `vX.Y.Z` (semantic versioning recommended)
- Organize changes by category: 新增功能 / 修复 / 改进 / 移除

### Cloud Behavior

On Streamlit Cloud, the changelog file is **read-only** (bundled with the repo).  
To update the changelog on Cloud, commit changes to `data/changelog_zh.md` and redeploy.

## Implementation Status

- [x] **V0 (MVP)**: Import layer, manual Rb, basic metrics, SQLite saving
- [x] **i18n**: Chinese/English language support with sidebar toggle
- [x] **Dependency handling**: Graceful error messages for missing packages
- [x] **Cloud-ready**: Lazy DB init, /tmp path on Cloud, diagnostics panel
- [x] **Changelog**: Version update report page
- [ ] **V1**: Semi-auto Rb, temperature fits, cycle segmentation
- [ ] **V2**: Analytics dashboards, similar recipe search

## License

Internal lab use.
