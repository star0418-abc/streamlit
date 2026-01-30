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

## Traceability

Every computed result stores:
- Raw file path + SHA256 hash
- Import column mapping
- All parameter values (L, S, thresholds, etc.)
- Software version
- Computed metrics + plot references

## Implementation Status

- [x] **V0 (MVP)**: Import layer, manual Rb, basic metrics, SQLite saving
- [x] **i18n**: Chinese/English language support with sidebar toggle
- [x] **Dependency handling**: Graceful error messages for missing packages
- [x] **Cloud-ready**: Lazy DB init, /tmp path on Cloud, diagnostics panel
- [ ] **V1**: Semi-auto Rb, temperature fits, cycle segmentation
- [ ] **V2**: Analytics dashboards, similar recipe search

## License

Internal lab use.
