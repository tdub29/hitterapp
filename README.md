# TORERO BASEBALL HITTER APP

So much of this repo is thanks to the great work of Thomas Nestico, Kyle Bland, Max Bay, for providing examples of models or other essential code in their githubs

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

## Repository structure

```
.
├── artifacts/          # Precomputed numpy arrays or other derived artifacts used at runtime.
├── configs/            # Configuration files (YAML/TOML/JSON) and environment templates.
├── data/
│   ├── external/       # Third-party or vendor data sources (read-only).
│   ├── processed/      # Cleaned/feature-ready data derived from raw inputs.
│   └── raw/            # Immutable raw data pulls/snapshots (never edit in place).
├── docs/               # Project documentation and reference materials.
├── models/             # Serialized model artifacts (JSON, pickle, ONNX).
├── notebooks/          # Exploratory analyses and experiments (kept out of src).
├── scripts/            # One-off or batch scripts for ingestion/ETL/automation.
├── src/                # Reusable library code (ingestion, features, modeling).
├── tests/              # Unit/integration tests for reproducibility and CI.
├── requirements.txt    # Python dependencies.
└── streamlit_app.py     # App entry point (kept at repo root for Streamlit).
```

### Top-level directory contents

- **artifacts/**: Runtime assets (e.g., grids, KDEs) generated from data/modeling steps so the app can load them quickly.
- **configs/**: Centralized configuration for environments (local, staging, prod) without hard-coding in code.
- **data/**: Clear separation of raw, processed, and external data to enforce reproducibility and auditability.
- **docs/**: Design notes, data dictionaries, or references to help onboarding and collaboration.
- **models/**: Versioned model artifacts tied to training runs and evaluation results.
- **notebooks/**: Exploratory work stays isolated from production code but is still tracked.
- **scripts/**: CLI or batch utilities for ingestion, feature building, training, evaluation, deployment tasks.
- **src/**: Reusable modules so pipelines can be tested and imported (e.g., `src/ingestion`, `src/features`).
- **tests/**: Ensures model/data pipelines and app utilities remain stable.

### Naming conventions

- **Files**: `snake_case` for data/artifact filenames; `PascalCase` only for class names.
- **Data snapshots**: `source_YYYYMMDD.csv` or `source_vN.csv` for deterministic lineage.
- **Models**: `model_name__metric__date.json` (e.g., `xslg__rmse_0.12__20240201.json`).
- **Notebooks**: `NN__short_topic__author.ipynb` to keep ordering and attribution.

### README content

- **Project overview** (goal, data sources, expected outputs).
- **Quickstart** (env setup, data requirements, run commands).
- **Data lineage** (raw → processed → features → model artifacts).
- **Modeling approach** (features, algorithms, evaluation metrics).
- **Deployment** (how to run Streamlit, environment variables).
- **Contributing** (tests, style, review process).

### Configuration

- Use a single config file per environment in `configs/` (e.g., `configs/local.yaml`).
- Keep secrets out of git; rely on environment variables or a secrets manager.
- Prefer reading configs in code to make paths and thresholds consistent across notebooks, scripts, and the app.

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
