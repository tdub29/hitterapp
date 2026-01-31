# TORERO BASEBALL HITTER APP

So much of this repo is thanks to the great work of Thomas Nestico, Kyle Bland, Max Bay, for providing examples of models or other essential code in their githubs.

## Project overview

This repo contains a Streamlit dashboard for USD baseball hitter analysis. The app loads pitch-by-pitch data, enriches it with derived metrics (count state, swing/contact flags, zone mapping, pitch categories), and uses pre-trained XGBoost models to estimate expected slugging (xSLG) and decision value for takes/swings. The UI is organized into multiple pages (heatmaps, spray chart, pitch locations, metrics tables, rolling trends) for day-to-day player evaluation and scouting.

Data is pulled from a CSV hosted in GitHub by default (`usd_baseball_TM_master_file.csv`), and the app expects TrackMan-style columns such as batter/pitcher identifiers, pitch location, exit velocity, launch angle, pitch type, and pitch outcomes. You can point the app to a different CSV by updating `file_path` in `streamlit_app.py`.

## What the Streamlit app shows

- **Hitter heatmaps**: Strike-zone grids that visualize xSLG, exit velocity, launch angle, swing rate, and other per-cell metrics.
- **Spray chart**: Batted-ball direction/ distance chart with exit velocity coloring.
- **Pitch locations**: Plots split by play result and by decision value model scores.
- **Hitter metrics table**: Per-batter rolling stats (EV, barrel%, swing/contact rates, zone%, etc.) plus decision values on the 20–80 scale.
- **Zone metrics (Heart/Shadow/Chase/Waste)**: Swing/contact/EV/xSLG and decision value summaries per zone.
- **Rolling trends**: 6-game rolling plots for swing%, chase%, contact%, EV, and other indicators.

## Models and artifacts

- **xSLG model**: `models/xSLG_model.json` uses exit velocity, launch angle, and their interaction to predict xSLG.
- **Decision value models**: `models/model_no_swing.json` and `models/model_swing.json` score takes/swings using plate location and count.
- **League KDE artifacts**: `artifacts/league_kde_earliest.npy`, `artifacts/grid_x.npy`, `artifacts/grid_y.npy` are used for league reference density plots.

## Repository layout

```
.
├── artifacts/          # Precomputed numpy arrays used by the app (KDE grids).
├── configs/            # Configuration placeholders (currently empty).
├── data/               # Data storage (not checked in).
├── docs/               # Reference docs (e.g., D1 baseball reference PDF).
├── models/             # XGBoost model JSON files used by the app.
├── notebooks/          # Reserved for experiments (currently empty).
├── scripts/            # Reserved for batch scripts (currently empty).
├── src/                # Reserved for reusable modules (currently empty).
├── tests/              # Reserved for tests (currently empty).
├── requirements.txt    # Python dependencies.
└── streamlit_app.py    # Streamlit entry point.
```

## Data requirements (expected columns)

The app renames columns into a standard schema. A typical input file should include the following fields (case-insensitive, with spaces stripped):

- **Game metadata**: `GameDate`, `Inning`, `PaOfInning`, `Count` (or `Balls` and `Strikes`)
- **Participants**: `Batter`, `Pitcher`, `BatterSide`, `PitcherThrows`
- **Pitch details**: `PitchTypeFull` (or `AutoPitchType`), `TaggedPitchType`
- **Plate location**: `Px`, `Pz`
- **Outcomes**: `PitchOutcome` (pitch call), `PitchResult` (play result), `PitchUID`
- **Batted-ball data**: `ExitVelocity`, `LaunchAng`, `ExitDir`, `Dist`

If your CSV uses different column names, update the `rename_mapping` dictionary in `streamlit_app.py`.

## Running the app locally

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Start Streamlit:

   ```bash
   streamlit run streamlit_app.py
   ```

The app will download the default CSV from GitHub at runtime. To use a local file, replace `file_path` in `streamlit_app.py` with a local path.
