# 2D RSSI-based Localization of a WiFi IoT Sensor

This repository contains the first R&D campaign led by Mohamad El Zoghbi to localize an ESP32 sensor mounted on a metallic board. We collected RSSI fingerprints for two router placements (2 m and 4 m away from the board) and used them to train a **neural network encoder** followed by a **local KNN** classifier operating on 2D cells (G-cells).

## Repository layout
- `ddeuxmetres/`, `dquatremetres/`: raw fingerprints per cell (34 points each) plus the acquisition script `collect_wifi.sh`.
- `contexte.md`, `lesgridcells.png`: physical dimensions of the board and the grid layout.
- `docs/bibliographie.md`: short literature survey.
- `docs/methodologie_maillage.md`: description of the grid, measurement protocol, and preprocessing steps.
- `src/localization/`: Python sources (data loading utilities, NN+L-KNN implementation, CLI pipeline).
- `reports/`: generated artifacts (`latest_metrics.json`, `embedding_pca.png`, etc.).

## Quick installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional extra dependencies (for the full benchmark sweep, CatBoost/LightGBM/XGBoost):
```bash
pip install catboost lightgbm xgboost
```

## Running the NN + L-KNN pipeline
From the repository root, expose `src/` on `PYTHONPATH` (or install the project in editable mode) before invoking the module:
```bash
PYTHONPATH=src python -m localization.pipeline \
  --campaign ddeuxmetres:2 \
  --campaign dquatremetres:4 \
  --hidden-layers 64 32 \
  --k-neighbors 5
```
Useful options:
- `--cell-width-m`, `--cell-height-m` to change the physical size of each cell.
- `--test-size` to adjust the validation split ratio.
- `--campaign folder:distance` to add/remove measurement campaigns.
- `--explain-k`, `--model-output`, `--confusion-matrix`, `--roc-curve`, `--predictions-report` to control explainability depth and artifact locations.

The pipeline produces:
- `reports/latest_metrics.json`: accuracy, mean errors, average confidence, micro/macro ROC AUC, router-distance accuracy (campagne 2 m vs 4 m) et baseline sans LogisticRegression.
- `reports/embedding_pca.png`: 2D projection of the embeddings to visually inspect cluster separation.
- `reports/confusion_matrix.png` and `reports/roc_micro_macro.png`: confusion matrix and ROC curves.
- `reports/confusion_cell_distance_with_logreg.png`: confusion matrix sur le couple (cellule, distance routeur prédite) avec la tête LogisticRegression.
- `reports/confusion_cell_distance_without_logreg.png`: même matrice mais en utilisant une baseline (distance = mode observée pour la cellule prédite).
- `reports/predictions.csv`: per-sample explainability file (true/pred cell, error, confidence, nearest neighbors, predicted router distance + confidence). Les colonnes `neighbor_i_embedding_distance` sont les distances dans l'espace latent (pas des distances physiques) : elles n'ont rien à voir avec la distance routeur (2 m ou 4 m) prédite séparément.
- `reports/localizer.joblib`: serialized model ready for offline inference.

The exported inference run name is `vertiloc-beacon-v1` by default. Override it at training time with:
```bash
PYTHONPATH=src python -m localization.pipeline \
  --campaign ddeuxmetres:2 \
  --campaign dquatremetres:4 \
  --run-name vertiloc-beacon-v2
```

## Continuous Integration
GitHub Actions runs fast checks on every push/PR (`.github/workflows/ci.yml`):
1. `python -m unittest discover -s tests` validates shared constants, catalog filtering, data loading, geometry, and orientation helpers.
2. `python -m localization.pipeline ...` trains/evaluates a bounded CLI smoke run.
3. `python scripts/notebook_smoke.py` replays the main notebook path (split, train, predict, explain).
4. `python scripts/benchmark_models.py ...` runs a reduced benchmark smoke on D005 only.
5. `python scripts/triscope_eval.py --iterations 5 --sample-size 10` checks the multi-head path without the full 50-draw run.
6. `python scripts/query_vertiloc.py ...` verifies the inference CLI entrypoint.

## Querying VertiLoc with custom RSSI readings
In practice, to run inference you call the CLI script `scripts/query_vertiloc.py`.
It loads the trained artifact `reports/localizer.joblib` through `localization.VertiLocInferenceModel`.

After training (`python -m localization.pipeline ...`), you can ask the model to localize a custom RSSI vector using:
```bash
PYTHONPATH=src python scripts/query_vertiloc.py \
  -42 -95 -44 -41 -43
# or
PYTHONPATH=src python scripts/query_vertiloc.py --vector "-42,-95,-44,-41,-43"
```
Arguments correspond to `[Signal, Noise, signal_A1, signal_A2, signal_A3]`.  
The script loads `reports/localizer.joblib` through `localization.VertiLocInferenceModel`, then predicts:
- the most likely `grid_cell`;
- the inferred router distance;
- a board-relative macro-zone (`top_left`, `middle_center`, etc.);
- the clamped board coordinates and OOD indicators.

For machine-readable output:
```bash
PYTHONPATH=src python scripts/query_vertiloc.py \
  --vector "-42,-95,-44,-41,-43" \
  --format json
```

To display usage examples directly from the CLI:
```bash
PYTHONPATH=src python3 scripts/query_vertiloc.py --example
```

For real-condition batch testing from a CSV:
```bash
PYTHONPATH=src python scripts/query_vertiloc.py \
  --input-csv reports/benchmarks/query_batch_example.csv \
  --output-csv reports/benchmarks/query_batch_predictions.csv \
  --output-json reports/benchmarks/query_batch_predictions.json
```
The input CSV must contain the columns:
`Signal`, `Noise`, `signal_A1`, `signal_A2`, `signal_A3`.

## Current results (80/20 split)
```
accuracy ≈ 99.7 %
mean distance error ≈ 1.8 mm (correct cell recovered in 99.7% of the cases)
```
These numbers show that the 32-d MLP encoder followed by a distance-weighted L-KNN (k=5) is sufficient to recover the correct cell even with the limited RSSI variability of the experiment.

How to investigate residual errors and confidence:
- inspect `reports/predictions.csv` (`confidence`, `confidence_margin`, `neighbor_i_cell/distance` columns);
- open `reports/confusion_matrix.png` to locate recurrent confusions;
- review `reports/roc_micro_macro.png` and the AUC entries in `latest_metrics.json` to ensure multi-class separation.

Reloading the trained model:
```python
import joblib
localizer = joblib.load("reports/localizer.joblib")
proba = localizer.predict_proba(new_feature_array)
```

Using the dedicated inference wrapper:
```python
from localization import VertiLocInferenceModel

model = VertiLocInferenceModel.load(
    model_path="reports/localizer.joblib",
    metrics_path="reports/latest_metrics.json",
)
predictions = model.predict_dataframe(
    df_with_signal_noise_columns,
    top_k=3,
    room="E102",
)
```

## Benchmarks and optional models
Run the full benchmark suite (room-agnostic LORO, room-aware split, router distance/room heads, stacking):
```bash
python3 scripts/benchmark_models.py
```
- Results are written to `reports/benchmarks/benchmark_summary.csv`, with confusions in the same folder.
- Any stage that is skipped (e.g., GPC for speed, missing CatBoost/LightGBM/XGBoost) or fails will be recorded with its reason in `reports/benchmarks/benchmark_failures.json`.
- Install `catboost`, `lightgbm`, and `xgboost` to fill the corresponding rows instead of `nan`/skipped.

### Circular orientation model
To train an orientation model that respects circular geometry (`back=0°`, `right=90°`, `front=180°`, `left=270°`, `back_right=45°`, etc.), use:
```bash
PYTHONPATH=src python3 scripts/train_orientation_model.py --datasets E102 --exclude-e102-elevation
```
By default, the script trains a classifier on the orientation modes, then reconstructs the angle through a circular mean of class probabilities. The regression variant remains available with `--approach circular_regression`. It writes:
- `reports/benchmarks/orientation_circular_model*.joblib`: serialized model bundle.
- `reports/benchmarks/orientation_circular_metrics*.json`: angular error + snapped-label accuracy.
- `reports/benchmarks/orientation_circular_predictions*.csv`: per-sample true/predicted angle and angular error.

Available orientation labels and degrees:
- `back` = `0`
- `back_right` = `45`
- `right` = `90`
- `front_right` = `135`
- `front` = `180`
- `front_left` = `225`
- `left` = `270`
- `back_left` = `315`

### XGBoost polar 360 experiment
To test the 360-degree feature-engineering path with one-second capture windows,
three polar regression heads (`radius`, `sin(theta)`, `cos(theta)`), and XGBoost/RF
cell-classifier baselines:
```bash
PYTHONPATH=src ./.venv/bin/python scripts/xgb_polar_360.py \
  --datasets E101,E102 \
  --n-estimators 400 --max-depth 5 --learning-rate 0.04 \
  --output-prefix xgb_polar_360_e101_e102_xgb400
```
Outputs are written under `reports/benchmarks/`:
- `*_metrics.csv`: p50/p80/p90 errors, cell accuracy, lateral/depth errors;
- `*_predictions.csv`: per-window predictions;
- `*_feature_importance.csv`: feature importances for XGBoost/RF heads;
- `*_xgb_cell_confusion_matrix.csv`: cell confusion matrix for the XGBoost classifier.

### Embedded/edge profiling (RAM/flash/warm-up/energy)
Use the dedicated profiler to compare deployment cost per model:
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=src python3 scripts/embedded_profile.py \
  --models rf,et,mlp \
  --room E102 --distances 4 \
  --max-train-samples 1200 --max-test-samples 500 \
  --warmup-runs 20 --repeats 8 --batch-size 32 \
  --host-power-w 6.0 \
  --esp32-current-ma 95 --esp32-inference-ms 12 \
  --output-prefix embedded_profile_e102_quick
```
Outputs:
- `reports/benchmarks/embedded_profile_e102_quick.csv`
- `reports/benchmarks/embedded_profile_e102_quick.json`

The script reports model footprint (disk/flash proxy), cold start, warm-up,
latency percentiles, throughput, and host/ESP32 energy per inference.

## Hybrid physical + ML (Sionna RT-oriented path)
To try an EM-aware bridge between a physical model and data-driven calibration:
```bash
PYTHONPATH=src python scripts/sionna_rt_hybrid.py --protocol random --backend sionna \
  --output reports/benchmarks/sionna_hybrid_metrics_random_sionna_customscene.json
```
LOCO across E102 campaigns:
```bash
PYTHONPATH=src python scripts/sionna_rt_hybrid.py --protocol loco_e102 --backend sionna \
  --output reports/benchmarks/sionna_hybrid_metrics_loco_e102_sionna.json
```
The script reports 3 model families:
- `data_driven_centroid`: nearest measured RSSI centroid (baseline);
- `physical_only`: physical proxy from selected backend (`sionna` or `analytic`);
- `hybrid_physical_ml`: physical prototypes calibrated via Ridge.

With backend `sionna`, the script now builds a custom E102 classroom scene
(walls, whiteboard, desk rows, and chair/table obstacles inferred from
`data/E102/e102_pic`) and varies obstacle layout by campaign+line.

Each JSON includes:
- per-model cell accuracy + mean/p90 localization error;
- deltas versus baseline and versus physical-only;
- protocol metadata (`random` split or `loco_e102` folds);
- backend metadata (`backend_requested`, `backend_effective`, `sionna_rt` import status).

Optional Sionna RT availability check:
```bash
PYTHONPATH=src python scripts/sionna_rt_hybrid.py --protocol random --backend auto --use-sionna
```
If Sionna is not installed, the benchmark still runs with the analytic physical proxy and reports the import error in JSON (`sionna_rt.reason`).

Optional install command for Sionna ecosystems (CUDA/TensorFlow compatibility required):
```bash
pip install sionna mitsuba drjit
```

## Suggested next steps
1. Increase dataset diversity (different router heights, obstacles, ESP32 orientation) to stress-test the embeddings.
2. Experiment with metric-learning objectives (contrastive/triplet loss) to enhance geometric consistency and reduce the number of fingerprints needed per cell.
3. Extend L-KNN to perform fine-grained interpolation inside a cell (e.g., weighted barycenter of the neighbors) for smoother localization.
