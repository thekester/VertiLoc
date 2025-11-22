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
- `reports/latest_metrics.json`: accuracy, mean errors, average confidence, micro/macro ROC AUC.
- `reports/embedding_pca.png`: 2D projection of the embeddings to visually inspect cluster separation.
- `reports/confusion_matrix.png` and `reports/roc_micro_macro.png`: confusion matrix and ROC curves.
- `reports/predictions.csv`: per-sample explainability file (true/pred cell, error, confidence, nearest neighbors).
- `reports/localizer.joblib`: serialized model ready for offline inference.

## Continuous Integration
GitHub Actions runs two smoke tests on every push/PR (`.github/workflows/ci.yml`):
1. `python -m localization.pipeline ...` trains/evaluates on both campaigns to validate the CLI pipeline.
2. `python scripts/notebook_smoke.py` replays the main steps from the notebook (split, train, predict, explain) to guarantee that the tutorial/code samples keep working.
3. `python scripts/query_vertiloc.py ...` runs twice (one 2 m sample, one 4 m sample) to confirm the query CLI returns a plausible cell prediction and logs the top-K neighbors.

## Querying VertiLoc with custom RSSI readings
After training (`python -m localization.pipeline ...`), you can ask the model to localize a custom RSSI vector using:
```bash
PYTHONPATH=src python scripts/query_vertiloc.py \
  -42 -95 -44 -41 -43
# or
PYTHONPATH=src python scripts/query_vertiloc.py --vector "-42,-95,-44,-41,-43"
```
Arguments correspond to `[Signal, Noise, signal_A1, signal_A2, signal_A3]`.  
The script loads `reports/localizer.joblib`, prints the predicted `grid_cell`, and lists the top-K neighbors used in the vote.

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

## Suggested next steps
1. Increase dataset diversity (different router heights, obstacles, ESP32 orientation) to stress-test the embeddings.
2. Experiment with metric-learning objectives (contrastive/triplet loss) to enhance geometric consistency and reduce the number of fingerprints needed per cell.
3. Extend L-KNN to perform fine-grained interpolation inside a cell (e.g., weighted barycenter of the neighbors) for smoother localization.
