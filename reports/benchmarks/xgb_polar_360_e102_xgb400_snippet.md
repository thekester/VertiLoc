### XGBoost polar 360 benchmark

- Dataset: `['E102']`; samples: `1500` windows.
- Best p80 model: `xgb_cell_classifier` with p50 `0.000 m`, p80 `0.000 m`, p90 `1.056 m`.
- XGB/RF agreement <= 0.10 m: `0.353`.
- Top features: `Signal_min, ant_median_max, a1_minus_a3_sq, signal_A3_min, signal_A1_median_mw, a1_minus_a3_abs, signal_A2_max, a1_over_a2_mw`.
- Note: arbitration smoothing is evaluated on a sorted capture proxy, not a real movement trajectory.
