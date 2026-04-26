### XGBoost polar 360 benchmark

- Dataset: `['E101', 'E102']`; samples: `2940` windows.
- Best p80 model: `xgb_cell_classifier` with p50 `0.000 m`, p80 `0.590 m`, p90 `1.494 m`.
- XGB/RF agreement <= 0.10 m: `0.221`.
- Top features: `Signal_min, signal_minus_a3_sq, signal_A2_median_log_abs, noise_minus_signal, a1_minus_a3_sq, ant_median_max, signal_minus_a1_abs, signal_minus_a3`.
- Note: arbitration smoothing is evaluated on a sorted capture proxy, not a real movement trajectory.
