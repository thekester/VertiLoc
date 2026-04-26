### XGBoost polar 360 benchmark

- Dataset: `['E101']`; samples: `1440` windows.
- Best p80 model: `rf_cell_classifier` with p50 `0.000 m`, p80 `0.434 m`, p90 `0.988 m`.
- XGB/RF agreement <= 0.10 m: `0.500`.
- Top features: `signal_minus_a2_sq, signal_minus_a2_abs, signal_A1_median_log_abs, signal_A2_median_log_abs, orientation_sin_input, signal_minus_a2, signal_minus_a1_abs, a1_minus_a2`.
- Note: arbitration smoothing is evaluated on a sorted capture proxy, not a real movement trajectory.
