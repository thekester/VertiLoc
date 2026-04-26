### XGBoost polar 360 benchmark

- Dataset: `['E101']`; samples: `1440` windows.
- Best p80 model: `rf_polar` with p50 `0.204 m`, p80 `0.576 m`, p90 `0.813 m`.
- XGB/RF agreement <= 0.10 m: `0.194`.
- Top features: `signal_minus_a1_abs, signal_minus_a2, signal_minus_a2_abs, signal_minus_a2_sq, orientation_sin_input, a1_minus_a2, signal_A1_median, orientation_cos_input`.
- Note: arbitration smoothing is evaluated on a sorted capture proxy, not a real movement trajectory.
