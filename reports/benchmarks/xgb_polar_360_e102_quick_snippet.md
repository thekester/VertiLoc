### XGBoost polar 360 benchmark

- Dataset: `['E102']`; samples: `1500` windows.
- Best p80 model: `rf_polar` with p50 `0.277 m`, p80 `0.697 m`, p90 `1.297 m`.
- XGB/RF agreement <= 0.10 m: `0.087`.
- Top features: `Signal_min, ant_median_max, a1_over_a3_mw, signal_A2_max, signal_A3_mean, orientation_cos_input, a1_minus_a3_abs, signal_A3_min`.
- Note: arbitration smoothing is evaluated on a sorted capture proxy, not a real movement trajectory.
