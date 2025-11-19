# Vertical board grid and measurement protocol

## Surface and cells
- Metallic board **2.35 m wide** by **1.15 m high**.
- Physical grid implemented with 3D-printed magnetic brackets as illustrated in `lesgridcells.png`.
- Each **G-cell** is **30 cm high** (`grid_x = 0…3`) and **25 cm wide** (`grid_y = 0…8`, with two unused columns on rows `1` and `3`).
- Center of a cell `(x, y)`:
  - `x_m = y * 0.25 + 0.125` (m)
  - `y_m = x * 0.30 + 0.15` (m)

## Radio setup
- **TP-Link Archer C7 v2** router running OpenWrt (interface `phy1-ap0`).
- **ESP32** probe placed at the center of each cell with a magnetic holder.
- `collect_wifi.sh` captures **25 samples** (`5 s` at `5 Hz`) and stores `Signal`, `Noise`, and per-antenna RSSI `A1/A2/A3` into a CSV named `X_Y.csv`.

## Measurement campaigns
| Folder          | Router-to-board distance | # Cells | Files | Samples |
|-----------------|-------------------------|--------:|------:|--------:|
| `ddeuxmetres`   | 2 m                      | 34      | 34    | 850     |
| `dquatremetres` | 4 m                      | 34      | 34    | 850     |

Total: **1,700 labelled measurements** across the board. Original ZIP archives (`ddeuxmetres.zip`, `dquatremetres.zip`) are preserved.

## Pre-processing
1. Load CSV files and enrich with metadata (`grid_x`, `grid_y`, `grid_cell`, `router_distance_m`, physical coordinates).
2. Model inputs: `Signal`, `Noise`, `signal_A1`, `signal_A2`, `signal_A3`, `router_distance_m`.
3. Physical coordinates (meters) are used for distance evaluation and embedding visualization.
