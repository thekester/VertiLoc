#!/usr/bin/env python3
"""Hybrid physical + ML benchmark toward a Sionna RT workflow.

What this script evaluates:
1) `data_driven_centroid`: nearest measured RSSI centroid per cell (baseline).
2) `physical_only`: physical proxy predicted from campaign geometry/context.
3) `hybrid_physical_ml`: physical prototypes calibrated by Ridge on measured RSSI.

Protocols:
- `random`: stratified train/test split.
- `loco_e102`: leave-one-campaign-out over E102 exp1..exp6.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from localization.data import CampaignSpec, load_measurements  # noqa: E402

FEATURE_COLUMNS = ["Signal", "Noise", "signal_A1", "signal_A2", "signal_A3"]
DEFAULT_E102_CAMPAIGNS = [
    CampaignSpec(PROJECT_ROOT / "data" / "E102" / "exp1", router_distance_m=4.0),
    CampaignSpec(PROJECT_ROOT / "data" / "E102" / "exp2", router_distance_m=4.0),
    CampaignSpec(PROJECT_ROOT / "data" / "E102" / "exp3", router_distance_m=4.0),
    CampaignSpec(PROJECT_ROOT / "data" / "E102" / "exp4", router_distance_m=4.0),
    CampaignSpec(PROJECT_ROOT / "data" / "E102" / "elevation" / "exp5", router_distance_m=4.0),
    CampaignSpec(PROJECT_ROOT / "data" / "E102" / "elevation" / "exp6", router_distance_m=4.0),
]
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "benchmarks" / "sionna_hybrid_metrics.json"


def parse_campaign_args(entries: Iterable[str]) -> list[CampaignSpec]:
    specs: list[CampaignSpec] = []
    for entry in entries:
        if ":" in entry:
            folder, distance = entry.split(":", 1)
            specs.append(CampaignSpec(Path(folder), float(distance)))
        else:
            specs.append(CampaignSpec(Path(entry)))
    return specs or DEFAULT_E102_CAMPAIGNS


def try_import_sionna(use_sionna: bool) -> dict[str, object]:
    if not use_sionna:
        return {"requested": False, "available": False, "reason": "disabled_by_flag"}
    try:
        import sionna  # type: ignore
        import sionna.rt  # type: ignore # noqa: F401

        return {
            "requested": True,
            "available": True,
            "version": getattr(sionna, "__version__", "unknown"),
            "reason": "import_ok",
        }
    except Exception as exc:  # pragma: no cover - environment-dependent
        return {"requested": True, "available": False, "reason": f"import_failed: {exc}"}


def _annotate_campaign_context(df: pd.DataFrame, default_router_height_m: float) -> pd.DataFrame:
    out = df.copy()
    campaign = out["campaign"].astype(str).str.lower()

    is_exp1 = campaign.str.contains("exp1")
    is_exp2 = campaign.str.contains("exp2")
    is_exp3 = campaign.str.contains("exp3")
    is_exp4 = campaign.str.contains("exp4")
    is_exp5 = campaign.str.contains("exp5")
    is_exp6 = campaign.str.contains("exp6")

    out["ctx_orient_right"] = (is_exp1 | is_exp2).astype(float)
    out["ctx_orient_left"] = (is_exp3 | is_exp4).astype(float)
    out["ctx_orient_front"] = (is_exp2 | is_exp3 | is_exp5 | is_exp6).astype(float)
    out["ctx_orient_back"] = (is_exp1 | is_exp4).astype(float)
    out["ctx_orientation_lr"] = out["ctx_orient_right"] - out["ctx_orient_left"]
    out["ctx_orientation_fb"] = out["ctx_orient_front"] - out["ctx_orient_back"]

    heights = np.full(len(out), float(default_router_height_m), dtype=float)
    heights[is_exp5.to_numpy()] = 0.0
    heights[is_exp6.to_numpy()] = 1.5
    out["ctx_router_height_m"] = heights

    out["ctx_is_ground"] = is_exp5.astype(float)
    out["ctx_is_elevated"] = is_exp6.astype(float)
    return out


class PhysicalBackendProtocol:
    def fit(self, train_df: pd.DataFrame) -> "PhysicalBackendProtocol":
        return self

    def predict(self, candidates_df: pd.DataFrame) -> np.ndarray:  # pragma: no cover - protocol
        raise NotImplementedError


@dataclass
class AnalyticPhysicsProxyModel(PhysicalBackendProtocol):
    """Analytic proxy for RSSI based on geometry + campaign context."""

    room_depth_m: float = 4.0
    coef_: np.ndarray | None = None
    x_center_: float | None = None

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        if self.x_center_ is None:
            raise RuntimeError("AnalyticPhysicsProxyModel is not initialized.")

        coord_x = df["coord_x_m"].to_numpy(dtype=float)
        coord_y = df["coord_y_m"].to_numpy(dtype=float)
        rd = df["router_distance_m"].to_numpy(dtype=float)
        h = df["ctx_router_height_m"].to_numpy(dtype=float)
        orient_lr = df["ctx_orientation_lr"].to_numpy(dtype=float)
        orient_fb = df["ctx_orientation_fb"].to_numpy(dtype=float)
        is_ground = df["ctx_is_ground"].to_numpy(dtype=float)
        is_elevated = df["ctx_is_elevated"].to_numpy(dtype=float)

        x_term = coord_x - self.x_center_
        y_term = coord_y + rd
        d = np.sqrt(x_term**2 + y_term**2 + h**2)
        pathloss = -10.0 * np.log10(np.clip(d, 1e-6, None))

        phase_x = 2.0 * np.pi * coord_x / max(0.25, float(df["coord_x_m"].max() - df["coord_x_m"].min() + 0.25))
        phase_y = 2.0 * np.pi * coord_y / max(0.30, self.room_depth_m)

        return np.column_stack(
            [
                np.ones_like(pathloss),
                pathloss,
                coord_x,
                coord_y,
                coord_x**2,
                coord_y**2,
                np.sin(phase_x),
                np.cos(phase_x),
                np.sin(phase_y),
                np.cos(phase_y),
                orient_lr,
                orient_fb,
                h,
                is_ground,
                is_elevated,
                pathloss * orient_lr,
                pathloss * orient_fb,
                pathloss * h,
            ]
        )

    def fit(self, train_df: pd.DataFrame) -> "AnalyticPhysicsProxyModel":
        self.x_center_ = float(train_df["coord_x_m"].median())
        X = self._design(train_df)
        y = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
        self.coef_, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return self

    def predict(self, candidates_df: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("AnalyticPhysicsProxyModel is not fitted.")
        return self._design(candidates_df) @ self.coef_


@dataclass
class SionnaPhysicsProxyModel(PhysicalBackendProtocol):
    """Sionna RT-backed physical proxy producing synthetic RSSI features."""

    frequency_hz: float = 2.437e9
    tx_power_dbm: float = 20.0
    receiver_height_m: float = 1.0
    noise_floor_dbm: float = -98.0
    max_depth: int = 2
    samples_per_src: int = 4000
    max_num_paths_per_src: int = 6000
    x_center_: float | None = None
    grid_x_min_: float | None = None
    grid_x_max_: float | None = None
    grid_y_min_: float | None = None
    grid_y_max_: float | None = None

    def __post_init__(self) -> None:
        import sionna.rt as rt  # type: ignore

        self.rt = rt
        self.path_solver = rt.PathSolver()
        self._scene_cache: dict[str, object] = {}

    def fit(self, train_df: pd.DataFrame) -> "SionnaPhysicsProxyModel":
        self.x_center_ = float(train_df["coord_x_m"].median())
        self.grid_x_min_ = float(train_df["grid_x"].min())
        self.grid_x_max_ = float(train_df["grid_x"].max())
        self.grid_y_min_ = float(train_df["grid_y"].min())
        self.grid_y_max_ = float(train_df["grid_y"].max())
        return self

    def _scene_key_for_row(self, row: pd.Series) -> str:
        campaign = str(row["campaign"]).lower()
        line_idx = int(round(float(row["grid_x"])))
        return f"e102::{campaign}::line{line_idx}"

    @staticmethod
    def _xml_box(
        name: str,
        material_id: str,
        *,
        sx: float,
        sy: float,
        sz: float,
        tx: float,
        ty: float,
        tz: float,
    ) -> str:
        return (
            f'<shape type="cube" id="{name}">'
            f'<transform name="to_world">'
            f'<scale x="{sx:.6f}" y="{sy:.6f}" z="{sz:.6f}"/>'
            f'<translate x="{tx:.6f}" y="{ty:.6f}" z="{tz:.6f}"/>'
            f"</transform>"
            f'<ref id="{material_id}" name="bsdf"/>'
            f"</shape>"
        )

    def _chairs_count(self, campaign: str, line_idx: int) -> int:
        if "exp1" in campaign:
            return {2: 2, 3: 2, 4: 2}.get(line_idx, 1)
        if "exp2" in campaign:
            return {1: 3, 2: 3, 4: 0}.get(line_idx, 1)
        if "exp3" in campaign:
            return {0: 1, 1: 1, 2: 2, 3: 0}.get(line_idx, 1)
        if "exp4" in campaign:
            return {0: 1, 1: 2, 2: 1, 3: 1}.get(line_idx, 1)
        # exp5/exp6 are elevation campaigns without dedicated photos here.
        return 1

    def _tables_variant(self, campaign: str, line_idx: int) -> tuple[bool, bool]:
        # (remove_middle_row, add_diagonal_table)
        if "exp1" in campaign and line_idx >= 2:
            return (True, False)
        if "exp3" in campaign and line_idx == 2:
            return (False, True)
        return (False, False)

    def _build_custom_e102_scene_xml(self, campaign: str, line_idx: int) -> str:
        # Dimensions calibrated from provided measurements and pictures.
        room_width = 5.4
        room_depth = 6.0
        room_height = 2.9
        y_center = -room_depth / 2.0  # Back wall aligned around y=0.
        whiteboard_width = 3.95
        whiteboard_z_bottom = 1.18
        whiteboard_z_top = 2.20
        whiteboard_z_center = 0.5 * (whiteboard_z_bottom + whiteboard_z_top)
        whiteboard_half_height = 0.5 * (whiteboard_z_top - whiteboard_z_bottom)

        xml = ['<scene version="2.1.0">']
        xml.append('<bsdf type="itu-radio-material" id="wall-mat"><string name="type" value="concrete"/><float name="thickness" value="0.20"/></bsdf>')
        xml.append('<bsdf type="itu-radio-material" id="board-mat"><string name="type" value="wood"/><float name="thickness" value="0.03"/></bsdf>')
        xml.append('<bsdf type="itu-radio-material" id="desk-mat"><string name="type" value="wood"/><float name="thickness" value="0.04"/></bsdf>')
        xml.append('<bsdf type="itu-radio-material" id="chair-mat"><string name="type" value="metal"/><float name="thickness" value="0.03"/></bsdf>')
        xml.append('<bsdf type="itu-radio-material" id="window-mat"><string name="type" value="glass"/><float name="thickness" value="0.01"/></bsdf>')

        # Room shell
        xml.append(
            self._xml_box(
                "floor",
                "wall-mat",
                sx=room_width / 2.0,
                sy=room_depth / 2.0,
                sz=0.03,
                tx=0.0,
                ty=y_center,
                tz=-0.03,
            )
        )
        xml.append(
            self._xml_box(
                "ceiling",
                "wall-mat",
                sx=room_width / 2.0,
                sy=room_depth / 2.0,
                sz=0.03,
                tx=0.0,
                ty=y_center,
                tz=room_height + 0.03,
            )
        )
        xml.append(
            self._xml_box(
                "wall_left",
                "wall-mat",
                sx=0.04,
                sy=room_depth / 2.0,
                sz=room_height / 2.0,
                tx=-room_width / 2.0 - 0.04,
                ty=y_center,
                tz=room_height / 2.0,
            )
        )
        xml.append(
            self._xml_box(
                "wall_right",
                "wall-mat",
                sx=0.04,
                sy=room_depth / 2.0,
                sz=room_height / 2.0,
                tx=room_width / 2.0 + 0.04,
                ty=y_center,
                tz=room_height / 2.0,
            )
        )
        xml.append(
            self._xml_box(
                "wall_back",
                "wall-mat",
                sx=room_width / 2.0,
                sy=0.04,
                sz=room_height / 2.0,
                tx=0.0,
                ty=y_center + room_depth / 2.0 + 0.04,
                tz=room_height / 2.0,
            )
        )
        xml.append(
            self._xml_box(
                "wall_front",
                "wall-mat",
                sx=room_width / 2.0,
                sy=0.04,
                sz=room_height / 2.0,
                tx=0.0,
                ty=y_center - room_depth / 2.0 - 0.04,
                tz=room_height / 2.0,
            )
        )

        # Whiteboard on the back side (as seen in photos)
        xml.append(
            self._xml_box(
                "whiteboard",
                "board-mat",
                sx=whiteboard_width / 2.0,
                sy=0.02,
                sz=whiteboard_half_height,
                tx=0.0,
                ty=0.02,
                tz=whiteboard_z_center,
            )
        )
        # Side window proxy (right wall)
        xml.append(
            self._xml_box(
                "window_right",
                "window-mat",
                sx=0.01,
                sy=0.30,
                sz=0.45,
                tx=room_width / 2.0,
                ty=-0.20,
                tz=1.50,
            )
        )

        # Desks layout approximated from pictures
        remove_middle_row, add_diagonal_table = self._tables_variant(campaign, line_idx)
        desk_rows = [-2.9, -2.1, -1.3, -0.6]
        if remove_middle_row:
            desk_rows = [-2.9, -1.3, -0.6]
        desk_cols = [-1.35, -0.45, 0.45, 1.35]
        desk_idx = 0
        for y in desk_rows:
            for x in desk_cols:
                desk_idx += 1
                xml.append(
                    self._xml_box(
                        f"desk_{desk_idx}",
                        "desk-mat",
                        sx=0.42,
                        sy=0.28,
                        sz=0.03,
                        tx=x,
                        ty=y,
                        tz=0.75,
                    )
                )
        if add_diagonal_table:
            xml.append(
                self._xml_box(
                    "desk_diag",
                    "desk-mat",
                    sx=0.55,
                    sy=0.28,
                    sz=0.03,
                    tx=0.15,
                    ty=-1.2,
                    tz=0.78,
                )
            )

        # Chairs on tables (dominant obstacles in provided photos)
        chair_slots = [
            (-1.25, -2.1),
            (-0.35, -2.1),
            (0.55, -2.1),
            (1.25, -2.1),
            (-0.80, -1.3),
            (0.80, -1.3),
            (0.15, -0.9),
        ]
        n_chairs = self._chairs_count(campaign, line_idx)
        for i, (x, y) in enumerate(chair_slots[:n_chairs], start=1):
            xml.append(
                self._xml_box(
                    f"chair_{i}",
                    "chair-mat",
                    sx=0.12,
                    sy=0.12,
                    sz=0.45,
                    tx=x,
                    ty=y,
                    tz=1.20,
                )
            )

        xml.append("</scene>")
        return "".join(xml)

    def _load_scene(self, key: str):
        if key in self._scene_cache:
            return self._scene_cache[key]
        campaign = key.split("::")[1]
        line_idx = int(key.split("::line")[-1])
        scene_xml = self._build_custom_e102_scene_xml(campaign, line_idx)
        scene = self.rt.load_scene_from_string(scene_xml)
        scene.frequency = self.frequency_hz
        scene.tx_array = self.rt.PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="V",
        )
        scene.rx_array = self.rt.PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="V",
        )
        self._scene_cache[key] = scene
        return scene

    def _orientation_angle(self, row: pd.Series) -> float:
        lr = float(row["ctx_orientation_lr"])
        fb = float(row["ctx_orientation_fb"])
        if fb > 0.5:
            return 0.0
        if fb < -0.5:
            return math.pi
        if lr > 0.5:
            return -math.pi / 2.0
        return math.pi / 2.0

    def _tx_positions(self, row: pd.Series) -> list[tuple[float, float, float]]:
        if self.x_center_ is None:
            raise RuntimeError("SionnaPhysicsProxyModel is not fitted.")

        angle = self._orientation_angle(row)
        c, s = math.cos(angle), math.sin(angle)
        h = max(0.1, float(row["ctx_router_height_m"]))
        base_x = 0.0
        base_y = -float(row["router_distance_m"])

        offsets = [(0.00, 0.00), (0.06, 0.02), (-0.06, -0.02)]
        txs: list[tuple[float, float, float]] = []
        for dx, dy in offsets:
            rx = c * dx - s * dy
            ry = s * dx + c * dy
            txs.append((base_x + rx, base_y + ry, h))
        return txs

    def _rx_position(self, row: pd.Series) -> tuple[float, float, float]:
        if self.x_center_ is None:
            raise RuntimeError("SionnaPhysicsProxyModel is not fitted.")
        if None in (self.grid_x_min_, self.grid_x_max_, self.grid_y_min_, self.grid_y_max_):
            raise RuntimeError("Grid bounds are not initialized.")

        grid_x = float(row["grid_x"])
        grid_y = float(row["grid_y"])

        # Map measurement grid onto the whiteboard surface:
        # - horizontal axis from column range to board width
        # - vertical axis from row range to board [1.18m, 2.20m]
        whiteboard_width = 3.95
        wb_z_bottom = 1.18
        wb_z_top = 2.20

        denom_y = max(1e-9, self.grid_y_max_ - self.grid_y_min_)
        frac_y = (grid_y - self.grid_y_min_) / denom_y
        x = -whiteboard_width / 2.0 + frac_y * whiteboard_width

        denom_x = max(1e-9, self.grid_x_max_ - self.grid_x_min_)
        frac_x = (grid_x - self.grid_x_min_) / denom_x
        # grid_x=0 appears at the top of the board in the collected images.
        z = wb_z_top - frac_x * (wb_z_top - wb_z_bottom)

        # Receiver is on the board plane with a tiny standoff.
        y = 0.03
        return (x, y, z)

    def _simulate_single_link(self, scene, tx_pos: tuple[float, float, float], rx_pos: tuple[float, float, float]) -> tuple[float, float, int]:
        for name in ("tx_tmp", "rx_tmp"):
            if name in scene.transmitters or name in scene.receivers:
                scene.remove(name)

        scene.add(self.rt.Transmitter("tx_tmp", position=tx_pos, power_dbm=self.tx_power_dbm))
        scene.add(self.rt.Receiver("rx_tmp", position=rx_pos))

        paths = self.path_solver(
            scene,
            max_depth=self.max_depth,
            max_num_paths_per_src=self.max_num_paths_per_src,
            samples_per_src=self.samples_per_src,
            synthetic_array=True,
            los=True,
            specular_reflection=True,
            diffuse_reflection=False,
            refraction=True,
            diffraction=False,
        )
        a, tau = paths.cir(out_type="numpy")
        power_lin = float(np.sum(np.abs(a) ** 2))

        tau_arr = np.asarray(tau).reshape(-1)
        valid_tau = tau_arr[np.isfinite(tau_arr) & (tau_arr >= 0)]
        if valid_tau.size > 1:
            delay_spread_ns = float(np.std(valid_tau) * 1e9)
        else:
            delay_spread_ns = 0.0

        for name in ("tx_tmp", "rx_tmp"):
            if name in scene.transmitters or name in scene.receivers:
                scene.remove(name)

        return power_lin, delay_spread_ns, int(valid_tau.size)

    def predict(self, candidates_df: pd.DataFrame) -> np.ndarray:
        rows = []
        for _, row in candidates_df.iterrows():
            scene_key = self._scene_key_for_row(row)
            scene = self._load_scene(scene_key)
            rx_pos = self._rx_position(row)
            tx_positions = self._tx_positions(row)

            ant_dbm = []
            noise_terms = []
            for tx_pos in tx_positions:
                p_lin, delay_spread_ns, n_paths = self._simulate_single_link(scene, tx_pos, rx_pos)
                sig_dbm = self.tx_power_dbm + 10.0 * math.log10(max(p_lin, 1e-15))
                # Surrogate mapping from multipath richness to noise floor variations.
                noise_dbm = self.noise_floor_dbm + 0.08 * delay_spread_ns + 0.35 * max(0, n_paths - 1)
                ant_dbm.append(sig_dbm)
                noise_terms.append(noise_dbm)

            lin_sum = sum(10.0 ** (x / 10.0) for x in ant_dbm)
            global_signal = 10.0 * math.log10(max(lin_sum, 1e-15))
            noise = float(np.mean(noise_terms))
            rows.append([global_signal, noise, ant_dbm[0], ant_dbm[1], ant_dbm[2]])

        return np.asarray(rows, dtype=float)


def _candidate_table(train_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        train_df.groupby(["grid_cell", "router_distance_m", "campaign"], as_index=False)
        .agg(
            coord_x_m=("coord_x_m", "mean"),
            coord_y_m=("coord_y_m", "mean"),
            grid_x=("grid_x", "mean"),
            grid_y=("grid_y", "mean"),
            n=("grid_cell", "size"),
            Signal=("Signal", "mean"),
            Noise=("Noise", "mean"),
            signal_A1=("signal_A1", "mean"),
            signal_A2=("signal_A2", "mean"),
            signal_A3=("signal_A3", "mean"),
            ctx_orient_right=("ctx_orient_right", "mean"),
            ctx_orient_left=("ctx_orient_left", "mean"),
            ctx_orient_front=("ctx_orient_front", "mean"),
            ctx_orient_back=("ctx_orient_back", "mean"),
            ctx_orientation_lr=("ctx_orientation_lr", "mean"),
            ctx_orientation_fb=("ctx_orientation_fb", "mean"),
            ctx_router_height_m=("ctx_router_height_m", "mean"),
            ctx_is_ground=("ctx_is_ground", "mean"),
            ctx_is_elevated=("ctx_is_elevated", "mean"),
        )
        .rename(columns={col: f"measured_{col}" for col in FEATURE_COLUMNS})
    )
    grouped["candidate_key"] = (
        grouped["grid_cell"]
        + "::d"
        + grouped["router_distance_m"].map(lambda d: f"{d:g}")
        + "::"
        + grouped["campaign"].astype(str)
    )
    return grouped


def _predict_by_nearest(observed: np.ndarray, prototypes: np.ndarray, labels: np.ndarray) -> np.ndarray:
    dists = np.linalg.norm(observed[:, None, :] - prototypes[None, :, :], axis=2)
    return labels[np.argmin(dists, axis=1)]


def _evaluate_predictions(test_df: pd.DataFrame, pred_keys: np.ndarray, candidates: pd.DataFrame) -> dict[str, float]:
    key_to_row = candidates.set_index("candidate_key")
    pred_df = key_to_row.loc[pred_keys]

    true_cells = test_df["grid_cell"].to_numpy()
    pred_cells = pred_df["grid_cell"].to_numpy()
    true_d = test_df["router_distance_m"].to_numpy(dtype=float)
    pred_d = pred_df["router_distance_m"].to_numpy(dtype=float)

    true_coords = test_df[["coord_x_m", "coord_y_m"]].to_numpy(dtype=float)
    pred_coords = pred_df[["coord_x_m", "coord_y_m"]].to_numpy(dtype=float)
    err_m = np.linalg.norm(true_coords - pred_coords, axis=1)

    return {
        "cell_accuracy": float(accuracy_score(true_cells, pred_cells)),
        "router_distance_accuracy": float(accuracy_score(true_d, pred_d)),
        "mean_error_m": float(err_m.mean()),
        "p90_error_m": float(np.percentile(err_m, 90)),
    }


def _compute_delta(new: dict[str, float], base: dict[str, float]) -> dict[str, float]:
    delta = {
        "cell_accuracy_abs": new["cell_accuracy"] - base["cell_accuracy"],
        "mean_error_m_abs": new["mean_error_m"] - base["mean_error_m"],
        "p90_error_m_abs": new["p90_error_m"] - base["p90_error_m"],
    }
    if "router_distance_accuracy" in new and "router_distance_accuracy" in base:
        delta["router_distance_accuracy_abs"] = new["router_distance_accuracy"] - base["router_distance_accuracy"]
    return delta


def _build_backend(name: str, room_depth_m: float) -> PhysicalBackendProtocol:
    if name == "analytic":
        return AnalyticPhysicsProxyModel(room_depth_m=room_depth_m)
    if name == "sionna":
        return SionnaPhysicsProxyModel()
    raise ValueError(f"Unknown backend: {name}")


def evaluate_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    ridge_alpha: float,
    room_depth_m: float,
    physical_backend: str,
) -> dict[str, object]:
    candidates = _candidate_table(train_df)
    labels = candidates["candidate_key"].to_numpy(dtype=object)
    observed_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=float)

    measured_proto = candidates[[f"measured_{c}" for c in FEATURE_COLUMNS]].to_numpy(dtype=float)

    pred_data = _predict_by_nearest(observed_test, measured_proto, labels)
    data_metrics = _evaluate_predictions(test_df, pred_data, candidates)

    backend = _build_backend(physical_backend, room_depth_m=room_depth_m).fit(train_df)
    phys_proto = backend.predict(candidates)

    pred_phys = _predict_by_nearest(observed_test, phys_proto, labels)
    phys_metrics = _evaluate_predictions(test_df, pred_phys, candidates)

    calibrator = Ridge(alpha=ridge_alpha, fit_intercept=True)
    calibrator.fit(phys_proto, measured_proto)
    hybrid_proto = calibrator.predict(phys_proto)

    pred_hybrid = _predict_by_nearest(observed_test, hybrid_proto, labels)
    hybrid_metrics = _evaluate_predictions(test_df, pred_hybrid, candidates)

    return {
        "samples": {
            "train": int(len(train_df)),
            "test": int(len(test_df)),
            "candidate_states": int(len(candidates)),
        },
        "models": {
            "data_driven_centroid": data_metrics,
            "physical_only": phys_metrics,
            "hybrid_physical_ml": hybrid_metrics,
        },
        "delta_vs_data": {
            "physical_only": _compute_delta(phys_metrics, data_metrics),
            "hybrid_physical_ml": _compute_delta(hybrid_metrics, data_metrics),
        },
        "delta_hybrid_vs_physical": _compute_delta(hybrid_metrics, phys_metrics),
    }


def run_random_protocol(
    df: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
    ridge_alpha: float,
    room_depth_m: float,
    physical_backend: str,
) -> dict[str, object]:
    stratify = df["grid_cell"].astype(str)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    result = evaluate_split(
        train_df,
        test_df,
        ridge_alpha=ridge_alpha,
        room_depth_m=room_depth_m,
        physical_backend=physical_backend,
    )
    result["protocol"] = "random"
    return result


def run_loco_e102_protocol(
    df: pd.DataFrame,
    *,
    ridge_alpha: float,
    room_depth_m: float,
    physical_backend: str,
) -> dict[str, object]:
    campaigns = sorted(df["campaign"].unique())
    folds: list[dict[str, object]] = []

    for held_out in campaigns:
        train_df = df[df["campaign"] != held_out].copy()
        test_df = df[df["campaign"] == held_out].copy()
        split_result = evaluate_split(
            train_df,
            test_df,
            ridge_alpha=ridge_alpha,
            room_depth_m=room_depth_m,
            physical_backend=physical_backend,
        )
        split_result["held_out_campaign"] = str(held_out)
        folds.append(split_result)

    def _mean_metric(model_name: str, metric_name: str) -> float:
        return float(np.mean([fold["models"][model_name][metric_name] for fold in folds]))

    summary = {
        "n_folds": len(folds),
        "models_mean": {
            "data_driven_centroid": {
                "cell_accuracy": _mean_metric("data_driven_centroid", "cell_accuracy"),
                "mean_error_m": _mean_metric("data_driven_centroid", "mean_error_m"),
                "p90_error_m": _mean_metric("data_driven_centroid", "p90_error_m"),
            },
            "physical_only": {
                "cell_accuracy": _mean_metric("physical_only", "cell_accuracy"),
                "mean_error_m": _mean_metric("physical_only", "mean_error_m"),
                "p90_error_m": _mean_metric("physical_only", "p90_error_m"),
            },
            "hybrid_physical_ml": {
                "cell_accuracy": _mean_metric("hybrid_physical_ml", "cell_accuracy"),
                "mean_error_m": _mean_metric("hybrid_physical_ml", "mean_error_m"),
                "p90_error_m": _mean_metric("hybrid_physical_ml", "p90_error_m"),
            },
        },
    }

    summary["delta_hybrid_vs_physical"] = _compute_delta(
        summary["models_mean"]["hybrid_physical_ml"],
        summary["models_mean"]["physical_only"],
    )
    summary["delta_hybrid_vs_data"] = _compute_delta(
        summary["models_mean"]["hybrid_physical_ml"],
        summary["models_mean"]["data_driven_centroid"],
    )

    return {
        "protocol": "loco_e102",
        "folds": folds,
        "summary": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hybrid physics+ML benchmark with optional Sionna RT check.")
    parser.add_argument(
        "--campaign",
        action="append",
        default=[],
        metavar="PATH[:DISTANCE]",
        help="Campaign folder path, optionally with explicit router distance in meters.",
    )
    parser.add_argument(
        "--protocol",
        choices=["random", "loco_e102"],
        default="random",
        help="Evaluation protocol.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "analytic", "sionna"],
        default="auto",
        help="Physical backend used for `physical_only` model.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio for random protocol.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for random protocol.")
    parser.add_argument("--ridge-alpha", type=float, default=1.0, help="Ridge regularization for calibration.")
    parser.add_argument("--room-depth-m", type=float, default=4.0, help="Approx room depth for analytic proxy.")
    parser.add_argument(
        "--default-router-height-m",
        type=float,
        default=0.75,
        help="Default router height when campaign-specific elevation is unknown.",
    )
    parser.add_argument("--use-sionna", action="store_true", help="Attempt to import sionna.rt and report status.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Output JSON path (default: {DEFAULT_OUTPUT}).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = parse_campaign_args(args.campaign)
    df_raw = load_measurements(specs)

    missing_cols = [col for col in FEATURE_COLUMNS if col not in df_raw.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in dataset: {missing_cols}")

    df = _annotate_campaign_context(df_raw, default_router_height_m=args.default_router_height_m)

    sionna_status = try_import_sionna(args.use_sionna or args.backend in {"auto", "sionna"})

    effective_backend = args.backend
    if args.backend == "auto":
        effective_backend = "sionna" if sionna_status.get("available") else "analytic"
    if effective_backend == "sionna" and not sionna_status.get("available"):
        raise RuntimeError(
            "Backend `sionna` requested but Sionna RT is unavailable. "
            f"Status: {sionna_status}"
        )

    if args.protocol == "random":
        protocol_results = run_random_protocol(
            df,
            test_size=args.test_size,
            random_state=args.random_state,
            ridge_alpha=args.ridge_alpha,
            room_depth_m=args.room_depth_m,
            physical_backend=effective_backend,
        )
    else:
        protocol_results = run_loco_e102_protocol(
            df,
            ridge_alpha=args.ridge_alpha,
            room_depth_m=args.room_depth_m,
            physical_backend=effective_backend,
        )

    payload = {
        "experiment": "sionna_rt_hybrid_proxy_v3",
        "feature_columns": FEATURE_COLUMNS,
        "campaigns": [str(spec.path) for spec in specs],
        "config": {
            "protocol": args.protocol,
            "backend_requested": args.backend,
            "backend_effective": effective_backend,
            "test_size": args.test_size,
            "random_state": args.random_state,
            "ridge_alpha": args.ridge_alpha,
            "room_depth_m": args.room_depth_m,
            "default_router_height_m": args.default_router_height_m,
        },
        "sionna_rt": sionna_status,
        "results": protocol_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))

    print(f"Saved: {args.output}")
    print(f"Protocol: {args.protocol} | backend={effective_backend}")
    if args.protocol == "random":
        models = protocol_results["models"]
        for name in ("data_driven_centroid", "physical_only", "hybrid_physical_ml"):
            m = models[name]
            print(
                f"{name:>20} -> cell_acc={m['cell_accuracy']:.4f}, "
                f"mean_err={m['mean_error_m']:.3f}m, p90={m['p90_error_m']:.3f}m"
            )
    else:
        summary = protocol_results["summary"]["models_mean"]
        for name in ("data_driven_centroid", "physical_only", "hybrid_physical_ml"):
            m = summary[name]
            print(
                f"{name:>20} (mean folds) -> cell_acc={m['cell_accuracy']:.4f}, "
                f"mean_err={m['mean_error_m']:.3f}m, p90={m['p90_error_m']:.3f}m"
            )
    print(f"Sionna RT status: {sionna_status}")


if __name__ == "__main__":
    main()
