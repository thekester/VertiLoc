"""TargetKNN stress test — anti-leakage LORO.

Met le TargetKNN à l'épreuve sur 5 axes :
  1. Courbe K (K=1..25) avec IC à 30 tirages — generalisation avec peu de mesures
  2. Sweep n_neighbors (1, 3, 5, 7) — sensibilité hyperparamètre
  3. Robustesse au bruit sur les supports (sigma=0,1,2,3 dBm)
  4. Source-augmenté vs TargetKNN pur — la donnée source aide-t-elle ?
  5. Difficulté par cellule — quelles cellules sont les plus dures à calibrer ?
  6. Contrôle labels mélangés — le gain disparaît-il si la calibration est fausse ?

Protocole anti-leakage STRICT :
  - Salle cible TOTALEMENT absente de toute phase d'entraînement
  - Le StandardScaler est ajusté uniquement sur les salles sources
  - Les K supports sont tirés ALÉATOIREMENT depuis le test set de la salle cible
    (sans remplacement, jamais utilisés dans la requête)
  - Le KNN source (baseline LORO) est entraîné UNIQUEMENT sur les salles sources
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from localization.catalog import BENCHMARK_REPORT_DIR, FEATURE_COLUMNS, filter_room_campaigns
from localization.data import load_measurements

REPORT_DIR = BENCHMARK_REPORT_DIR
SEED = 42

warnings.filterwarnings("ignore", message=".*unique classes.*")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_multiroom() -> pd.DataFrame:
    campaigns = filter_room_campaigns()
    frames = []
    for room, specs in campaigns.items():
        for spec in specs:
            if spec.path.exists():
                df = load_measurements([spec])
                df["room"] = room
                frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _cell_lookup(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("grid_cell")[["coord_x_m", "coord_y_m"]].first()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(y_true: np.ndarray, y_pred: np.ndarray, lookup: pd.DataFrame) -> dict:
    acc = float((y_true == y_pred).mean())
    pc = lookup.reindex(y_pred)[["coord_x_m", "coord_y_m"]].to_numpy()
    tc = lookup.reindex(y_true)[["coord_x_m", "coord_y_m"]].to_numpy()
    mask = ~(np.isnan(pc).any(1) | np.isnan(tc).any(1))
    err = float(np.linalg.norm(pc[mask] - tc[mask], axis=1).mean()) if mask.sum() else float("nan")
    p75 = float(np.percentile(np.linalg.norm(pc[mask] - tc[mask], axis=1), 75)) if mask.sum() else float("nan")
    p90 = float(np.percentile(np.linalg.norm(pc[mask] - tc[mask], axis=1), 90)) if mask.sum() else float("nan")
    return {"cell_acc": acc, "mean_error_m": err, "p75_error_m": p75, "p90_error_m": p90, "n": int(len(y_true))}


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------

def target_knn(
    scaler: StandardScaler,
    X_sup: np.ndarray, y_sup: np.ndarray,
    X_q: np.ndarray,
    n_neighbors: int = 3,
    noise_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """KNN entraîné uniquement sur les K supports de la salle cible.

    noise_sigma > 0 : ajoute du bruit gaussien aux supports (test de robustesse).
    Les données sources ne sont PAS utilisées — anti-leakage garanti.
    """
    X_s = X_sup.copy().astype(np.float32)
    if noise_sigma > 0 and rng is not None:
        X_s = X_s + rng.normal(0, noise_sigma, X_s.shape).astype(np.float32)
    X_s = scaler.transform(X_s)
    X_q_s = scaler.transform(X_q.astype(np.float32))
    k = min(n_neighbors, len(X_s))
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn.fit(X_s, y_sup)
    return knn.predict(X_q_s)


def source_knn(
    scaler: StandardScaler,
    X_src: np.ndarray, y_src: np.ndarray,
    X_q: np.ndarray,
    n_neighbors: int = 7,
) -> np.ndarray:
    """KNN entraîné UNIQUEMENT sur les salles sources (baseline LORO sans adaptation)."""
    X_src_s = scaler.transform(X_src.astype(np.float32))
    X_q_s = scaler.transform(X_q.astype(np.float32))
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
    knn.fit(X_src_s, y_src)
    return knn.predict(X_q_s)


def hybrid_knn(
    scaler: StandardScaler,
    X_src: np.ndarray, y_src: np.ndarray,
    X_sup: np.ndarray, y_sup: np.ndarray,
    X_q: np.ndarray,
    n_neighbors: int = 5,
    target_replicas: int = 10,
) -> np.ndarray:
    """KNN combinant données sources + supports cibles (répliquer pour sur-pondérer).

    Les supports cibles sont répétés `target_replicas` fois pour compenser leur
    petite taille vs le corpus source. Valide car le label est connu à l'installation.
    """
    X_sup_rep = np.repeat(X_sup, target_replicas, axis=0)
    y_sup_rep = np.repeat(y_sup, target_replicas)
    X_comb = np.vstack([X_src, X_sup_rep]).astype(np.float32)
    y_comb = np.concatenate([y_src, y_sup_rep])
    X_c_s = scaler.transform(X_comb)
    X_q_s = scaler.transform(X_q.astype(np.float32))
    k = min(n_neighbors, len(X_c_s))
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn.fit(X_c_s, y_comb)
    return knn.predict(X_q_s)


def full_target_knn(
    scaler: StandardScaler,
    X_all_target: np.ndarray, y_all_target: np.ndarray,
    X_q: np.ndarray,
    n_neighbors: int = 7,
) -> np.ndarray:
    """KNN entraîné sur TOUTE la salle cible — plafond théorique."""
    X_t_s = scaler.transform(X_all_target.astype(np.float32))
    X_q_s = scaler.transform(X_q.astype(np.float32))
    knn = KNeighborsClassifier(n_neighbors=min(n_neighbors, len(X_t_s)), weights="distance")
    knn.fit(X_t_s, y_all_target)
    return knn.predict(X_q_s)


# ---------------------------------------------------------------------------
# Support sampler
# ---------------------------------------------------------------------------

def _sample_support_query(
    y_test: np.ndarray, k: int, cells: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    sup_idx, qry_idx = [], []
    for cell in cells:
        ci = np.where(y_test == cell)[0]
        k_eff = min(k, len(ci) - 1)
        if k_eff <= 0:
            qry_idx.extend(ci)
            continue
        perm = rng.permutation(len(ci))
        sup_idx.extend(ci[perm[:k_eff]])
        qry_idx.extend(ci[perm[k_eff:]])
    return np.array(sup_idx), np.array(qry_idx)


def _new_audit(source_rooms: list[str], target_room: str, n_source: int, n_target: int) -> dict:
    if target_room in source_rooms:
        raise RuntimeError(f"Leakage: target room {target_room} is present in source rooms.")
    return {
        "source_rooms": source_rooms,
        "target_room": target_room,
        "target_room_present_in_source_training": False,
        "source_sample_count": int(n_source),
        "target_sample_count": int(n_target),
        "support_query_overlap_max": 0,
        "split_checks": 0,
        "split_stats_by_k": {},
    }


def _update_split_audit(audit: dict, k: int, support_idx: np.ndarray, query_idx: np.ndarray) -> None:
    overlap = int(np.intersect1d(support_idx, query_idx).size)
    if overlap:
        raise RuntimeError(f"Leakage: support/query overlap detected for K={k}: {overlap} sample(s).")

    audit["support_query_overlap_max"] = max(audit["support_query_overlap_max"], overlap)
    audit["split_checks"] += 1

    stats = audit["split_stats_by_k"].setdefault(
        str(k),
        {
            "checks": 0,
            "support_min": None,
            "support_max": 0,
            "support_sum": 0,
            "query_min": None,
            "query_max": 0,
            "query_sum": 0,
        },
    )
    stats["checks"] += 1
    stats["support_min"] = (
        int(len(support_idx))
        if stats["support_min"] is None
        else min(stats["support_min"], int(len(support_idx)))
    )
    stats["support_max"] = max(stats["support_max"], int(len(support_idx)))
    stats["support_sum"] += int(len(support_idx))
    stats["query_min"] = (
        int(len(query_idx))
        if stats["query_min"] is None
        else min(stats["query_min"], int(len(query_idx)))
    )
    stats["query_max"] = max(stats["query_max"], int(len(query_idx)))
    stats["query_sum"] += int(len(query_idx))


def _finalize_audit(audit: dict) -> None:
    for stats in audit["split_stats_by_k"].values():
        checks = max(stats["checks"], 1)
        stats["support_mean"] = float(stats["support_sum"] / checks)
        stats["query_mean"] = float(stats["query_sum"] / checks)
        del stats["support_sum"]
        del stats["query_sum"]


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict:
    print("\n=== TargetKNN Stress Test — Protocole anti-leakage strict ===")
    print("  Salle cible absente de tout entraînement — scaler ajusté sur sources seules.")
    seed = args.seed

    df = load_multiroom()
    rooms = sorted(df["room"].unique())
    lookup = _cell_lookup(df)
    X_all = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_all = df["grid_cell"].to_numpy()
    room_arr = df["room"].to_numpy()

    out: dict = {
        "config": vars(args),
        "protocol": (
            "LORO strict: scaler fitted on source rooms only; "
            "K support samples drawn from target room test set, never seen during training; "
            "query = remaining target room samples."
        ),
        "folds": {},
        "averages": {},
        "cell_difficulty": {},
    }

    # ── AXIS 1 : courbe K complète + sweep n_neighbors ──────────────────────
    k_values = args.k_values
    nn_values = args.nn_values

    for fold_idx, held_out in enumerate(rooms):
        print(f"\n  ── Fold: held_out={held_out} ──")
        train_mask = room_arr != held_out
        test_mask  = room_arr == held_out

        X_src, y_src = X_all[train_mask], y_all[train_mask]
        X_tgt, y_tgt = X_all[test_mask],  y_all[test_mask]
        source_rooms = sorted(np.unique(room_arr[train_mask]).tolist())

        # Scaler ajusté UNIQUEMENT sur les sources (anti-leakage)
        scaler = StandardScaler().fit(X_src)

        cells = np.unique(y_tgt)
        n_cells = len(cells)
        n_tgt = len(y_tgt)
        print(f"    Source: {len(y_src)} samples | Cible: {n_tgt} samples, {n_cells} cellules")

        fold: dict = {
            "anti_leakage_audit": _new_audit(
                source_rooms=source_rooms,
                target_room=held_out,
                n_source=len(y_src),
                n_target=n_tgt,
            )
        }

        # Baseline : SourceKNN sans labels cibles
        pred_src = source_knn(scaler, X_src, y_src, X_tgt, n_neighbors=7)
        m_src = _metrics(y_tgt, pred_src, lookup)
        fold["SourceKNN_LORO"] = m_src
        print(f"    SourceKNN (sans labels cibles) : acc={m_src['cell_acc']:.4f}  err={m_src['mean_error_m']:.3f}m")

        # Plafond : FullTargetKNN (entraîné sur TOUT le test — référence théorique)
        # Utiliser une fraction 50/50 pour éviter que ce soit trivial
        rng_full = np.random.default_rng(seed + fold_idx)
        perm_full = rng_full.permutation(n_tgt)
        half = n_tgt // 2
        m_full = _metrics(
            y_tgt[perm_full[half:]],
            full_target_knn(scaler, X_tgt[perm_full[:half]], y_tgt[perm_full[:half]],
                            X_tgt[perm_full[half:]], n_neighbors=7),
            lookup,
        )
        fold["FullTargetKNN_half"] = m_full
        print(f"    FullTargetKNN (50% labels)     : acc={m_full['cell_acc']:.4f}  err={m_full['mean_error_m']:.3f}m")

        # ── Axe 1 : courbe K × n_neighbors ──────────────────────────────────
        print(f"    Courbe K avec {args.n_trials} tirages :")
        for k in k_values:
            fold[f"K{k}"] = {}
            for nn in nn_values:
                accs, errs, p90s = [], [], []
                for trial in range(args.n_trials):
                    rng = np.random.default_rng(seed + fold_idx * 100_000 + trial * 1_000 + nn)
                    si, qi = _sample_support_query(y_tgt, k, cells, rng)
                    _update_split_audit(fold["anti_leakage_audit"], k, si, qi)
                    if len(qi) == 0:
                        continue
                    pred = target_knn(scaler, X_tgt[si], y_tgt[si], X_tgt[qi], n_neighbors=nn)
                    m = _metrics(y_tgt[qi], pred, lookup)
                    accs.append(m["cell_acc"]); errs.append(m["mean_error_m"]); p90s.append(m["p90_error_m"])
                fold[f"K{k}"][f"TargetKNN_nn{nn}"] = {
                    "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                    "err_mean": float(np.mean(errs)), "p90_mean": float(np.mean(p90s)),
                    "n_trials": len(accs),
                }
            # Print best nn at this K
            best_nn = max(nn_values, key=lambda nn: fold[f"K{k}"][f"TargetKNN_nn{nn}"]["acc_mean"])
            b = fold[f"K{k}"][f"TargetKNN_nn{nn}"]
            bv = fold[f"K{k}"][f"TargetKNN_nn{best_nn}"]
            print(f"      K={k:>2}  best_nn={best_nn}  acc={bv['acc_mean']:.4f}±{bv['acc_std']:.4f}"
                  f"  err={bv['err_mean']:.3f}m  p90={bv['p90_mean']:.3f}m")

        # ── Axe 2 : Robustesse au bruit sur les supports ─────────────────────
        print(f"    Robustesse bruit (K={args.robust_k}, nn=3) :")
        fold["robustness"] = {}
        for sigma in args.noise_sigmas:
            accs, errs = [], []
            for trial in range(args.n_trials):
                rng = np.random.default_rng(seed + fold_idx * 100_000 + trial * 1_000)
                si, qi = _sample_support_query(y_tgt, args.robust_k, cells, rng)
                _update_split_audit(fold["anti_leakage_audit"], args.robust_k, si, qi)
                if len(qi) == 0:
                    continue
                pred = target_knn(scaler, X_tgt[si], y_tgt[si], X_tgt[qi],
                                  n_neighbors=3, noise_sigma=sigma, rng=rng)
                m = _metrics(y_tgt[qi], pred, lookup)
                accs.append(m["cell_acc"]); errs.append(m["mean_error_m"])
            fold["robustness"][f"sigma_{sigma:.1f}"] = {
                "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                "err_mean": float(np.mean(errs)),
            }
            print(f"      sigma={sigma:.1f} dBm  acc={np.mean(accs):.4f}±{np.std(accs):.4f}  err={np.mean(errs):.3f}m")

        # ── Axe 3 : Hybride Source + Target ──────────────────────────────────
        print(f"    Hybride Source+Target (nn=5, replicas=10) :")
        fold["hybrid"] = {}
        for k in args.hybrid_k_values:
            accs_t, errs_t, accs_h, errs_h = [], [], [], []
            for trial in range(args.n_trials):
                rng = np.random.default_rng(seed + fold_idx * 100_000 + trial * 1_000)
                si, qi = _sample_support_query(y_tgt, k, cells, rng)
                _update_split_audit(fold["anti_leakage_audit"], k, si, qi)
                if len(qi) == 0:
                    continue
                # pur TargetKNN
                pt = target_knn(scaler, X_tgt[si], y_tgt[si], X_tgt[qi], n_neighbors=3)
                mt = _metrics(y_tgt[qi], pt, lookup)
                accs_t.append(mt["cell_acc"]); errs_t.append(mt["mean_error_m"])
                # hybride
                ph = hybrid_knn(scaler, X_src, y_src, X_tgt[si], y_tgt[si], X_tgt[qi],
                                n_neighbors=5, target_replicas=10)
                mh = _metrics(y_tgt[qi], ph, lookup)
                accs_h.append(mh["cell_acc"]); errs_h.append(mh["mean_error_m"])
            fold["hybrid"][f"K{k}"] = {
                "TargetKNN": {"acc": float(np.mean(accs_t)), "err": float(np.mean(errs_t))},
                "Hybrid":    {"acc": float(np.mean(accs_h)), "err": float(np.mean(errs_h))},
            }
            delta = np.mean(accs_h) - np.mean(accs_t)
            print(f"      K={k:>2}  TargetKNN={np.mean(accs_t):.4f}  Hybrid={np.mean(accs_h):.4f}"
                  f"  delta={delta:+.4f}")

        # ── Axe 4 : Difficulté par cellule (K=5) ─────────────────────────────
        cell_accs: dict[str, list[float]] = {str(c): [] for c in cells}
        for trial in range(args.n_trials):
            rng = np.random.default_rng(seed + fold_idx * 100_000 + trial * 1_000)
            si, qi = _sample_support_query(y_tgt, 5, cells, rng)
            _update_split_audit(fold["anti_leakage_audit"], 5, si, qi)
            if len(qi) == 0:
                continue
            pred = target_knn(scaler, X_tgt[si], y_tgt[si], X_tgt[qi], n_neighbors=3)
            for c in cells:
                mask_c = (y_tgt[qi] == c)
                if mask_c.sum() > 0:
                    cell_accs[str(c)].append(float((pred[mask_c] == c).mean()))
        fold["cell_difficulty_K5"] = {
            c: {"acc_mean": float(np.mean(v)), "acc_std": float(np.std(v))}
            for c, v in cell_accs.items() if v
        }
        # Top-5 hardest
        sorted_cells = sorted(fold["cell_difficulty_K5"].items(), key=lambda x: x[1]["acc_mean"])
        print(f"    Cellules les plus difficiles (K=5) :")
        for cell, stat in sorted_cells[:5]:
            print(f"      {cell:<10} acc={stat['acc_mean']:.3f} ± {stat['acc_std']:.3f}")

        # ── Axe 6 : Contrôle anti-triche par labels mélangés ────────────────
        print("    Contrôle labels mélangés :")
        fold["label_shuffle_control"] = {}
        for k in args.control_k_values:
            valid_accs, shuffled_accs = [], []
            valid_errs, shuffled_errs = [], []
            for trial in range(args.n_trials):
                rng = np.random.default_rng(seed + fold_idx * 100_000 + trial * 1_000 + k)
                si, qi = _sample_support_query(y_tgt, k, cells, rng)
                _update_split_audit(fold["anti_leakage_audit"], k, si, qi)
                if len(qi) == 0:
                    continue

                y_support_valid = y_tgt[si]
                y_support_shuffled = y_support_valid.copy()
                rng.shuffle(y_support_shuffled)

                pred_valid = target_knn(scaler, X_tgt[si], y_support_valid, X_tgt[qi], n_neighbors=3)
                pred_shuffled = target_knn(
                    scaler,
                    X_tgt[si],
                    y_support_shuffled,
                    X_tgt[qi],
                    n_neighbors=3,
                )
                m_valid = _metrics(y_tgt[qi], pred_valid, lookup)
                m_shuffled = _metrics(y_tgt[qi], pred_shuffled, lookup)
                valid_accs.append(m_valid["cell_acc"])
                shuffled_accs.append(m_shuffled["cell_acc"])
                valid_errs.append(m_valid["mean_error_m"])
                shuffled_errs.append(m_shuffled["mean_error_m"])

            fold["label_shuffle_control"][f"K{k}"] = {
                "TargetKNN_valid_labels": {
                    "acc_mean": float(np.mean(valid_accs)),
                    "err_mean": float(np.mean(valid_errs)),
                },
                "TargetKNN_shuffled_support_labels": {
                    "acc_mean": float(np.mean(shuffled_accs)),
                    "err_mean": float(np.mean(shuffled_errs)),
                },
                "delta_acc": float(np.mean(valid_accs) - np.mean(shuffled_accs)),
            }
            print(
                f"      K={k:>2}  vrais_labels={np.mean(valid_accs):.4f}"
                f"  labels_melanges={np.mean(shuffled_accs):.4f}"
                f"  delta={np.mean(valid_accs) - np.mean(shuffled_accs):+.4f}"
            )

        _finalize_audit(fold["anti_leakage_audit"])
        out["folds"][held_out] = fold

    # ── Moyennes cross-room ───────────────────────────────────────────────────
    print("\n  ── Moyennes cross-room ──")

    # SourceKNN baseline
    src_accs = [out["folds"][r]["SourceKNN_LORO"]["cell_acc"] for r in rooms]
    full_accs = [out["folds"][r]["FullTargetKNN_half"]["cell_acc"] for r in rooms]
    print(f"  SourceKNN (sans labels)  : acc={np.mean(src_accs):.4f}  (plancher)")
    print(f"  FullTargetKNN (50%)      : acc={np.mean(full_accs):.4f}  (plafond pratique)")
    out["averages"]["SourceKNN_LORO"] = {"acc_mean": float(np.mean(src_accs))}
    out["averages"]["FullTargetKNN_half"] = {"acc_mean": float(np.mean(full_accs))}

    # Courbe K — best nn par K
    print()
    best_nn_global: dict[int, int] = {}
    for k in k_values:
        best_nn = max(nn_values, key=lambda nn: np.mean(
            [out["folds"][r][f"K{k}"][f"TargetKNN_nn{nn}"]["acc_mean"] for r in rooms]
        ))
        best_nn_global[k] = best_nn
        accs = [out["folds"][r][f"K{k}"][f"TargetKNN_nn{best_nn}"]["acc_mean"] for r in rooms]
        stds = [out["folds"][r][f"K{k}"][f"TargetKNN_nn{best_nn}"]["acc_std"] for r in rooms]
        errs = [out["folds"][r][f"K{k}"][f"TargetKNN_nn{best_nn}"]["err_mean"] for r in rooms]
        p90s = [out["folds"][r][f"K{k}"][f"TargetKNN_nn{best_nn}"]["p90_mean"] for r in rooms]
        out["averages"][f"TargetKNN_K{k}"] = {
            "best_nn": best_nn,
            "acc_mean": float(np.mean(accs)), "acc_std_cross_room": float(np.std(accs)),
            "acc_within_trial_std": float(np.mean(stds)),
            "err_mean": float(np.mean(errs)), "p90_mean": float(np.mean(p90s)),
        }
        print(f"  TargetKNN K={k:>2} nn={best_nn}  acc={np.mean(accs):.4f}±{np.mean(stds):.4f}"
              f"  err={np.mean(errs):.3f}m  p90={np.mean(p90s):.3f}m")

    # Robustesse
    print()
    for sigma in args.noise_sigmas:
        accs = [out["folds"][r]["robustness"][f"sigma_{sigma:.1f}"]["acc_mean"] for r in rooms]
        errs = [out["folds"][r]["robustness"][f"sigma_{sigma:.1f}"]["err_mean"] for r in rooms]
        out["averages"][f"Robustness_sigma{sigma:.1f}"] = {
            "acc_mean": float(np.mean(accs)), "err_mean": float(np.mean(errs))
        }
        print(f"  Bruit sigma={sigma:.1f} dBm  (K={args.robust_k})  acc={np.mean(accs):.4f}  err={np.mean(errs):.3f}m")

    # Hybride
    print()
    for k in args.hybrid_k_values:
        at = [out["folds"][r]["hybrid"][f"K{k}"]["TargetKNN"]["acc"] for r in rooms]
        ah = [out["folds"][r]["hybrid"][f"K{k}"]["Hybrid"]["acc"] for r in rooms]
        out["averages"][f"Hybrid_K{k}"] = {
            "TargetKNN_acc": float(np.mean(at)), "Hybrid_acc": float(np.mean(ah)),
            "delta": float(np.mean(ah) - np.mean(at)),
        }
        print(f"  Hybride K={k:>2}  TargetKNN={np.mean(at):.4f}  Hybrid={np.mean(ah):.4f}"
              f"  delta={np.mean(ah)-np.mean(at):+.4f}")

    print()
    for k in args.control_k_values:
        valid = [
            out["folds"][r]["label_shuffle_control"][f"K{k}"]["TargetKNN_valid_labels"]["acc_mean"]
            for r in rooms
        ]
        shuffled = [
            out["folds"][r]["label_shuffle_control"][f"K{k}"]["TargetKNN_shuffled_support_labels"]["acc_mean"]
            for r in rooms
        ]
        out["averages"][f"LabelShuffleControl_K{k}"] = {
            "valid_labels_acc": float(np.mean(valid)),
            "shuffled_support_labels_acc": float(np.mean(shuffled)),
            "delta_acc": float(np.mean(valid) - np.mean(shuffled)),
        }
        print(
            f"  Contrôle labels K={k:>2}  vrais={np.mean(valid):.4f}"
            f"  melanges={np.mean(shuffled):.4f}"
            f"  delta={np.mean(valid) - np.mean(shuffled):+.4f}"
        )

    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TargetKNN stress test — protocole anti-leakage strict"
    )
    parser.add_argument("--k-values", nargs="+", type=int,
                        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25])
    parser.add_argument("--nn-values", nargs="+", type=int, default=[1, 3, 5, 7])
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Tirages pour estimer la variance de sélection support.")
    parser.add_argument("--robust-k", type=int, default=10,
                        help="K fixe pour le test de robustesse au bruit.")
    parser.add_argument("--noise-sigmas", nargs="+", type=float,
                        default=[0.0, 0.5, 1.0, 2.0, 3.0])
    parser.add_argument("--hybrid-k-values", nargs="+", type=int, default=[3, 5, 10, 25])
    parser.add_argument("--control-k-values", nargs="+", type=int, default=[5, 10, 25])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-name", default="targetknn_stress_test.json")
    args = parser.parse_args()

    results = run(args)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / args.output_name
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
