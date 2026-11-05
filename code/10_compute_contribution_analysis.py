from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from pipeline_utils import ensure_dirs, read_csv_tail, sigmoid, softplus, project_root

ROOT = project_root(__file__)
PATHS = ensure_dirs(ROOT)
TABLE = PATHS["tables"]
PRED = PATHS["predictions"]

RUNS = {
    "distill": {
        "pred": PRED / "01_chronological_test_predictions.csv",
        "coef": TABLE / "01_chronological_coefficients_all.csv",
        "thr": TABLE / "01_chronological_thresholds_all.csv",
        "pre": TABLE / "01_chronological_preprocessing.json",
    },
    "nodistill": {
        "pred": PRED / "05_nodistill_test_predictions.csv",
        "coef": TABLE / "05_nodistill_coefficients_all.csv",
        "thr": TABLE / "05_nodistill_thresholds_all.csv",
        "pre": TABLE / "05_nodistill_preprocessing.json",
    },
}

X_RAW = ["Air_Temp", "Wet_Bulb_Temp", "Dewpoint"]
BETA = ["beta_Air_Temp", "beta_Wet_Bulb_Temp", "beta_Dewpoint"]
CUT = ["thr_Air_std", "thr_WBT_std", "thr_Dew_std"]
COMPONENTS = ["Air temperature", "Wet-bulb temperature", "Dewpoint"]
GATE_K = 5.0


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    return {
        "auc": float(roc_auc_score(y, p)),
        "accuracy": float(accuracy_score(y, p > 0.5)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
    }


share_rows = []
ablation_rows = []
verification_rows = []
for label, paths in RUNS.items():
    for p in paths.values():
        if not p.exists():
            raise FileNotFoundError(f"Missing {label} input: {p}")
    pred = pd.read_csv(paths["pred"])
    n = len(pred)
    if not all(c in pred.columns for c in ["y_true", "y_prob"] + X_RAW):
        raise KeyError(f"{paths['pred']} must contain y_true, y_prob, and {X_RAW}")
    coef = read_csv_tail(paths["coef"], n, ["beta0"] + BETA)
    thr = read_csv_tail(paths["thr"], n, CUT)
    pre = json.loads(paths["pre"].read_text(encoding="utf-8"))
    x_mean = np.asarray(pre["x_mean"], float)
    x_scale = np.asarray(pre["x_scale"], float)

    x_raw = pred[X_RAW].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    x_std = (x_raw - x_mean.reshape(1, -1)) / x_scale.reshape(1, -1)
    cuts = thr[CUT].to_numpy(float)
    betas = coef[BETA].to_numpy(float)
    beta0 = coef["beta0"].to_numpy(float)
    gates = softplus(GATE_K * (x_std - cuts)) / GATE_K
    contributions = betas * gates
    logits = beta0 + contributions.sum(axis=1)
    reconstructed = sigmoid(logits)
    supplied = pred["y_prob"].to_numpy(float)
    y = pred["y_true"].to_numpy(int)

    verification_rows.append({
        "run": label,
        "n_test": int(n),
        "max_abs_prediction_difference": float(np.nanmax(np.abs(reconstructed - supplied))),
        "mean_abs_prediction_difference": float(np.nanmean(np.abs(reconstructed - supplied))),
    })

    mean_abs = np.nanmean(np.abs(contributions), axis=0)
    shares = mean_abs / mean_abs.sum()
    for component, value, share in zip(COMPONENTS, mean_abs, shares):
        share_rows.append({
            "run": label,
            "component": component,
            "mean_absolute_log_odds_contribution": float(value),
            "share": float(share),
            "share_percent": float(100 * share),
        })

    base = metrics(y, reconstructed)
    ablation_rows.append({"run": label, "removed_component": "None (full model)", **base})
    for j, component in enumerate(COMPONENTS):
        p_minus = sigmoid(logits - contributions[:, j])
        ablation_rows.append({"run": label, "removed_component": component, **metrics(y, p_minus)})

pd.DataFrame(share_rows).to_csv(TABLE / "10_temperature_contribution_shares.csv", index=False)
pd.DataFrame(ablation_rows).to_csv(TABLE / "10_temperature_component_ablation.csv", index=False)
pd.DataFrame(verification_rows).to_csv(TABLE / "10_prediction_reconstruction_check.csv", index=False)
print("TEMPERATURE CONTRIBUTION SHARES")
print(pd.DataFrame(share_rows).to_string(index=False))
print("\nPREDICTION RECONSTRUCTION CHECK")
print(pd.DataFrame(verification_rows).to_string(index=False))
