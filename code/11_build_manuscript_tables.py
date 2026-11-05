from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_utils import ensure_dirs, json_dump, project_root

ROOT = project_root(__file__)
PATHS = ensure_dirs(ROOT)
T = PATHS["tables"]
S = PATHS["station"]
C = PATHS["comparison"]


def read_one(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path).iloc[0].to_dict()

chron = read_one(T / "01_chronological_metrics.csv")
spatial = read_one(T / "02_spatial_blocked_metrics.csv")
nodistill = read_one(T / "05_nodistill_metrics.csv")
stability = pd.read_csv(T / "04_stability_summary_sign_aligned.csv")

stab_map = {r["quantity"]: r for _, r in stability.iterrows()}
summary_rows = [
    {
        "analysis": "Chronological holdout, index-only distilled",
        "auc": float(chron["auc"]),
        "accuracy": float(chron["accuracy"]),
        "average_precision": float(chron["average_precision"]),
        "brier": float(chron["brier"]),
        "teacher_student_cut_mse_val": float(chron["teacher_student_cut_mse_val"]),
    },
    {
        "analysis": "Spatial-blocked holdout, index-only distilled",
        "auc": float(spatial["auc"]),
        "accuracy": float(spatial["accuracy"]),
        "average_precision": float(spatial["average_precision"]),
        "brier": float(spatial["brier"]),
        "teacher_student_cut_mse_val": float(spatial["teacher_student_cut_mse_val"]),
    },
    {
        "analysis": "Station-blocked stability median",
        "auc": float(stab_map["auc_test"]["median"]),
        "accuracy": float(stab_map["acc_test"]["median"]),
        "average_precision": np.nan,
        "brier": np.nan,
        "teacher_student_cut_mse_val": float(stab_map["teacher_student_cut_mse_val"]["median"]),
    },
    {
        "analysis": "Chronological holdout, index-only no distillation",
        "auc": float(nodistill["auc"]),
        "accuracy": float(nodistill["accuracy"]),
        "average_precision": float(nodistill["average_precision"]),
        "brier": float(nodistill["brier"]),
        "teacher_student_cut_mse_val": np.nan,
    },
]
pd.DataFrame(summary_rows).to_csv(T / "11_main_model_validation_summary.csv", index=False)

chron_w = pd.read_csv(T / "01_chronological_index_weights.csv").set_index("variable")["alpha_stdZ"]
spatial_w = pd.read_csv(T / "02_spatial_blocked_index_weights.csv").set_index("variable")["alpha_stdZ"]
stab_w = {
    q.removeprefix("alpha_"): float(row["median"])
    for q, row in stab_map.items()
    if q.startswith("alpha_") and q != "alpha_bias_stdZ"
}
variables = sorted(set(chron_w.index) & set(spatial_w.index) & set(stab_w))
comparison = pd.DataFrame({
    "variable": variables,
    "chronological_alpha": [float(chron_w[v]) for v in variables],
    "spatial_blocked_alpha": [float(spatial_w[v]) for v in variables],
    "stability_median_alpha": [float(stab_w[v]) for v in variables],
})
comparison["same_sign_all_three"] = (
    np.sign(comparison["chronological_alpha"]) == np.sign(comparison["spatial_blocked_alpha"])
) & (
    np.sign(comparison["chronological_alpha"]) == np.sign(comparison["stability_median_alpha"])
)
comparison.to_csv(T / "11_index_weight_comparison.csv", index=False)


def cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

similarity = pd.DataFrame([
    {"comparison": "Chronological vs spatial blocked", "cosine_similarity": cosine(comparison.chronological_alpha, comparison.spatial_blocked_alpha)},
    {"comparison": "Chronological vs stability median", "cosine_similarity": cosine(comparison.chronological_alpha, comparison.stability_median_alpha)},
    {"comparison": "Spatial blocked vs stability median", "cosine_similarity": cosine(comparison.spatial_blocked_alpha, comparison.stability_median_alpha)},
])
similarity.to_csv(T / "11_index_direction_similarity.csv", index=False)

core = ["Elevation", "RH", "Longitude", "gridded_data_pres", "lapse_rate"]
core_cmp = comparison[comparison.variable.isin(core)].copy()
core_similarity = pd.DataFrame([
    {"comparison": "Chronological vs spatial blocked", "cosine_similarity": cosine(core_cmp.chronological_alpha, core_cmp.spatial_blocked_alpha)},
    {"comparison": "Chronological vs stability median", "cosine_similarity": cosine(core_cmp.chronological_alpha, core_cmp.stability_median_alpha)},
    {"comparison": "Spatial blocked vs stability median", "cosine_similarity": cosine(core_cmp.spatial_blocked_alpha, core_cmp.stability_median_alpha)},
])
core_similarity.to_csv(T / "11_dominant_driver_similarity.csv", index=False)

shares = pd.read_csv(T / "10_temperature_contribution_shares.csv")
student_teacher = pd.read_csv(C / "08_student_teacher_summary.csv")
station_summary = pd.read_csv(S / "07_distill_station_summary.csv").iloc[0]
threshold_params = pd.read_csv(T / "01_chronological_threshold_index_parameters.csv")

key = {
    "chronological": chron,
    "spatial_blocked": spatial,
    "nodistill": nodistill,
    "stability_auc_median": float(stab_map["auc_test"]["median"]),
    "stability_auc_q025": float(stab_map["auc_test"]["q025"]),
    "stability_auc_q975": float(stab_map["auc_test"]["q975"]),
    "stability_accuracy_median": float(stab_map["acc_test"]["median"]),
    "stability_ordering_fraction": float(stab_map["ordering_fraction"]["median"]),
    "station_index_dew_pearson": float(station_summary["pearson_index_dew"]),
    "station_index_dew_spearman": float(station_summary["spearman_index_dew"]),
    "temperature_contribution_shares": shares.to_dict(orient="records"),
    "student_teacher_agreement": student_teacher.to_dict(orient="records"),
    "threshold_index_parameters": threshold_params.to_dict(orient="records"),
    "index_direction_similarity": similarity.to_dict(orient="records"),
    "dominant_driver_similarity": core_similarity.to_dict(orient="records"),
}
json_dump(T / "11_key_manuscript_values.json", key)
print("[ok] Manuscript tables and key-value snapshot created")
print(pd.DataFrame(summary_rows).to_string(index=False))
