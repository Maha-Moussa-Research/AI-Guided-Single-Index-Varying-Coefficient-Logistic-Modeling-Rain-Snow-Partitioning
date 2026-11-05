from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_utils import ensure_dirs, json_dump, pearson, project_root, spearman

ROOT = project_root(__file__)
PATHS = ensure_dirs(ROOT)
STUDENT = PATHS["station"] / "07_distill_station_means.csv"
TEACHER = PATHS["teacher"] / "06_teacher_station_means.csv"
OUT = PATHS["comparison"]

for p in [STUDENT, TEACHER]:
    if not p.exists():
        raise FileNotFoundError(f"Required station summary missing: {p}")

student = pd.read_csv(STUDENT)
teacher = pd.read_csv(TEACHER)
student["Longitude"] = student["Longitude"].round(6)
student["Latitude"] = student["Latitude"].round(6)
teacher["Longitude"] = teacher["Longitude"].round(6)
teacher["Latitude"] = teacher["Latitude"].round(6)

s_keep = ["Longitude", "Latitude", "Cls", "Elevation", "thr_Air_C", "thr_WBT_C", "thr_Dew_C", "Index_I"]
t_keep = ["Longitude", "Latitude", "Cls", "Elevation", "thr_Air_C", "thr_WBT_C", "thr_Dew_C"]
student = student[[c for c in s_keep if c in student.columns]].rename(columns={
    "Cls": "student_Cls", "Elevation": "student_Elevation",
    "thr_Air_C": "student_thr_Air_C", "thr_WBT_C": "student_thr_WBT_C", "thr_Dew_C": "student_thr_Dew_C",
})
teacher = teacher[[c for c in t_keep if c in teacher.columns]].rename(columns={
    "Cls": "teacher_Cls", "Elevation": "teacher_Elevation",
    "thr_Air_C": "teacher_thr_Air_C", "thr_WBT_C": "teacher_thr_WBT_C", "thr_Dew_C": "teacher_thr_Dew_C",
})

compare = student.merge(teacher, on=["Longitude", "Latitude"], how="inner", validate="one_to_one")
compare["climate_group"] = compare.get("student_Cls", compare.get("teacher_Cls", "unknown"))
if "student_Cls" in compare and "teacher_Cls" in compare:
    compare["climate_match"] = compare["student_Cls"].astype(str) == compare["teacher_Cls"].astype(str)

rows = []
for label, suffix in [("Air", "Air"), ("WBT", "WBT"), ("Dew", "Dew")]:
    s = f"student_thr_{suffix}_C"
    t = f"teacher_thr_{suffix}_C"
    d = compare[s] - compare[t]
    compare[f"diff_{suffix}_C"] = d
    rows.append({
        "variable": label,
        "n_station": int(compare[[s, t]].dropna().shape[0]),
        "pearson": pearson(compare[s], compare[t]),
        "spearman": spearman(compare[s], compare[t]),
        "mae_C": float(np.nanmean(np.abs(d))),
        "rmse_C": float(np.sqrt(np.nanmean(d ** 2))),
        "mean_diff_C": float(np.nanmean(d)),
        "median_diff_C": float(np.nanmedian(d)),
        "loa_lower_C": float(np.nanmean(d) - 1.96 * np.nanstd(d, ddof=1)),
        "loa_upper_C": float(np.nanmean(d) + 1.96 * np.nanstd(d, ddof=1)),
    })

summary = pd.DataFrame(rows)
compare.to_csv(OUT / "08_student_teacher_station_compare.csv", index=False)
summary.to_csv(OUT / "08_student_teacher_summary.csv", index=False)
json_dump(OUT / "08_student_teacher_summary.json", summary.to_dict(orient="records"))

climate_rows = []
for climate, sub in compare.groupby("climate_group", dropna=False):
    for label, suffix in [("Air", "Air"), ("WBT", "WBT"), ("Dew", "Dew")]:
        s = f"student_thr_{suffix}_C"
        t = f"teacher_thr_{suffix}_C"
        pair = sub[[s, t]].dropna()
        diff = pair[s] - pair[t]
        climate_rows.append({
            "climate_group": climate,
            "variable": label,
            "n_station": int(len(pair)),
            "pearson": pearson(pair[s], pair[t]) if len(pair) >= 3 else np.nan,
            "spearman": spearman(pair[s], pair[t]) if len(pair) >= 3 else np.nan,
            "mae_C": float(np.mean(np.abs(diff))) if len(pair) else np.nan,
            "rmse_C": float(np.sqrt(np.mean(diff ** 2))) if len(pair) else np.nan,
            "mean_diff_C": float(np.mean(diff)) if len(pair) else np.nan,
        })
climate = pd.DataFrame(climate_rows).sort_values(["variable", "climate_group"])
climate.to_csv(OUT / "08_student_teacher_by_climate.csv", index=False)

print("[ok] Student-teacher comparison complete")
print(summary.to_string(index=False))
