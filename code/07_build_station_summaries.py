from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_utils import ensure_dirs, json_dump, pearson, project_root, spearman

ROOT = project_root(__file__)
PATHS = ensure_dirs(ROOT)
TABLE = PATHS["tables"]
STATION = PATHS["station"]
CHUNK = 400_000
ROUND_COORDS = 6
SAMPLE_PER_CHUNK = 5_000

RUNS = {
    "distill": {
        "coef": TABLE / "01_chronological_coefficients_all.csv",
        "thr": TABLE / "01_chronological_thresholds_all.csv",
        "station": STATION / "07_distill_station_means.csv",
        "sample": STATION / "07_distill_event_sample.csv",
    },
    "nodistill": {
        "coef": TABLE / "05_nodistill_coefficients_all.csv",
        "thr": TABLE / "05_nodistill_thresholds_all.csv",
        "station": STATION / "07_nodistill_station_means.csv",
        "sample": STATION / "07_nodistill_event_sample.csv",
    },
}

COEF_USE = [
    "Longitude", "Latitude", "Elevation", "Index_I", "beta0",
    "beta_Air_Temp", "beta_Wet_Bulb_Temp", "beta_Dewpoint",
]
THR_USE = [
    "Cls", "Longitude", "Latitude", "Elevation", "RH", "gridded_data_pres",
    "Z0", "lapse_rate", "temp_gradient", "ApT",
    "thr_Air_std", "thr_WBT_std", "thr_Dew_std",
    "thr_Air_C", "thr_WBT_C", "thr_Dew_C",
]
VALUE_COLS = [
    "Elevation", "Index_I", "beta0", "beta_Air_Temp", "beta_Wet_Bulb_Temp", "beta_Dewpoint",
    "RH", "gridded_data_pres", "Z0", "lapse_rate", "temp_gradient", "ApT",
    "thr_Air_std", "thr_WBT_std", "thr_Dew_std", "thr_Air_C", "thr_WBT_C", "thr_Dew_C",
]
CORR_COLS = ["Z0", "ApT", "temp_gradient", "lapse_rate"]


def process_run(label: str, paths: dict[str, Path]) -> None:
    for key in ["coef", "thr"]:
        if not paths[key].exists():
            raise FileNotFoundError(f"Missing {label} {key} file: {paths[key]}")

    coef_iter = pd.read_csv(paths["coef"], usecols=lambda c: c in COEF_USE, chunksize=CHUNK, low_memory=False)
    thr_iter = pd.read_csv(paths["thr"], usecols=lambda c: c in THR_USE, chunksize=CHUNK, low_memory=False)

    accumulated: pd.DataFrame | None = None
    samples: list[pd.DataFrame] = []
    corr_n = 0
    corr_sum = np.zeros(len(CORR_COLS), dtype=float)
    corr_cross = np.zeros((len(CORR_COLS), len(CORR_COLS)), dtype=float)
    total_rows = 0

    for chunk_id, (coef, thr) in enumerate(zip(coef_iter, thr_iter), start=1):
        if len(coef) != len(thr):
            raise RuntimeError(f"Row mismatch in {label}, chunk {chunk_id}")
        for coord in ["Longitude", "Latitude"]:
            if coord in coef and coord in thr:
                if not np.allclose(coef[coord].to_numpy(float), thr[coord].to_numpy(float), equal_nan=True):
                    raise RuntimeError(f"Coordinate misalignment in {label}, chunk {chunk_id}")

        frame = thr.copy()
        for c in ["Index_I", "beta0", "beta_Air_Temp", "beta_Wet_Bulb_Temp", "beta_Dewpoint"]:
            frame[c] = pd.to_numeric(coef[c], errors="coerce")
        # Prefer threshold-file elevation because it is exported with all driver columns.
        frame["Longitude"] = pd.to_numeric(frame["Longitude"], errors="coerce").round(ROUND_COORDS)
        frame["Latitude"] = pd.to_numeric(frame["Latitude"], errors="coerce").round(ROUND_COORDS)
        frame["Cls"] = frame["Cls"].astype(str)
        for c in VALUE_COLS:
            if c in frame:
                frame[c] = pd.to_numeric(frame[c], errors="coerce")
        frame = frame.dropna(subset=["Longitude", "Latitude", "Index_I"])
        total_rows += len(frame)

        keys = ["Longitude", "Latitude", "Cls"]
        tmp = frame[keys].copy()
        for c in VALUE_COLS:
            if c not in frame:
                continue
            tmp[f"sum__{c}"] = frame[c].fillna(0.0)
            tmp[f"n__{c}"] = frame[c].notna().astype(np.int64)
        grouped = tmp.groupby(keys, as_index=False).sum(numeric_only=True)
        if accumulated is None:
            accumulated = grouped
        else:
            accumulated = pd.concat([accumulated, grouped], ignore_index=True)
            accumulated = accumulated.groupby(keys, as_index=False).sum(numeric_only=True)

        n_take = min(SAMPLE_PER_CHUNK, len(frame))
        if n_take:
            sample_cols = keys + [c for c in VALUE_COLS if c in frame]
            samples.append(frame[sample_cols].sample(n=n_take, random_state=2025 + chunk_id))

        corr_frame = frame[CORR_COLS].dropna().to_numpy(float)
        if len(corr_frame):
            corr_n += len(corr_frame)
            corr_sum += corr_frame.sum(axis=0)
            corr_cross += corr_frame.T @ corr_frame

        print(f"[{label}] chunk {chunk_id}: {len(frame):,} valid rows; stations accumulated={len(accumulated):,}")

    if accumulated is None:
        raise RuntimeError(f"No rows were aggregated for {label}")

    out = accumulated[["Longitude", "Latitude", "Cls"]].copy()
    for c in VALUE_COLS:
        s, n = f"sum__{c}", f"n__{c}"
        if s in accumulated and n in accumulated:
            out[c] = accumulated[s] / accumulated[n].replace(0, np.nan)
    out.to_csv(paths["station"], index=False)
    pd.concat(samples, ignore_index=True).to_csv(paths["sample"], index=False)

    summary = {
        "run": label,
        "event_rows": int(total_rows),
        "stations": int(len(out)),
        "pearson_index_dew": pearson(out["Index_I"], out["thr_Dew_C"]),
        "spearman_index_dew": spearman(out["Index_I"], out["thr_Dew_C"]),
        "pearson_index_wbt": pearson(out["Index_I"], out["thr_WBT_C"]),
        "pearson_index_air": pearson(out["Index_I"], out["thr_Air_C"]),
    }
    json_dump(STATION / f"07_{label}_station_summary.json", summary)
    pd.DataFrame([summary]).to_csv(STATION / f"07_{label}_station_summary.csv", index=False)
    print(f"[ok] {label} station means -> {paths['station']}")

    if label == "distill" and corr_n > 1:
        mean = corr_sum / corr_n
        cov = (corr_cross - corr_n * np.outer(mean, mean)) / (corr_n - 1)
        sd = np.sqrt(np.diag(cov))
        corr = cov / np.outer(sd, sd)
        corr_df = pd.DataFrame(corr, index=CORR_COLS, columns=CORR_COLS)
        corr_df.to_csv(TABLE / "07_vertical_structure_correlation.csv")
        print("[ok] vertical-structure correlation matrix saved")


for run_label, run_paths in RUNS.items():
    process_run(run_label, run_paths)

print("ALL STATION SUMMARIES COMPLETED")
