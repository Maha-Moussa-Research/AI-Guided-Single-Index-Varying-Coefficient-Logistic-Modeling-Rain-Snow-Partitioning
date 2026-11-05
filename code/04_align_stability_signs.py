# ===================== Align stability index directions =====================
# Align each stability replicate's single-index coefficient vector to the
# canonical chronological model. This is post-processing only: it does not
# change fitted predictions, cutpoints, or performance metrics.

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parent.parent

TABLE_DIR = REPO_ROOT / "outputs" / "data" / "tables"

RUNS_CSV = TABLE_DIR / "03_stability_runs.csv"
REFERENCE_WEIGHTS_CSV = TABLE_DIR / "01_chronological_index_weights.csv"

OUT_RUNS = TABLE_DIR / "04_stability_runs_sign_aligned.csv"
OUT_SUMMARY = TABLE_DIR / "04_stability_summary_sign_aligned.csv"

for required in (RUNS_CSV, REFERENCE_WEIGHTS_CSV):
    if not required.exists():
        raise FileNotFoundError(f"Required file not found: {required}")

runs = pd.read_csv(RUNS_CSV)
reference = pd.read_csv(REFERENCE_WEIGHTS_CSV)

required_reference_columns = {"variable", "alpha_stdZ"}
missing_reference = required_reference_columns.difference(reference.columns)
if missing_reference:
    raise KeyError(
        f"Reference-weight file is missing columns: {sorted(missing_reference)}"
    )

# Replicate columns are named alpha_<variable>.
candidate_alpha_cols = [
    c for c in runs.columns
    if c.startswith("alpha_") and c != "alpha_bias_stdZ"
]

reference_map = {
    str(row["variable"]): float(row["alpha_stdZ"])
    for _, row in reference.iterrows()
}

aligned_alpha_cols = []
reference_vector = []

for col in candidate_alpha_cols:
    variable = col.removeprefix("alpha_")
    if variable in reference_map:
        aligned_alpha_cols.append(col)
        reference_vector.append(reference_map[variable])

if not aligned_alpha_cols:
    raise RuntimeError(
        "No replicate alpha columns matched the chronological reference weights."
    )

reference_vector = np.asarray(reference_vector, dtype=float)
reference_norm = np.linalg.norm(reference_vector)
if not np.isfinite(reference_norm) or reference_norm == 0:
    raise ValueError("Chronological reference alpha vector is invalid.")

# Quantities that change sign when the index orientation I -> -I is reversed.
signed_index_columns = [
    c for c in [
        "alpha_bias_stdZ",
        "corr_index_dew",
        "corr_index_wbt",
        "corr_index_air",
    ]
    if c in runs.columns
]
signed_columns = aligned_alpha_cols + signed_index_columns

aligned = runs.copy()
flip_flags: list[int] = []
dots: list[float] = []
cosines: list[float] = []

for idx, row in aligned.iterrows():
    replicate_vector = pd.to_numeric(
        row[aligned_alpha_cols], errors="coerce"
    ).to_numpy(dtype=float)

    if not np.all(np.isfinite(replicate_vector)):
        flip_flags.append(0)
        dots.append(np.nan)
        cosines.append(np.nan)
        continue

    replicate_norm = np.linalg.norm(replicate_vector)
    dot = float(np.dot(replicate_vector, reference_vector))
    cosine = (
        dot / (replicate_norm * reference_norm)
        if replicate_norm > 0
        else np.nan
    )
    flip = dot < 0

    if flip:
        aligned.loc[idx, signed_columns] = (
            pd.to_numeric(aligned.loc[idx, signed_columns], errors="coerce") * -1.0
        )

    flip_flags.append(int(flip))
    dots.append(dot)
    cosines.append(float(cosine) if np.isfinite(cosine) else np.nan)

aligned["sign_flipped"] = flip_flags
aligned["alignment_dot_before_flip"] = dots
aligned["alignment_cosine_before_flip"] = cosines
aligned["alignment_cosine_after_flip"] = np.abs(
    aligned["alignment_cosine_before_flip"]
)


def summarize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    exclude = {"label", "sign_flipped"}
    rows = []

    for column in df.columns:
        if column in exclude:
            continue

        values = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy()
        if values.size == 0:
            continue

        rows.append(
            {
                "quantity": column,
                "n_reps": int(values.size),
                "median": float(np.median(values)),
                "q025": float(np.quantile(values, 0.025)),
                "q975": float(np.quantile(values, 0.975)),
                "mean": float(np.mean(values)),
                "sd": (
                    float(np.std(values, ddof=1))
                    if values.size > 1
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


summary = summarize_numeric(aligned)

aligned.to_csv(OUT_RUNS, index=False)
summary.to_csv(OUT_SUMMARY, index=False)

print(f"[ok] Saved aligned replicate results -> {OUT_RUNS}")
print(f"[ok] Saved aligned summary -> {OUT_SUMMARY}")
print(
    f"[alignment] Replicates flipped: "
    f"{int(np.sum(flip_flags))}/{len(flip_flags)}"
)
print(
    "[alignment] Median absolute cosine similarity after alignment: "
    f"{np.nanmedian(aligned['alignment_cosine_after_flip']):.6f}"
)

key_quantities = [
    "auc_test",
    "acc_test",
    "ordering_fraction",
    "teacher_student_cut_mse_val",
    "corr_index_dew",
    "corr_index_wbt",
    "corr_index_air",
    "corr_student_teacher_wbt",
    "mae_student_teacher_wbt",
    "rmse_student_teacher_wbt",
] + aligned_alpha_cols

print("\nKEY SIGN-ALIGNED RESULTS:")
print(
    summary.loc[summary["quantity"].isin(key_quantities)]
    .to_string(index=False)
)
