from __future__ import annotations

"""Create publication figures for the clean index-only VC-Student project.

Expected location after download:
    <project>/code/05_make_main_figures.py

The script reads the clean outputs created by steps 01--08 and writes both
600-dpi PNG and vector PDF files to:
    outputs/figures

It samples the multi-gigabyte row-level coefficient/threshold files in chunks,
so they are never loaded fully into memory.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

THIS = Path(__file__).resolve()
PROJECT = THIS.parent.parent
OUT = PROJECT / "outputs"
DATA_OUT = OUT / "data"
TABLE_DIR = DATA_OUT / "tables"
PRED_DIR = DATA_OUT / "predictions"
FIG_DIR = OUT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DPI = 600
RNG = np.random.default_rng(2026)

PRETTY = {
    "RH": "Relative humidity",
    "gridded_data_pres": "Pressure",
    "Elevation": "Elevation",
    "Longitude": "Longitude",
    "Latitude": "Latitude",
    "Z0": r"$Z_0$",
    "ApT": "ApT",
    "lapse_rate": "Lapse rate",
    "temp_gradient": "Temperature gradient",
}


def pretty(name: str) -> str:
    return PRETTY.get(name, name)


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {png}")
    print(f"[ok] {pdf}")


def finite_xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def binned_mean(x, y, bins: int = 35, min_n: int = 50) -> pd.DataFrame:
    x, y = finite_xy(x, y)
    if len(x) == 0:
        return pd.DataFrame(columns=["x", "mean", "q25", "q75", "n"])
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if len(edges) < 4:
        return pd.DataFrame(columns=["x", "mean", "q25", "q75", "n"])
    groups = np.digitize(x, edges[1:-1], right=True)
    rows = []
    for idx in range(len(edges) - 1):
        mask = groups == idx
        if mask.sum() < min_n:
            continue
        rows.append(
            {
                "x": float(np.mean(x[mask])),
                "mean": float(np.mean(y[mask])),
                "q25": float(np.quantile(y[mask], 0.25)),
                "q75": float(np.quantile(y[mask], 0.75)),
                "n": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def sample_aligned_row_outputs(max_rows: int = 220_000, chunksize: int = 250_000) -> pd.DataFrame:
    coef_path = require(TABLE_DIR / "01_chronological_coefficients_all.csv")
    thr_path = require(TABLE_DIR / "01_chronological_thresholds_all.csv")

    coef_cols = [
        "Index_I",
        "beta0",
        "beta_Air_Temp",
        "beta_Wet_Bulb_Temp",
        "beta_Dewpoint",
    ]
    thr_cols = ["thr_Dew_C", "thr_WBT_C", "thr_Air_C"]

    samples = []
    remaining = max_rows

    coef_iter = pd.read_csv(coef_path, usecols=coef_cols, chunksize=chunksize)
    thr_iter = pd.read_csv(thr_path, usecols=thr_cols, chunksize=chunksize)

    for coef_chunk, thr_chunk in zip(coef_iter, thr_iter):
        if len(coef_chunk) != len(thr_chunk):
            raise RuntimeError("Coefficient and threshold chunks are not row-aligned.")

        # Keep a roughly uniform sample while respecting the global cap.
        take = min(max(1, max_rows // 80), len(coef_chunk), remaining)
        if take <= 0:
            break
        idx = RNG.choice(len(coef_chunk), size=take, replace=False)
        part = pd.concat(
            [
                coef_chunk.iloc[idx].reset_index(drop=True),
                thr_chunk.iloc[idx].reset_index(drop=True),
            ],
            axis=1,
        )
        samples.append(part)
        remaining -= len(part)
        if remaining <= 0:
            break

    if not samples:
        raise RuntimeError("No row-level data were sampled.")

    result = pd.concat(samples, ignore_index=True)
    print(f"[sample] retained {len(result):,} aligned rows")
    return result


def figure_validation_summary() -> None:
    path = require(TABLE_DIR / "08_main_model_validation_summary.csv")
    df = pd.read_csv(path)
    labels = [
        "Chronological",
        "Spatial blocked",
        "Station stability",
    ]
    x = np.arange(len(df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(x - width / 2, df["AUC"], width, label="AUC")
    ax.bar(x + width / 2, df["Accuracy"], width, label="Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.88, 1.0)
    ax.set_ylabel("Performance")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    for xpos, auc, acc in zip(x, df["AUC"], df["Accuracy"]):
        ax.text(xpos - width / 2, auc + 0.002, f"{auc:.3f}", ha="center", fontsize=9)
        ax.text(xpos + width / 2, acc + 0.002, f"{acc:.3f}", ha="center", fontsize=9)
    save(fig, "Figure_01_validation_performance")


def figure_index_weights() -> None:
    path = require(TABLE_DIR / "05_index_weight_comparison.csv")
    df = pd.read_csv(path)
    df = df.sort_values("StabilityMedian", key=lambda s: np.abs(s), ascending=True)
    y = np.arange(len(df))

    med = df["StabilityMedian"].to_numpy(float)
    lo = df["StabilityQ025"].to_numpy(float)
    hi = df["StabilityQ975"].to_numpy(float)
    xerr = np.vstack([med - lo, hi - med])

    fig, ax = plt.subplots(figsize=(8.2, 5.5))
    ax.errorbar(
        med,
        y,
        xerr=xerr,
        fmt="o",
        capsize=3,
        label="Station-resampling median (95% range)",
    )
    ax.scatter(df["ChronologicalAlpha"], y + 0.15, marker="s", label="Chronological")
    ax.scatter(df["SpatialBlockedAlpha"], y - 0.15, marker="^", label="Spatial blocked")
    ax.axvline(0, linewidth=1, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels([pretty(v) for v in df["Variable"]])
    ax.set_xlabel(r"Identified index coefficient, $\alpha_j$")
    ax.legend(frameon=False, loc="best")
    ax.grid(axis="x", alpha=0.25)
    save(fig, "Figure_02_index_weight_stability")


def figure_direction_similarity() -> None:
    all_path = require(TABLE_DIR / "06_index_direction_similarity.csv")
    core_path = require(TABLE_DIR / "07_dominant_driver_similarity.csv")
    all_df = pd.read_csv(all_path).rename(columns={"CosineSimilarity": "All drivers"})
    core_df = pd.read_csv(core_path).rename(columns={"CosineSimilarity": "Five dominant drivers"})
    df = all_df.merge(core_df, on="Comparison", how="inner")

    labels = [
        "Chronological vs\nspatial blocked",
        "Chronological vs\nstability median",
        "Spatial blocked vs\nstability median",
    ]
    y = np.arange(len(df))
    h = 0.34
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    ax.barh(y + h / 2, df["All drivers"], height=h, label="All drivers")
    ax.barh(y - h / 2, df["Five dominant drivers"], height=h, label="Five dominant drivers")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.70, 1.01)
    ax.set_xlabel("Cosine similarity")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.25)
    for i, row in df.iterrows():
        ax.text(row["All drivers"] + 0.004, i + h / 2, f"{row['All drivers']:.3f}", va="center", fontsize=9)
        ax.text(row["Five dominant drivers"] + 0.004, i - h / 2, f"{row['Five dominant drivers']:.3f}", va="center", fontsize=9)
    save(fig, "Figure_03_index_direction_similarity")


def load_prediction_sets():
    specs = [
        ("Chronological", PRED_DIR / "01_chronological_test_predictions.csv"),
        ("Spatial blocked", PRED_DIR / "02_spatial_blocked_test_predictions.csv"),
    ]
    out = []
    for label, path in specs:
        df = pd.read_csv(require(path), usecols=["y_true", "y_prob"])
        y = df["y_true"].to_numpy(int)
        p = df["y_prob"].to_numpy(float)
        out.append((label, y, p))
    return out


def figure_roc_pr() -> None:
    pred_sets = load_prediction_sets()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1))

    for label, y, p in pred_sets:
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        auc = roc_auc_score(y, p)
        ap = average_precision_score(y, p)
        axes[0].plot(fpr, tpr, linewidth=2, label=f"{label} (AUC={auc:.3f})")
        axes[1].plot(recall, precision, linewidth=2, label=f"{label} (AP={ap:.3f})")

    axes[0].plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    axes[0].set_xlabel("False-positive rate")
    axes[0].set_ylabel("True-positive rate")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].legend(frameon=False)

    prevalence = np.mean(pred_sets[0][1])
    axes[1].axhline(prevalence, linestyle="--", linewidth=1)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].legend(frameon=False)
    save(fig, "Figure_04_roc_precision_recall")


def calibration_points(y, p, bins=20):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(p, edges[1:-1], right=True)
    rows = []
    for b in range(bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        rows.append((float(np.mean(p[mask])), float(np.mean(y[mask])), int(mask.sum())))
    return pd.DataFrame(rows, columns=["predicted", "observed", "n"])


def figure_calibration() -> None:
    pred_sets = load_prediction_sets()
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    for label, y, p in pred_sets:
        cal = calibration_points(y, p)
        brier = brier_score_loss(y, p)
        ax.plot(cal["predicted"], cal["observed"], marker="o", linewidth=2, label=f"{label} (Brier={brier:.3f})")
    ax.set_xlabel("Predicted rain probability")
    ax.set_ylabel("Observed rain frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    save(fig, "Figure_05_probability_calibration")


def figure_thresholds_vs_index(sample: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for column, label in [
        ("thr_Dew_C", "Dewpoint cutoff"),
        ("thr_WBT_C", "Wet-bulb cutoff"),
        ("thr_Air_C", "Air-temperature cutoff"),
    ]:
        curve = binned_mean(sample["Index_I"], sample[column], bins=40, min_n=40)
        ax.plot(curve["x"], curve["mean"], linewidth=2.2, label=label)
        ax.fill_between(curve["x"], curve["q25"], curve["q75"], alpha=0.14)
    ax.set_xlabel(r"Learned environmental index, $I(\mathbf{Z})$")
    ax.set_ylabel("Cutoff (°C)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    save(fig, "Figure_06_cutoffs_vs_index")


def figure_beta_vs_index(sample: pd.DataFrame) -> None:
    columns = [
        ("beta0", r"$\beta_0(I)$"),
        ("beta_Air_Temp", r"$\beta_{Air}(I)$"),
        ("beta_Wet_Bulb_Temp", r"$\beta_{WBT}(I)$"),
        ("beta_Dewpoint", r"$\beta_{Dew}(I)$"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.0))
    for ax, (column, label) in zip(axes.ravel(), columns):
        curve = binned_mean(sample["Index_I"], sample[column], bins=40, min_n=40)
        ax.plot(curve["x"], curve["mean"], linewidth=2.2)
        ax.fill_between(curve["x"], curve["q25"], curve["q75"], alpha=0.16)
        ax.axhline(0, linestyle="--", linewidth=0.8)
        ax.set_xlabel(r"$I(\mathbf{Z})$")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    save(fig, "Figure_07_varying_coefficients_vs_index")


def figure_threshold_distributions(sample: pd.DataFrame) -> None:
    columns = ["thr_Dew_C", "thr_WBT_C", "thr_Air_C"]
    labels = ["Dewpoint", "Wet bulb", "Air temperature"]
    arrays = [sample[c].dropna().to_numpy(float) for c in columns]
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    parts = ax.violinplot(arrays, showmeans=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_alpha(0.65)
    ax.set_xticks(np.arange(1, 4))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Learned cutoff (°C)")
    ax.grid(axis="y", alpha=0.25)
    save(fig, "Figure_08_cutoff_distributions")


def write_manifest() -> None:
    files = sorted(FIG_DIR.glob("Figure_*.*"))
    manifest = pd.DataFrame(
        {
            "filename": [p.name for p in files],
            "format": [p.suffix.lstrip(".") for p in files],
            "size_bytes": [p.stat().st_size for p in files],
        }
    )
    manifest_path = TABLE_DIR / "09_figure_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"[ok] {manifest_path}")

    metadata = {
        "project": str(PROJECT),
        "figure_directory": str(FIG_DIR),
        "figures": [str(p) for p in files],
    }
    (OUT / "figure_artifacts_index.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main() -> None:
    print(f"[paths] PROJECT -> {PROJECT}")
    print(f"[paths] FIG_DIR -> {FIG_DIR}")

    figure_validation_summary()
    figure_index_weights()
    figure_direction_similarity()
    figure_roc_pr()
    figure_calibration()

    sampled = sample_aligned_row_outputs()
    figure_thresholds_vs_index(sampled)
    figure_beta_vs_index(sampled)
    figure_threshold_distributions(sampled)
    write_manifest()

    print("\nALL CORE INDEX-ONLY FIGURES COMPLETED")


if __name__ == "__main__":
    main()
