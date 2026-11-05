# MAKE_FINAL_PAPER_FIGURES.ps1
# ============================================================================
# Self-contained FINAL figure pipeline for the AOAS rain--snow revision.
#
# PURPOSE
#   - Keep ONE permanent figure-generation code file for GitHub.
#   - Generate/curate ONLY the figure assets used by the FINAL manuscript.
#   - Do NOT refit any model or change any scientific values.
#   - Final Figure 8 is generated WITHOUT the Pearson-correlation annotation.
#
# PUT THIS FILE AT
#   <project>\code\MAKE_FINAL_PAPER_FIGURES.ps1
#
# RUN FROM THE PROJECT ROOT
#   powershell -ExecutionPolicy Bypass -File ".\code\MAKE_FINAL_PAPER_FIGURES.ps1"
#
# OUTPUT
#   <project>\figures_final\
#
# `figures_final` is deliberately separate from the working `figures` folder.
# This makes GitHub cleanup safe: after checking the output, you can keep only
# this curated folder (or rename it to `figures` in the GitHub repository).
#
# FINAL MANUSCRIPT ASSETS CREATED/CURATED
#   Fig. 1  koppen.png
#   Fig. 2  split_coverage_map__distill_spatial_blocked_latlon.png
#   Fig. 3  map_air_threshold__distill.png
#   Fig. 4  R1_threshold_distributions.png
#   Fig. 5  map_air_threshold_student_distill.png
#           map_air_threshold_difference_student_minus_teacher.png
#   Fig. 6  Figure_05_student_teacher_wbt_by_climate.png
#   Fig. 7  Figure_06_index_weights.png
#   Fig. 8  Figure_16_air_index_distillation_comparison.png  [NO Pearson r]
#   Fig. 9  map_index_Iz.png
#   Fig.10  Figure_09_probability_vs_index.png
#   Fig.11  distill__R3_thr_Air_C_vs_all_drivers_facets.png
#   Fig.12  index_effects_per1SD.png
#   Fig.13  distill__profile_corr_heatmap.png
#   Fig.14  distill__R4_beta_vs_index_facets.pdf
#   Fig.15  Figure_13_roc_precision_recall.png
#   Fig.16  Figure_14_reliability.png
#   Fig.17  map_air_threshold__distill.png
#           map_air_threshold__nodistill.png
#
# REQUIRED PRECOMPUTED ANALYSIS OUTPUTS
#   The script uses the final model outputs already produced under outputs/data.
#   It does not train or refit VC--Student/DCut--Teacher models.
#
# PYTHON PACKAGES
#   numpy, pandas, matplotlib, scikit-learn
#
# R PACKAGES
#   readr, dplyr, ggplot2, sf, rnaturalearth, rnaturalearthdata,
#   scales, viridis
# ============================================================================

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ThisFile = $MyInvocation.MyCommand.Path
$CodeDir = Split-Path -Parent $ThisFile
if ((Split-Path -Leaf $CodeDir).ToLowerInvariant() -ne "code") {
    throw "Place MAKE_FINAL_PAPER_FIGURES.ps1 inside the project's code folder."
}

$Project = Split-Path -Parent $CodeDir
$FinalFigDir = Join-Path $Project "figures_final"
$RunId = [Guid]::NewGuid().ToString("N")
$WorkFigDir = Join-Path $Project (".final_figure_build_" + $RunId)

# Start from a clean dedicated final folder. This never deletes <project>\figures.
if (Test-Path -LiteralPath $FinalFigDir) {
    Remove-Item -LiteralPath $FinalFigDir -Recurse -Force
}
New-Item -ItemType Directory -Path $FinalFigDir -Force | Out-Null
New-Item -ItemType Directory -Path $WorkFigDir -Force | Out-Null

Write-Host "============================================================"
Write-Host "AOAS FINAL-PAPER FIGURE PIPELINE"
Write-Host "Project:       $Project"
Write-Host "Build folder:  $WorkFigDir"
Write-Host "Final folder:  $FinalFigDir"
Write-Host "============================================================"

$PythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $PythonCmd) { throw "Python was not found on PATH." }
$Python = $PythonCmd.Source
Write-Host "[tool] Python : $Python"

$PreferredR = "C:\Users\A02381175\AppData\Local\Programs\R\R-4.5.2\bin\x64\Rscript.exe"
if (Test-Path -LiteralPath $PreferredR) {
    $Rscript = $PreferredR
} else {
    $Rcmd = Get-Command Rscript -ErrorAction SilentlyContinue
    if (-not $Rcmd) { throw "Rscript was not found on PATH." }
    $Rscript = $Rcmd.Source
}
Write-Host "[tool] Rscript: $Rscript"

# Expose the project/build locations to the temporary embedded scripts.
$OldProjectEnv = $env:AOAS_PROJECT_ROOT
$OldFigEnv = $env:AOAS_FIG_WORK
$env:AOAS_PROJECT_ROOT = $Project
$env:AOAS_FIG_WORK = $WorkFigDir

$TmpPython = Join-Path $CodeDir ("_tmp_final_nonmap_" + $RunId + ".py")
$TmpContextR = Join-Path $CodeDir ("_tmp_final_context_" + $RunId + ".R")
$TmpMapsR = Join-Path $CodeDir ("_tmp_final_maps_" + $RunId + ".R")
$TmpDiffR = Join-Path $CodeDir ("_tmp_final_difference_" + $RunId + ".R")

$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory=$true)][string]$Path,
        [Parameter(Mandatory=$true)][string]$Text
    )
    [System.IO.File]::WriteAllText($Path, $Text, $Utf8NoBom)
}

$PythonSource = @'
from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(os.environ["AOAS_PROJECT_ROOT"]).resolve()
FIG = Path(os.environ["AOAS_FIG_WORK"]).resolve()
FIG.mkdir(parents=True, exist_ok=True)

OUT = ROOT / "outputs" / "data"
T = OUT / "tables"
S = OUT / "station_summaries"
PR = OUT / "predictions"
DPI = 600

mpl.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": DPI,
    "font.size": 10,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.22,
    "axes.edgecolor": "#777777",
    "axes.linewidth": 0.7,
})

pretty = {
    "RH": "RH",
    "gridded_data_pres": r"$P_s$",
    "Elevation": "Elev",
    "Longitude": "Lon",
    "Latitude": "Lat",
    "Z0": r"$Z_0$",
    "lapse_rate": r"$\Gamma_{ll}$",
    "temp_gradient": r"$\Gamma_b$",
    "ApT": "ApT",
}


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return path


def save_png(fig, filename: str, width: float, height: float) -> None:
    path = FIG / filename
    fig.set_size_inches(width, height)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("[ok]", path)


def save_pdf(fig, filename: str, width: float, height: float) -> None:
    path = FIG / filename
    fig.set_size_inches(width, height)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("[ok]", path)


def binned_curve(frame: pd.DataFrame, xcol: str, ycol: str, n_bins: int = 20) -> pd.DataFrame:
    d = frame[[xcol, ycol]].copy()
    d[xcol] = pd.to_numeric(d[xcol], errors="coerce")
    d[ycol] = pd.to_numeric(d[ycol], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 10:
        raise RuntimeError(f"Too few valid rows for {xcol}: {len(d)}")
    q = max(2, min(int(n_bins), int(d[xcol].nunique())))
    try:
        d["_bin"] = pd.qcut(d[xcol], q=q, duplicates="drop")
    except ValueError:
        d["_bin"] = pd.cut(d[xcol], bins=q, duplicates="drop")
    g = d.groupby("_bin", observed=True)
    out = g.agg(
        x=(xcol, "mean"),
        mean=(ycol, "mean"),
        q25=(ycol, lambda v: float(np.quantile(v, 0.25))),
        q75=(ycol, lambda v: float(np.quantile(v, 0.75))),
    ).reset_index(drop=True)
    return out.replace([np.inf, -np.inf], np.nan).dropna().sort_values("x")


def local_bin_density(x: np.ndarray, y: np.ndarray, bins_x: int = 95, bins_y: int = 95) -> np.ndarray:
    H, xedges, yedges = np.histogram2d(x, y, bins=(bins_x, bins_y))
    ix = np.clip(np.digitize(x, xedges) - 1, 0, H.shape[0] - 1)
    iy = np.clip(np.digitize(y, yedges) - 1, 0, H.shape[1] - 1)
    density = H[ix, iy].astype(float)
    density[density < 1] = 1
    return density


station = pd.read_csv(require(S / "07_distill_station_means.csv"), low_memory=False)
sample = pd.read_csv(require(S / "07_distill_event_sample.csv"), low_memory=False)
pred = pd.read_csv(
    require(PR / "01_chronological_test_predictions.csv"),
    usecols=["y_true", "y_prob", "Index_I"],
    low_memory=False,
).dropna()

# -----------------------------------------------------------------------------
# Final Figure 4: threshold distributions -> R1_threshold_distributions.png
# -----------------------------------------------------------------------------
cols = ["thr_Dew_C", "thr_WBT_C", "thr_Air_C"]
fig, ax = plt.subplots()
data = [station[c].dropna().to_numpy(float) for c in cols]
parts = ax.violinplot(data, showmeans=True, showextrema=False, widths=0.8)
colors = ["#76c7c0", "#80b1d3", "#d9d9d9"]
for body, color in zip(parts["bodies"], colors):
    body.set_facecolor(color)
    body.set_edgecolor("none")
    body.set_alpha(0.90)
parts["cmeans"].set_color("#333333")
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["Dew cut", "WBT cut", "Air cut"])
ax.set_ylabel(r"Threshold ($^\circ$C)")
ax.axhline(0, color="#888888", lw=0.7)
save_png(fig, "R1_threshold_distributions.png", 6.8, 4.6)

# -----------------------------------------------------------------------------
# Final Figure 7: identified contextual-index weights
# -----------------------------------------------------------------------------
w = pd.read_csv(require(T / "01_chronological_index_weights.csv"))
w = w.sort_values("alpha_stdZ", key=lambda v: np.abs(v), ascending=True)
fig, ax = plt.subplots()
ypos = np.arange(len(w))
bar_colors = mpl.colormaps["Set3"](np.linspace(0.05, 0.95, len(w)))
bars = ax.barh(ypos, w["alpha_stdZ"], color=bar_colors, edgecolor="#333333", linewidth=0.6)
ax.set_yticks(ypos)
ax.set_yticklabels([pretty.get(v, v) for v in w["variable"]])
ax.axvline(0, color="#555555", lw=0.8)
span = max(abs(w["alpha_stdZ"].min()), abs(w["alpha_stdZ"].max()))
for bar, val in zip(bars, w["alpha_stdZ"]):
    ax.text(
        val + (0.02 * span if val >= 0 else -0.02 * span),
        bar.get_y() + bar.get_height() / 2,
        f"{val:+.2f}",
        va="center",
        ha="left" if val >= 0 else "right",
        fontsize=8,
    )
ax.set_xlabel(r"Identified $\alpha$ weight (standardized-$Z$ units)")
ax.set_title(r"Index weights $\alpha$ (shared across cuts)")
save_png(fig, "Figure_06_index_weights.png", 7.0, 4.8)

# -----------------------------------------------------------------------------
# Final Figure 8: station-mean air cutpoint vs contextual index, NO Pearson text
# -----------------------------------------------------------------------------
d = station[["Index_I", "thr_Air_C"]].copy()
for c in d.columns:
    d[c] = pd.to_numeric(d[c], errors="coerce")
d = d.replace([np.inf, -np.inf], np.nan).dropna().sort_values("Index_I")
if d.empty:
    raise RuntimeError("No valid station-mean rows available for final Figure 8.")
x = d["Index_I"].to_numpy(float)
y = d["thr_Air_C"].to_numpy(float)
density = local_bin_density(x, y)
order = np.argsort(-density)
x, y, density = x[order], y[order], density[order]
fig, ax = plt.subplots(figsize=(9.7, 6.9))
sc = ax.scatter(
    x,
    y,
    c=density,
    cmap="viridis",
    norm=LogNorm(vmin=1, vmax=float(density.max())),
    s=12,
    alpha=0.72,
    linewidths=0,
    rasterized=True,
    zorder=2,
)
ax.set_xlabel(r"Contextual index $I(\tilde{z})$ (std units)", fontsize=14)
ax.set_ylabel(r"Station-mean air-temperature cutpoint ($^\circ$C)", fontsize=14)
ax.grid(True, linestyle="--", alpha=0.18, linewidth=0.8, zorder=0)
ax.tick_params(labelsize=11)
cb = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.15, fraction=0.07, aspect=42)
cb.set_label("Local point density (count per 2D bin, log scale)", fontsize=12)
cb.ax.tick_params(labelsize=10)
fig.subplots_adjust(bottom=0.19, top=0.98)
fig.savefig(FIG / "Figure_16_air_index_distillation_comparison.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("[ok]", FIG / "Figure_16_air_index_distillation_comparison.png")

# -----------------------------------------------------------------------------
# Final Figure 10: held-out fitted vs empirical rain probability by index bin
# -----------------------------------------------------------------------------
pred["bin"] = pd.qcut(pred["Index_I"], 20, duplicates="drop")
b = pred.groupby("bin", observed=True).agg(
    Index_I=("Index_I", "mean"),
    fitted=("y_prob", "mean"),
    empirical=("y_true", "mean"),
).reset_index(drop=True)
fig, ax = plt.subplots()
ax.plot(b["Index_I"], b["fitted"], "o-", lw=2, ms=4, label="Mean fitted rain probability")
ax.plot(b["Index_I"], b["empirical"], "s--", lw=1.8, ms=3.5, label="Empirical rain frequency")
ax.set_xlabel(r"Learned index $I(Z)$ (std units)")
ax.set_ylabel("Rain probability")
ax.set_ylim(0, 1)
ax.legend(frameon=False, loc="upper left")
save_png(fig, "Figure_09_probability_vs_index.png", 6.8, 4.8)

# -----------------------------------------------------------------------------
# Final Figure 11: air cutpoint vs nine environmental drivers (binned means/IQR)
# -----------------------------------------------------------------------------
drivers = [
    ("RH", "RH"),
    ("gridded_data_pres", r"$P_s$"),
    ("Z0", r"$Z_0$"),
    ("lapse_rate", r"$\Gamma_{ll}$"),
    ("temp_gradient", r"$\Gamma_b$"),
    ("ApT", "ApT"),
    ("Elevation", "Elev"),
    ("Longitude", "Lon"),
    ("Latitude", "Lat"),
]
needed = ["thr_Air_C"] + [x for x, _ in drivers]
missing = [c for c in needed if c not in sample.columns]
if missing:
    raise RuntimeError("Event sample missing Figure 11 columns: " + ", ".join(missing))
fig, axes = plt.subplots(3, 3, figsize=(10.6, 7.6), squeeze=False)
cycle = mpl.rcParams["axes.prop_cycle"].by_key().get("color", [])
curves = []
for k, (ax, (driver, label)) in enumerate(zip(axes.ravel(), drivers)):
    curve = binned_curve(sample, driver, "thr_Air_C", 20)
    curves.append(curve)
    color = cycle[k % len(cycle)] if cycle else None
    ax.plot(curve["x"], curve["mean"], lw=2.0, color=color)
    ax.fill_between(curve["x"], curve["q25"], curve["q75"], alpha=0.18, color=color)
    ax.set_xlabel(label)
    ax.set_ylabel("Air cut")
ymin = min(float(c["q25"].min()) for c in curves)
ymax = max(float(c["q75"].max()) for c in curves)
pad = max(0.05, 0.08 * (ymax - ymin if ymax > ymin else 1.0))
for ax in axes.ravel():
    ax.set_ylim(ymin - pad, ymax + pad)
fig.tight_layout(w_pad=1.3, h_pad=1.2)
fig.savefig(FIG / "distill__R3_thr_Air_C_vs_all_drivers_facets.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("[ok]", FIG / "distill__R3_thr_Air_C_vs_all_drivers_facets.png")

# -----------------------------------------------------------------------------
# Final Figure 12: finite-difference threshold sensitivities per +1 SD
# -----------------------------------------------------------------------------
effects = pd.read_csv(require(T / "01_chronological_index_effects_per_1SD.csv"))
effects["label"] = effects["driver"].map(pretty).fillna(effects["driver"])
fig, axes = plt.subplots(1, 3, sharex=True)
for ax, col, title in zip(
    axes,
    ["thr_Air_C_per1SD", "thr_WBT_C_per1SD", "thr_Dew_C_per1SD"],
    ["Air cut", "WBT cut", "Dew cut"],
):
    dd = effects.sort_values(col)
    colors = mpl.colormaps["Set3"](np.linspace(0.05, 0.95, len(dd)))
    bars = ax.barh(dd["label"], dd[col], color=colors, edgecolor="#555555", linewidth=0.4)
    ax.axvline(0, color="#555555", lw=0.7)
    ax.set_title(title)
    ax.set_xlabel(r"Effect ($^\circ$C)")
    for bar, val in zip(bars, dd[col]):
        ax.text(
            val,
            bar.get_y() + bar.get_height() / 2,
            f" {val:+.2f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=7,
        )
save_png(fig, "index_effects_per1SD.png", 13.0, 4.4)

# -----------------------------------------------------------------------------
# Final Figure 13: correlation matrix of vertical-structure predictors
# -----------------------------------------------------------------------------
corr = pd.read_csv(require(T / "07_vertical_structure_correlation.csv"), index_col=0)
fig, ax = plt.subplots()
im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
labels = [pretty.get(c, c) for c in corr.columns]
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        if i != j:
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
cb = fig.colorbar(im, ax=ax, shrink=0.75)
cb.set_label("corr")
save_png(fig, "distill__profile_corr_heatmap.png", 5.6, 4.8)

# -----------------------------------------------------------------------------
# Final Figure 14: index-varying coefficient functions (manuscript uses PDF)
# -----------------------------------------------------------------------------
coef_cols = [
    ("beta0", r"$\beta_0$"),
    ("beta_Air_Temp", r"$\beta_{Air}$"),
    ("beta_Dewpoint", r"$\beta_{Dew}$"),
    ("beta_Wet_Bulb_Temp", r"$\beta_{WBT}$"),
]
fig, axes = plt.subplots(2, 2, squeeze=False)
for ax, (col, title) in zip(axes.ravel(), coef_cols):
    dd = sample[["Index_I", col]].dropna().sort_values("Index_I").copy()
    dd["bin"] = pd.qcut(dd["Index_I"], 80, duplicates="drop")
    curve = dd.groupby("bin", observed=True).agg(
        x=("Index_I", "mean"),
        y=(col, "median"),
        lo=(col, lambda v: np.quantile(v, 0.10)),
        hi=(col, lambda v: np.quantile(v, 0.90)),
    ).reset_index(drop=True)
    ax.fill_between(curve["x"], curve["lo"], curve["hi"], alpha=0.18)
    ax.plot(curve["x"], curve["y"], color="#6b1f2b", lw=2)
    ax.set_title(f"{title} vs Learned index $I(Z)$ (std units)")
    ax.set_xlabel(r"Learned index $I(Z)$ (std units)")
    ax.set_ylabel("Coefficient")
save_pdf(fig, "distill__R4_beta_vs_index_facets.pdf", 9.2, 6.6)

# -----------------------------------------------------------------------------
# Final Figure 15: ROC and precision-recall
# -----------------------------------------------------------------------------
y_true = pred["y_true"].to_numpy(int)
p_hat = pred["y_prob"].to_numpy(float)
fpr, tpr, _ = roc_curve(y_true, p_hat)
precision, recall, _ = precision_recall_curve(y_true, p_hat)
auc = roc_auc_score(y_true, p_hat)
ap = average_precision_score(y_true, p_hat)
fig, axes = plt.subplots(1, 2)
axes[0].plot(fpr, tpr, lw=2)
axes[0].plot([0, 1], [0, 1], "--", lw=1)
axes[0].set_xlabel("FPR")
axes[0].set_ylabel("TPR")
axes[0].set_title("ROC curve")
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)
axes[1].plot(recall, precision, lw=2)
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-recall curve")
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1)
save_png(fig, "Figure_13_roc_precision_recall.png", 8.5, 3.5)
print(f"[metric] Figure 15 AUC={auc:.4f}; AP={ap:.3f}")

# -----------------------------------------------------------------------------
# Final Figure 16: reliability diagram
# -----------------------------------------------------------------------------
bins = np.linspace(0, 1, 21)
ids = np.digitize(p_hat, bins[1:-1], right=True)
rows = []
for i in range(20):
    m = ids == i
    if m.any():
        rows.append((p_hat[m].mean(), y_true[m].mean(), int(m.sum())))
rel = pd.DataFrame(rows, columns=["pred", "emp", "n"])
ece = float((np.abs(rel.pred - rel.emp) * rel.n).sum() / rel.n.sum())
brier = brier_score_loss(y_true, p_hat)
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], "--", lw=1)
ax.plot(rel.pred, rel.emp, "o-", lw=2)
ax.set_xlabel("Predicted probability")
ax.set_ylabel("Empirical frequency")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title(f"Reliability (ECE={ece:.3f}, Brier={brier:.3f})")
save_png(fig, "Figure_14_reliability.png", 4.8, 4.8)
print(f"[metric] Figure 16 ECE={ece:.3f}; Brier={brier:.4f}")

print("PASS: FINAL NON-MAP FIGURES GENERATED")

'@

$ContextRSource = @'
suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(ggplot2)
})

ROOT <- normalizePath(Sys.getenv("AOAS_PROJECT_ROOT"), winslash = "/", mustWork = TRUE)
FIG  <- normalizePath(Sys.getenv("AOAS_FIG_WORK"), winslash = "/", mustWork = FALSE)
dir.create(FIG, recursive = TRUE, showWarnings = FALSE)

# -----------------------------------------------------------------------------
# Final Figure 1: Köppen-Geiger climate-zone context map.
# The manuscript uses broad display groups while analyses retain 29 three-letter
# climate classes. This map is for geographic context only.
# -----------------------------------------------------------------------------
kg_file <- file.path(ROOT, "data", "Koeppen-Geiger-ASCII.txt")
if (!file.exists(kg_file)) stop("Missing Köppen-Geiger grid: ", kg_file)

kg <- read.table(
  kg_file,
  header = TRUE,
  sep = "",
  fill = TRUE,
  stringsAsFactors = FALSE
) %>%
  transmute(
    Lon = as.numeric(Lon),
    Lat = as.numeric(Lat),
    Cls = as.character(Cls)
  ) %>%
  filter(is.finite(Lon), is.finite(Lat), !is.na(Cls), Cls != "") %>%
  mutate(
    display_group = case_when(
      Cls == "Af" ~ "Af",
      Cls == "Am" ~ "Am",
      Cls == "As" ~ "As",
      Cls == "Aw" ~ "Aw",
      grepl("^BS", Cls) ~ "BS",
      grepl("^BW", Cls) ~ "BW",
      grepl("^Cf", Cls) ~ "Cf",
      grepl("^Cs", Cls) ~ "Cs",
      grepl("^Cw", Cls) ~ "Cw",
      grepl("^Df", Cls) ~ "Df",
      grepl("^Ds", Cls) ~ "Ds",
      grepl("^Dw", Cls) ~ "Dw",
      Cls == "EF" ~ "EF",
      Cls == "ET" ~ "ET",
      TRUE ~ NA_character_
    )
  ) %>%
  filter(!is.na(display_group))

climate_levels <- c("Af", "Am", "As", "Aw", "BS", "BW", "Cf", "Cs", "Cw", "Df", "Ds", "Dw", "EF", "ET")
kg$display_group <- factor(kg$display_group, levels = climate_levels)

# Stable categorical palette chosen to closely match the manuscript context map.
climate_colors <- c(
  "Af" = "#0B2E68",
  "Am" = "#164B6E",
  "As" = "#1D5C63",
  "Aw" = "#39705F",
  "BS" = "#4C754A",
  "BW" = "#667B3D",
  "Cf" = "#819044",
  "Cs" = "#C49A2C",
  "Cw" = "#D2A43A",
  "Df" = "#F2A34B",
  "Ds" = "#D77B5E",
  "Dw" = "#E5A09E",
  "EF" = "#F3B3C9",
  "ET" = "#E4B2E8"
)

p1 <- ggplot(kg, aes(x = Lon, y = Lat, fill = display_group)) +
  geom_raster() +
  scale_fill_manual(values = climate_colors, drop = FALSE, name = "Climate Zone") +
  coord_fixed(xlim = c(-180, 180), ylim = c(-90, 90), expand = FALSE) +
  theme_void(base_size = 11) +
  theme(
    panel.background = element_rect(fill = "gray70", color = "black", linewidth = 0.6),
    plot.background = element_rect(fill = "white", color = NA),
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 9),
    legend.key.width = grid::unit(0.30, "cm"),
    legend.key.height = grid::unit(0.35, "cm"),
    plot.margin = grid::unit(c(4, 4, 4, 4), "pt")
  ) +
  guides(fill = guide_legend(nrow = 1, byrow = TRUE))

ggsave(
  file.path(FIG, "koppen.png"),
  p1,
  width = 10.6,
  height = 5.2,
  dpi = 600,
  bg = "white"
)
message("[ok] ", file.path(FIG, "koppen.png"))

# -----------------------------------------------------------------------------
# Final Figure 6: student-teacher wet-bulb agreement for the six largest
# Köppen-Geiger climate groups (the six panels shown in the final manuscript).
# -----------------------------------------------------------------------------
cmp_file <- file.path(ROOT, "outputs", "data", "comparison", "08_student_teacher_station_compare.csv")
if (!file.exists(cmp_file)) stop("Missing student-teacher comparison table: ", cmp_file)

cmp <- readr::read_csv(cmp_file, show_col_types = FALSE) %>%
  filter(
    is.finite(student_thr_WBT_C),
    is.finite(teacher_thr_WBT_C),
    !is.na(climate_group),
    climate_group != ""
  )

major <- cmp %>%
  count(climate_group, sort = TRUE) %>%
  slice_head(n = 6) %>%
  pull(climate_group)

stats <- cmp %>%
  filter(climate_group %in% major) %>%
  group_by(climate_group) %>%
  summarize(
    n = n(),
    r = ifelse(n() >= 2, cor(teacher_thr_WBT_C, student_thr_WBT_C, use = "complete.obs"), NA_real_),
    .groups = "drop"
  ) %>%
  mutate(
    facet_label = ifelse(
      is.finite(r),
      sprintf("%s (n=%d, r=%.2f)", climate_group, n, r),
      sprintf("%s (n=%d)", climate_group, n)
    )
  )

plot_data <- cmp %>%
  filter(climate_group %in% major) %>%
  left_join(stats %>% select(climate_group, facet_label), by = "climate_group")

label_order <- stats %>%
  arrange(match(climate_group, major)) %>%
  pull(facet_label)
plot_data$facet_label <- factor(plot_data$facet_label, levels = label_order)

p6 <- ggplot(plot_data, aes(x = teacher_thr_WBT_C, y = student_thr_WBT_C)) +
  geom_point(color = "#B2182B", alpha = 0.24, size = 0.65) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#377EB8", linewidth = 0.55) +
  geom_smooth(method = "lm", se = FALSE, color = "#8C2D34", linewidth = 0.70) +
  facet_wrap(~ facet_label, scales = "free", ncol = 3) +
  labs(
    x = expression("DCut-Teacher wet-bulb cutpoint (" * degree * "C)"),
    y = expression("VC-Student wet-bulb cutpoint (" * degree * "C)")
  ) +
  theme_minimal(base_size = 10) +
  theme(
    panel.border = element_rect(fill = NA, color = "gray55", linewidth = 0.5),
    panel.grid.minor = element_blank(),
    strip.text = element_text(size = 8.5),
    plot.background = element_rect(fill = "white", color = NA)
  )

ggsave(
  file.path(FIG, "Figure_05_student_teacher_wbt_by_climate.png"),
  p6,
  width = 9.0,
  height = 6.2,
  dpi = 600,
  bg = "white"
)
message("[ok] ", file.path(FIG, "Figure_05_student_teacher_wbt_by_climate.png"))

message("PASS: FINAL CONTEXT/CLIMATE FIGURES GENERATED")

'@

$MapsRSource = @'
# 13_fix_polar_map_artifacts_v7.R
# -------------------------------------------------------------------------
# Reviewer-specific map repair.
#
# Goal:
#   Remove spurious polar / dateline connector lines while preserving:
#     * every scientific station point,
#     * the fitted values and color scales,
#     * the coastline,
#     * the Köppen-Geiger climate-zone boundaries,
#     * the manuscript figure filenames.
#
# Important design choice:
#   Background geography is NEVER plotted as sf polygon/line geometry.
#   Instead, coastline and climate boundaries are converted to short,
#   explicit x/y line segments.  This prevents GEOMETRYCOLLECTION,
#   antimeridian, and polar connector artifacts by construction.
#
# No model fitting is performed.
# -------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(viridis)
  library(sf)
  library(rnaturalearth)
  library(rnaturalearthdata)
  library(scales)
  library(grid)
})

# =========================
# 1. Locate project
# =========================
args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)

if (length(file_arg) > 0) {
  SCRIPT <- normalizePath(
    sub("^--file=", "", file_arg[1]),
    winslash = "/",
    mustWork = TRUE
  )
  SCRIPT_DIR <- dirname(SCRIPT)
} else {
  SCRIPT_DIR <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}

ROOT <- if (tolower(basename(SCRIPT_DIR)) == "code") {
  normalizePath(dirname(SCRIPT_DIR), winslash = "/", mustWork = TRUE)
} else {
  normalizePath(file.path(SCRIPT_DIR, ".."), winslash = "/", mustWork = TRUE)
}

FIG_MAIN  <- normalizePath(Sys.getenv("AOAS_FIG_WORK"), winslash = "/", mustWork = FALSE)
FIG_AUDIT <- file.path(ROOT, "figures", "_audit_polar_v7")
TAB       <- file.path(ROOT, "outputs", "data", "tables")
STA       <- file.path(ROOT, "outputs", "data", "station_summaries")
COMP      <- file.path(ROOT, "outputs", "data", "comparison")
DATA      <- file.path(ROOT, "data")

dir.create(FIG_MAIN, recursive = TRUE, showWarnings = FALSE)
dir.create(FIG_AUDIT, recursive = TRUE, showWarnings = FALSE)

split_file     <- file.path(TAB, "02_spatial_blocked_split_assignments.csv")
distill_file   <- file.path(STA, "07_distill_station_means.csv")
nodistill_file <- file.path(STA, "07_nodistill_station_means.csv")
compare_file   <- file.path(COMP, "08_student_teacher_station_compare.csv")
kg_file        <- file.path(DATA, "Koeppen-Geiger-ASCII.txt")

required <- c(split_file, distill_file, nodistill_file, compare_file)
missing_required <- required[!file.exists(required)]

if (length(missing_required) > 0) {
  stop(
    "Missing required input file(s):
  ",
    paste(missing_required, collapse = "
  ")
  )
}

# =========================
# 2. Read scientific outputs
# =========================
split_raw <- readr::read_csv(split_file, show_col_types = FALSE) %>%
  mutate(
    Longitude = as.numeric(Longitude),
    Latitude  = as.numeric(Latitude)
  )

D <- readr::read_csv(distill_file, show_col_types = FALSE) %>%
  mutate(
    Longitude = as.numeric(Longitude),
    Latitude  = as.numeric(Latitude)
  )

N <- readr::read_csv(nodistill_file, show_col_types = FALSE) %>%
  mutate(
    Longitude = as.numeric(Longitude),
    Latitude  = as.numeric(Latitude)
  )

cmp <- readr::read_csv(compare_file, show_col_types = FALSE) %>%
  mutate(
    Longitude = as.numeric(Longitude),
    Latitude  = as.numeric(Latitude)
  )

all_lat <- c(split_raw$Latitude, D$Latitude, N$Latitude, cmp$Latitude)
all_lat <- all_lat[is.finite(all_lat)]

if (length(all_lat) == 0) stop("No finite station latitudes were found.")

max_station_lat <- max(all_lat)

MAP_XMIN <- -180
MAP_XMAX <-  180
MAP_YMIN <-    0
MAP_YMAX <- min(90, max(85, ceiling(max_station_lat + 1)))

message("[info] Maximum station latitude = ", round(max_station_lat, 3))
message("[info] Map latitude extent = 0 to ", MAP_YMAX)

# =========================
# 3. Safe short-segment builder
# =========================
MAX_COAST_STEP_DEG <- 8
POLAR_LAT_DEG      <- 70
MAX_POLAR_VERTICAL <- 4

build_safe_segments <- function(path_xy) {

  req <- c("path", "x", "y")
  if (!all(req %in% names(path_xy))) {
    stop("build_safe_segments() requires columns: path, x, y")
  }

  out <- path_xy %>%
    filter(is.finite(x), is.finite(y)) %>%
    group_by(path) %>%
    mutate(
      xend = lead(x),
      yend = lead(y)
    ) %>%
    ungroup() %>%
    filter(is.finite(xend), is.finite(yend)) %>%
    mutate(
      dx = xend - x,
      dy = yend - y,
      step = sqrt(dx^2 + dy^2),
      high_lat = pmax(y, yend) >= POLAR_LAT_DEG,
      bad_polar_vertical =
        high_lat &
        abs(dx) < 0.20 &
        abs(dy) > MAX_POLAR_VERTICAL,
      bad_jump =
        abs(dx) > MAX_COAST_STEP_DEG |
        abs(dy) > MAX_COAST_STEP_DEG |
        step > MAX_COAST_STEP_DEG
    ) %>%
    filter(
      !bad_polar_vertical,
      !bad_jump,
      x >= MAP_XMIN, x <= MAP_XMAX,
      xend >= MAP_XMIN, xend <= MAP_XMAX,
      y >= MAP_YMIN, y <= MAP_YMAX,
      yend >= MAP_YMIN, yend <= MAP_YMAX
    ) %>%
    select(path, x, y, xend, yend, dx, dy, step)

  out
}

# -------------------------
# Internal unit test
# -------------------------
# This deliberately includes:
#   * one normal short segment,
#   * one long vertical high-Arctic connector,
#   * one antimeridian jump.
# Only the normal segment must survive.
toy <- tibble(
  path = c(1, 1, 1, 2, 2),
  x    = c(0.0, 0.5, 0.5, 179.0, -179.0),
  y    = c(60.0, 61.0, 82.0, 65.0, 65.0)
)

toy_safe <- build_safe_segments(toy)

if (nrow(toy_safe) != 1L) {
  stop("Internal line-segment self-test failed.")
}

message("[self-test] Pathological vertical/dateline connectors removed: PASS")

# =========================
# 4. Coastline -> explicit short segments
# =========================
coast_sf <- rnaturalearth::ne_coastline(
  scale = "medium",
  returnclass = "sf"
)

# No st_crop(), st_wrap_dateline(), st_boundary(), or st_make_valid().
# st_coordinates() only reads the existing line vertices.
coast_coords <- as.data.frame(
  sf::st_coordinates(sf::st_geometry(coast_sf))
)

if (!all(c("X", "Y") %in% names(coast_coords))) {
  stop("Could not extract X/Y coordinates from Natural Earth coastline.")
}

level_cols <- grep("^L[0-9]+$", names(coast_coords), value = TRUE)

if (length(level_cols) > 0) {
  coast_coords$path <- do.call(
    interaction,
    c(
      coast_coords[level_cols],
      list(drop = TRUE, lex.order = TRUE)
    )
  )
} else {
  coast_coords$path <- factor(1)
}

coast_path_xy <- coast_coords %>%
  transmute(
    path = path,
    x = as.numeric(X),
    y = as.numeric(Y)
  )

coast_segments <- build_safe_segments(coast_path_xy)

if (nrow(coast_segments) < 500) {
  stop(
    "Too few safe coastline segments were retained (",
    nrow(coast_segments),
    "). Background rendering aborted."
  )
}

if (
  any(coast_segments$step > MAX_COAST_STEP_DEG + 1e-8) ||
  any(
    pmax(coast_segments$y, coast_segments$yend) >= POLAR_LAT_DEG &
      abs(coast_segments$dx) < 0.20 &
      abs(coast_segments$dy) > MAX_POLAR_VERTICAL + 1e-8
  )
) {
  stop("Coastline safety verification failed.")
}

message(
  "[check] Safe coastline segments = ",
  format(nrow(coast_segments), big.mark = ","),
  "; maximum retained segment length = ",
  sprintf("%.3f", max(coast_segments$step)),
  " degrees"
)

# =========================
# 5. Köppen-Geiger -> safe neighboring-cell boundary segments
# =========================
kg_segments <- NULL

infer_regular_spacing <- function(v, label) {

  u <- sort(unique(round(as.numeric(v[is.finite(v)]), 6)))

  if (length(u) < 3) {
    stop("Not enough finite ", label, " coordinates to infer grid spacing.")
  }

  d <- diff(u)
  d <- d[is.finite(d) & d > 1e-7]

  if (length(d) == 0) {
    stop("Could not infer positive ", label, " grid spacing.")
  }

  # Missing cells can create multiples of the basic spacing.
  candidates <- sort(unique(round(d[d <= 2], 6)))

  if (length(candidates) == 0) {
    stop("Could not identify plausible ", label, " grid spacing.")
  }

  spacing <- min(candidates)

  if (!is.finite(spacing) || spacing <= 0 || spacing > 2) {
    stop("Implausible ", label, " grid spacing: ", spacing)
  }

  spacing
}

if (file.exists(kg_file)) {

  kg <- read.table(
    kg_file,
    header = TRUE,
    sep = "",
    fill = TRUE,
    stringsAsFactors = FALSE
  ) %>%
    transmute(
      Lon = as.numeric(Lon),
      Lat = as.numeric(Lat),
      Cls = as.character(Cls)
    ) %>%
    filter(
      is.finite(Lon),
      is.finite(Lat),
      Lon >= MAP_XMIN,
      Lon <= MAP_XMAX,
      Lat >= MAP_YMIN,
      Lat <= MAP_YMAX,
      !is.na(Cls),
      Cls != ""
    )

  if (nrow(kg) < 100) {
    stop("Köppen-Geiger table contains too few valid Northern Hemisphere cells.")
  }

  DX <- infer_regular_spacing(kg$Lon, "longitude")
  DY <- infer_regular_spacing(kg$Lat, "latitude")

  message(
    "[check] Detected Köppen-Geiger spacing: dLon=",
    sprintf("%.6f", DX),
    ", dLat=",
    sprintf("%.6f", DY),
    " degrees"
  )

  if (abs(DX - 0.5) > 1e-4 || abs(DY - 0.5) > 1e-4) {
    stop(
      "Unexpected Köppen-Geiger grid spacing. Expected 0.5 degrees but detected ",
      "dLon=", DX, ", dLat=", DY, "."
    )
  }

  # IMPORTANT FIX RELATIVE TO v6:
  # The ASCII grid can use quarter-degree centers such as -179.75, -179.25.
  # R's round() uses ties-to-even, so integer-rounding Lon*2 or Lat*2 can
  # collapse or skip neighboring cells. Build grid indices relative to the
  # actual minimum coordinate instead.
  LON0 <- min(kg$Lon, na.rm = TRUE)
  LAT0 <- min(kg$Lat, na.rm = TRUE)

  kg <- kg %>%
    mutate(
      xi = as.integer(round((Lon - LON0) / DX)),
      yi = as.integer(round((Lat - LAT0) / DY))
    ) %>%
    distinct(xi, yi, .keep_all = TRUE)

  # Verify one index step really represents one physical grid step.
  idx_error_x <- abs(kg$Lon - (LON0 + kg$xi * DX))
  idx_error_y <- abs(kg$Lat - (LAT0 + kg$yi * DY))

  max_x_error <- max(idx_error_x, na.rm = TRUE)
  max_y_error <- max(idx_error_y, na.rm = TRUE)

  if (
    !is.finite(max_x_error) ||
    !is.finite(max_y_error) ||
    max_x_error > 1e-4 ||
    max_y_error > 1e-4
  ) {
    stop(
      "Köppen-Geiger grid indexing verification failed: ",
      "max longitude error=", max_x_error,
      ", max latitude error=", max_y_error
    )
  }

  east <- kg %>%
    select(xi, yi, Lon, Lat, Cls) %>%
    left_join(
      kg %>%
        transmute(
          xi = xi - 1L,
          yi = yi,
          Cls_east = Cls
        ),
      by = c("xi", "yi")
    ) %>%
    filter(!is.na(Cls_east), Cls != Cls_east) %>%
    transmute(
      x = Lon + DX / 2,
      xend = Lon + DX / 2,
      y = Lat - DY / 2,
      yend = Lat + DY / 2,
      source = "east-west"
    )

  north <- kg %>%
    select(xi, yi, Lon, Lat, Cls) %>%
    left_join(
      kg %>%
        transmute(
          xi = xi,
          yi = yi - 1L,
          Cls_north = Cls
        ),
      by = c("xi", "yi")
    ) %>%
    filter(!is.na(Cls_north), Cls != Cls_north) %>%
    transmute(
      x = Lon - DX / 2,
      xend = Lon + DX / 2,
      y = Lat + DY / 2,
      yend = Lat + DY / 2,
      source = "north-south"
    )

  message(
    "[check] Raw class-change boundaries: east-west=",
    nrow(east),
    ", north-south=",
    nrow(north)
  )

  kg_segments <- bind_rows(east, north) %>%
    mutate(
      x = pmax(MAP_XMIN, pmin(MAP_XMAX, x)),
      xend = pmax(MAP_XMIN, pmin(MAP_XMAX, xend)),
      y = pmax(MAP_YMIN, pmin(MAP_YMAX, y)),
      yend = pmax(MAP_YMIN, pmin(MAP_YMAX, yend)),
      step = sqrt((xend - x)^2 + (yend - y)^2)
    ) %>%
    filter(
      is.finite(x), is.finite(xend),
      is.finite(y), is.finite(yend),
      step > 0,
      step <= max(DX, DY) + 1e-8
    ) %>%
    distinct(x, y, xend, yend, .keep_all = TRUE)

  if (nrow(kg_segments) < 50) {
    stop(
      "Köppen-Geiger boundary construction still produced too few segments (",
      nrow(kg_segments),
      "). Diagnostic: valid cells=", nrow(kg),
      ", east-west=", nrow(east),
      ", north-south=", nrow(north),
      "."
    )
  }

  max_kg_step <- max(kg_segments$step, na.rm = TRUE)

  if (
    !is.finite(max_kg_step) ||
    max_kg_step > max(DX, DY) + 1e-8
  ) {
    stop("Köppen-Geiger segment-length safety check failed.")
  }

  message(
    "[check] Safe Köppen-Geiger boundary segments = ",
    format(nrow(kg_segments), big.mark = ","),
    "; maximum segment length = ",
    sprintf("%.3f", max_kg_step),
    " degrees"
  )

} else {
  message(
    "[warning] Koeppen-Geiger-ASCII.txt not found; ",
    "maps will contain safe coastline only."
  )
}

# -------------------------
# Additional indexing self-test for quarter-degree centers
# -------------------------
kg_toy <- tibble(
  Lon = c(0.25, 0.75, 0.25, 0.75),
  Lat = c(0.25, 0.25, 0.75, 0.75),
  Cls = c("A", "B", "A", "B")
)

DX_toy <- 0.5
DY_toy <- 0.5
LON0_toy <- min(kg_toy$Lon)
LAT0_toy <- min(kg_toy$Lat)

kg_toy <- kg_toy %>%
  mutate(
    xi = as.integer(round((Lon - LON0_toy) / DX_toy)),
    yi = as.integer(round((Lat - LAT0_toy) / DY_toy))
  )

toy_east <- kg_toy %>%
  left_join(
    kg_toy %>%
      transmute(
        xi = xi - 1L,
        yi = yi,
        Cls_east = Cls
      ),
    by = c("xi", "yi")
  ) %>%
  filter(!is.na(Cls_east), Cls != Cls_east)

if (nrow(toy_east) != 2L) {
  stop("Quarter-degree-center Köppen-Geiger indexing self-test failed.")
}

message("[self-test] Quarter-degree climate-grid indexing: PASS")

# =========================
# 6. Plot helpers
# =========================
map_base <- function() {

  ggplot() +
    coord_sf(
      crs = sf::st_crs(4326),
      default_crs = sf::st_crs(4326),
      xlim = c(MAP_XMIN, MAP_XMAX),
      ylim = c(MAP_YMIN, MAP_YMAX),
      expand = FALSE,
      datum = NA,
      clip = "on"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      axis.text = element_blank(),
      axis.title = element_blank(),
      axis.ticks = element_blank(),
      panel.grid = element_blank(),
      panel.border = element_rect(
        color = "black",
        fill = NA,
        linewidth = 0.7
      ),
      panel.background = element_rect(
        fill = "gray96",
        color = NA
      ),
      legend.position = "bottom",
      legend.direction = "horizontal",
      plot.background = element_rect(
        fill = "white",
        color = "white"
      ),
      plot.margin = grid::unit(rep(5, 4), "pt")
    )
}

# Add linework AFTER station points so the thin geography remains legible.
# Fixed colors mean these layers never enter the scientific color legend.
add_safe_map_lines <- function(p) {

  if (!is.null(kg_segments) && nrow(kg_segments) > 0) {
    p <- p +
      geom_segment(
        data = kg_segments,
        inherit.aes = FALSE,
        aes(x = x, y = y, xend = xend, yend = yend),
        color = "gray72",
        linewidth = 0.16,
        alpha = 0.90,
        lineend = "butt"
      )
  }

  p +
    geom_segment(
      data = coast_segments,
      inherit.aes = FALSE,
      aes(x = x, y = y, xend = xend, yend = yend),
      color = "black",
      linewidth = 0.22,
      alpha = 0.90,
      lineend = "butt"
    )
}

safe_limits <- function(x, fallback) {

  x <- x[is.finite(x)]

  if (length(x) < 2) return(fallback)

  q <- suppressWarnings(
    as.numeric(quantile(x, c(0.01, 0.99), na.rm = TRUE))
  )

  if (length(q) != 2 || any(!is.finite(q)) || q[1] >= q[2]) {
    return(fallback)
  }

  q
}

# =========================
# 7. Prepare plotting data
# =========================
split <- split_raw %>%
  filter(
    is.finite(Longitude),
    is.finite(Latitude),
    Longitude >= MAP_XMIN,
    Longitude <= MAP_XMAX,
    Latitude >= MAP_YMIN,
    Latitude <= MAP_YMAX,
    !is.na(split_set)
  ) %>%
  distinct(Longitude, Latitude, split_set) %>%
  mutate(
    group = case_when(
      split_set %in% c("train", "val") ~ "Train/validation",
      split_set == "test" ~ "Test",
      split_set == "buffer_excluded" ~ "Buffer excluded",
      TRUE ~ as.character(split_set)
    )
  )

D_air <- D %>%
  filter(
    is.finite(Longitude),
    is.finite(Latitude),
    is.finite(thr_Air_C),
    Longitude >= MAP_XMIN,
    Longitude <= MAP_XMAX,
    Latitude >= MAP_YMIN,
    Latitude <= MAP_YMAX
  )

N_air <- N %>%
  filter(
    is.finite(Longitude),
    is.finite(Latitude),
    is.finite(thr_Air_C),
    Longitude >= MAP_XMIN,
    Longitude <= MAP_XMAX,
    Latitude >= MAP_YMIN,
    Latitude <= MAP_YMAX
  )

D_idx <- D %>%
  filter(
    is.finite(Longitude),
    is.finite(Latitude),
    is.finite(Index_I),
    Longitude >= MAP_XMIN,
    Longitude <= MAP_XMAX,
    Latitude >= MAP_YMIN,
    Latitude <= MAP_YMAX
  )

cmp_plot <- cmp %>%
  filter(
    is.finite(Longitude),
    is.finite(Latitude),
    is.finite(student_thr_Air_C),
    is.finite(teacher_thr_Air_C),
    Longitude >= MAP_XMIN,
    Longitude <= MAP_XMAX,
    Latitude >= MAP_YMIN,
    Latitude <= MAP_YMAX
  )

message(
  "[check] Station rows retained: split=", nrow(split),
  ", distilled air=", nrow(D_air),
  ", no-distill air=", nrow(N_air),
  ", index=", nrow(D_idx),
  ", student-teacher=", nrow(cmp_plot)
)

# =========================
# 8. Back up current manuscript PNGs
# =========================
targets <- c(
  "split_coverage_map__distill_spatial_blocked_latlon.png",
  "map_air_threshold__distill.png",
  "map_air_threshold_student_distill.png",
  "map_air_threshold_teacher.png",
  "map_index_Iz.png",
  "map_air_threshold__nodistill.png"
)

backup_dir <- file.path(
  FIG_MAIN,
  paste0(
    "backup_before_polar_v7_",
    format(Sys.time(), "%Y%m%d_%H%M%S")
  )
)
dir.create(backup_dir, recursive = TRUE, showWarnings = FALSE)

for (f in targets) {
  old <- file.path(FIG_MAIN, f)
  if (file.exists(old)) {
    file.copy(old, file.path(backup_dir, f), overwrite = TRUE)
  }
}

save_map <- function(p, filename, w = 7.2, h = 3.0, dpi = 600) {

  out_main  <- file.path(FIG_MAIN, filename)
  out_audit <- file.path(FIG_AUDIT, filename)

  ggsave(
    filename = out_main,
    plot = p,
    width = w,
    height = h,
    dpi = dpi,
    bg = "white"
  )

  ggsave(
    filename = out_audit,
    plot = p,
    width = w,
    height = h,
    dpi = dpi,
    bg = "white"
  )

  message("[ok] ", out_main)
}

# Also create a linework-only audit image.
p_background <- add_safe_map_lines(map_base())

ggsave(
  file.path(FIG_AUDIT, "BACKGROUND_ONLY_v7.png"),
  p_background,
  width = 7.2,
  height = 3.8,
  dpi = 300,
  bg = "white"
)

# =========================
# 9. Figure 1: blocked split
# =========================
p_split <- map_base() +
  geom_point(
    data = split,
    aes(x = Longitude, y = Latitude, color = group),
    size = 0.40,
    alpha = 0.72
  )

p_split <- add_safe_map_lines(p_split) +
  scale_color_manual(
    values = c(
      "Train/validation" = "#377EB8",
      "Test" = "#E66101",
      "Buffer excluded" = "gray60"
    ),
    breaks = c(
      "Train/validation",
      "Test",
      "Buffer excluded"
    ),
    name = NULL
  )

save_map(
  p_split,
  "split_coverage_map__distill_spatial_blocked_latlon.png"
)

# =========================
# 10. Figure 2 / Figure 16-left: distilled air threshold
# =========================
air_lim <- safe_limits(
  D_air$thr_Air_C,
  fallback = c(-2.5, 0.5)
)

p_air_distill <- map_base() +
  geom_point(
    data = D_air,
    aes(x = Longitude, y = Latitude, color = thr_Air_C),
    size = 0.62,
    alpha = 0.88
  )

p_air_distill <- add_safe_map_lines(p_air_distill) +
  scale_color_viridis_c(
    option = "viridis",
    limits = air_lim,
    oob = scales::squish,
    name = expression(
      "Mean air temp threshold (" * degree * "C)"
    )
  )

save_map(
  p_air_distill,
  "map_air_threshold__distill.png"
)

# =========================
# 11. Figure 7: contextual index
# =========================
idx_lim <- safe_limits(
  D_idx$Index_I,
  fallback = c(-2.5, 2.5)
)

p_index <- map_base() +
  geom_point(
    data = D_idx,
    aes(x = Longitude, y = Latitude, color = Index_I),
    size = 0.62,
    alpha = 0.88
  )

p_index <- add_safe_map_lines(p_index) +
  scale_color_viridis_c(
    option = "viridis",
    limits = idx_lim,
    oob = scales::squish,
    name = "Context index I(z) (std units)"
  )

save_map(
  p_index,
  "map_index_Iz.png"
)

# =========================
# 12. Figure 4: student / teacher air thresholds
# =========================
st_lim <- range(
  c(
    cmp_plot$student_thr_Air_C,
    cmp_plot$teacher_thr_Air_C
  ),
  na.rm = TRUE
)

st_lim <- c(
  floor(st_lim[1] * 2) / 2,
  ceiling(st_lim[2] * 2) / 2
)

if (
  any(!is.finite(st_lim)) ||
  st_lim[1] >= st_lim[2]
) {
  st_lim <- c(-2.5, 0.5)
}

st_breaks <- pretty(st_lim, n = 6)

p_student <- map_base() +
  geom_point(
    data = cmp_plot,
    aes(x = Longitude, y = Latitude, color = student_thr_Air_C),
    size = 0.62,
    alpha = 0.86
  )

p_student <- add_safe_map_lines(p_student) +
  scale_color_viridis_c(
    option = "viridis",
    limits = st_lim,
    breaks = st_breaks,
    oob = scales::squish,
    name = expression(
      "Air threshold (" * degree * "C)"
    )
  )

p_teacher <- map_base() +
  geom_point(
    data = cmp_plot,
    aes(x = Longitude, y = Latitude, color = teacher_thr_Air_C),
    size = 0.62,
    alpha = 0.86
  )

p_teacher <- add_safe_map_lines(p_teacher) +
  scale_color_viridis_c(
    option = "viridis",
    limits = st_lim,
    breaks = st_breaks,
    oob = scales::squish,
    name = expression(
      "Air threshold (" * degree * "C)"
    )
  )

save_map(
  p_student,
  "map_air_threshold_student_distill.png"
)

save_map(
  p_teacher,
  "map_air_threshold_teacher.png"
)

# =========================
# 13. Figure 16-right: no-distillation air threshold
# =========================
# Same scale as the distilled panel, matching v5.
p_air_nodistill <- map_base() +
  geom_point(
    data = N_air,
    aes(x = Longitude, y = Latitude, color = thr_Air_C),
    size = 0.62,
    alpha = 0.88
  )

p_air_nodistill <- add_safe_map_lines(p_air_nodistill) +
  scale_color_viridis_c(
    option = "viridis",
    limits = air_lim,
    oob = scales::squish,
    name = expression(
      "Mean air temp threshold (" * degree * "C)"
    )
  )

save_map(
  p_air_nodistill,
  "map_air_threshold__nodistill.png"
)

# =========================
# 14. Final verification
# =========================
missing_targets <- targets[
  !file.exists(file.path(FIG_MAIN, targets))
]

if (length(missing_targets) > 0) {
  stop(
    "Some corrected figure files were not created:
  ",
    paste(missing_targets, collapse = "
  ")
  )
}

audit_missing <- targets[
  !file.exists(file.path(FIG_AUDIT, targets))
]

if (length(audit_missing) > 0) {
  stop(
    "Some audit copies were not created:
  ",
    paste(audit_missing, collapse = "
  ")
  )
}

message("")
message("=======================================================")
message("PASS: POLAR MAP ARTIFACT FIX v7 COMPLETED")
message("Project root: ", ROOT)
message("Updated manuscript figures: ", FIG_MAIN)
message("Audit copies: ", FIG_AUDIT)
message("Background-only audit: ", file.path(FIG_AUDIT, "BACKGROUND_ONLY_v7.png"))
message("Backups: ", backup_dir)
message("No model fitting or scientific values were changed.")
message("Every station observation is still rendered with geom_point().")
message("Coastline is rendered only as verified short x/y segments.")
message("Köppen-Geiger boundaries are rendered only as 0.5-degree x/y segments.")
message("No sf polygon/line layer is used for the background.")
message("=======================================================")

'@

$DifferenceRSource = @'
# 15_make_revised_figure4_difference.R
# -------------------------------------------------------------------------
# AOAS reviewer Minor Point 3:
# Create ONLY the revised Figure 4 right panel:
#   VC-Student minus DCut-Teacher station-mean air-threshold difference.
#
# This does NOT refit any model and does NOT alter any scientific values.
# It reads the existing student-teacher comparison table.
# -------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(sf)
  library(rnaturalearth)
  library(rnaturalearthdata)
  library(scales)
  library(grid)
})

# -------------------------
# Locate project root
# -------------------------
args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)

if (length(file_arg) > 0) {
  SCRIPT <- normalizePath(
    sub("^--file=", "", file_arg[1]),
    winslash = "/",
    mustWork = TRUE
  )
  SCRIPT_DIR <- dirname(SCRIPT)
} else {
  SCRIPT_DIR <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}

ROOT <- if (tolower(basename(SCRIPT_DIR)) == "code") {
  normalizePath(dirname(SCRIPT_DIR), winslash = "/", mustWork = TRUE)
} else {
  normalizePath(file.path(SCRIPT_DIR, ".."), winslash = "/", mustWork = TRUE)
}

FIG_DIR <- normalizePath(Sys.getenv("AOAS_FIG_WORK"), winslash = "/", mustWork = FALSE)
DATA_DIR <- file.path(ROOT, "data")

dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)

# Preferred clean-pipeline comparison table.
candidate_compare <- c(
  file.path(ROOT, "outputs", "data", "comparison", "08_student_teacher_station_compare.csv"),
  file.path(ROOT, "outputs", "comparison", "student_teacher_station_compare.csv")
)

compare_file <- candidate_compare[file.exists(candidate_compare)][1]

if (length(compare_file) == 0 || is.na(compare_file)) {
  stop(
    "Could not find the student-teacher comparison table.
Checked:
  ",
    paste(candidate_compare, collapse = "
  ")
  )
}

kg_file <- file.path(DATA_DIR, "Koeppen-Geiger-ASCII.txt")

message("[info] Comparison file: ", compare_file)

# -------------------------
# Read station comparison
# -------------------------
cmp <- readr::read_csv(compare_file, show_col_types = FALSE)

needed <- c(
  "Longitude",
  "Latitude",
  "student_thr_Air_C",
  "teacher_thr_Air_C"
)

missing_cols <- setdiff(needed, names(cmp))

if (length(missing_cols) > 0) {
  stop(
    "Comparison table is missing required column(s): ",
    paste(missing_cols, collapse = ", ")
  )
}

cmp <- cmp %>%
  mutate(
    Longitude = as.numeric(Longitude),
    Latitude = as.numeric(Latitude),
    student_thr_Air_C = as.numeric(student_thr_Air_C),
    teacher_thr_Air_C = as.numeric(teacher_thr_Air_C),
    diff_Air_C = student_thr_Air_C - teacher_thr_Air_C
  ) %>%
  filter(
    is.finite(Longitude),
    is.finite(Latitude),
    is.finite(diff_Air_C),
    Longitude >= -180,
    Longitude <= 180,
    Latitude >= 0,
    Latitude <= 85
  )

if (nrow(cmp) == 0) {
  stop("No valid station-level air-threshold differences were found.")
}

# -------------------------
# Summary printed for audit
# -------------------------
mae <- mean(abs(cmp$diff_Air_C))
rmse <- sqrt(mean(cmp$diff_Air_C^2))
mean_diff <- mean(cmp$diff_Air_C)
median_diff <- median(cmp$diff_Air_C)
max_abs <- max(abs(cmp$diff_Air_C))

message("[check] Stations = ", format(nrow(cmp), big.mark = ","))
message("[check] Mean Student-Teacher difference = ", sprintf("%.4f C", mean_diff))
message("[check] Median Student-Teacher difference = ", sprintf("%.4f C", median_diff))
message("[check] MAE = ", sprintf("%.4f C", mae))
message("[check] RMSE = ", sprintf("%.4f C", rmse))
message("[check] Maximum absolute difference = ", sprintf("%.4f C", max_abs))

# Symmetric robust display range centered exactly at zero.
# Values outside the 99th percentile are retained but squished to the end colors.
diff_lim <- as.numeric(
  quantile(abs(cmp$diff_Air_C), 0.99, na.rm = TRUE)
)

if (!is.finite(diff_lim) || diff_lim <= 0) {
  diff_lim <- max_abs
}

if (!is.finite(diff_lim) || diff_lim <= 0) {
  diff_lim <- 1
}

message(
  "[check] Symmetric color scale = ",
  sprintf("[-%.4f, +%.4f] C (99th percentile |difference|)", diff_lim, diff_lim)
)

# -------------------------
# Safe coastline segments
# -------------------------
MAP_XMIN <- -180
MAP_XMAX <- 180
MAP_YMIN <- 0
MAP_YMAX <- 85

MAX_STEP <- 8
POLAR_LAT <- 70
MAX_POLAR_VERTICAL <- 4

build_safe_segments <- function(path_xy) {

  path_xy %>%
    filter(is.finite(x), is.finite(y)) %>%
    group_by(path) %>%
    mutate(
      xend = lead(x),
      yend = lead(y)
    ) %>%
    ungroup() %>%
    filter(is.finite(xend), is.finite(yend)) %>%
    mutate(
      dx = xend - x,
      dy = yend - y,
      step = sqrt(dx^2 + dy^2),
      bad_polar_vertical =
        pmax(y, yend) >= POLAR_LAT &
        abs(dx) < 0.20 &
        abs(dy) > MAX_POLAR_VERTICAL,
      bad_jump =
        abs(dx) > MAX_STEP |
        abs(dy) > MAX_STEP |
        step > MAX_STEP
    ) %>%
    filter(
      !bad_polar_vertical,
      !bad_jump,
      x >= MAP_XMIN, x <= MAP_XMAX,
      xend >= MAP_XMIN, xend <= MAP_XMAX,
      y >= MAP_YMIN, y <= MAP_YMAX,
      yend >= MAP_YMIN, yend <= MAP_YMAX
    ) %>%
    select(path, x, y, xend, yend)
}

coast_sf <- rnaturalearth::ne_coastline(
  scale = "medium",
  returnclass = "sf"
)

coast_coords <- as.data.frame(
  sf::st_coordinates(sf::st_geometry(coast_sf))
)

level_cols <- grep("^L[0-9]+$", names(coast_coords), value = TRUE)

if (length(level_cols) > 0) {
  coast_coords$path <- do.call(
    interaction,
    c(coast_coords[level_cols], list(drop = TRUE, lex.order = TRUE))
  )
} else {
  coast_coords$path <- factor(1)
}

coast_segments <- coast_coords %>%
  transmute(
    path = path,
    x = as.numeric(X),
    y = as.numeric(Y)
  ) %>%
  build_safe_segments()

if (nrow(coast_segments) < 500) {
  stop("Safe coastline construction failed.")
}

# -------------------------
# Safe Köppen-Geiger boundaries
# -------------------------
kg_segments <- NULL

infer_spacing <- function(v) {
  u <- sort(unique(round(as.numeric(v[is.finite(v)]), 6)))
  d <- diff(u)
  d <- sort(unique(round(d[is.finite(d) & d > 1e-7 & d <= 2], 6)))
  if (length(d) == 0) return(NA_real_)
  min(d)
}

if (file.exists(kg_file)) {

  kg <- read.table(
    kg_file,
    header = TRUE,
    sep = "",
    fill = TRUE,
    stringsAsFactors = FALSE
  ) %>%
    transmute(
      Lon = as.numeric(Lon),
      Lat = as.numeric(Lat),
      Cls = as.character(Cls)
    ) %>%
    filter(
      is.finite(Lon),
      is.finite(Lat),
      Lon >= MAP_XMIN,
      Lon <= MAP_XMAX,
      Lat >= MAP_YMIN,
      Lat <= MAP_YMAX,
      !is.na(Cls),
      Cls != ""
    )

  DX <- infer_spacing(kg$Lon)
  DY <- infer_spacing(kg$Lat)

  if (
    !is.finite(DX) ||
    !is.finite(DY) ||
    abs(DX - 0.5) > 1e-4 ||
    abs(DY - 0.5) > 1e-4
  ) {
    stop(
      "Unexpected Köppen-Geiger grid spacing: dLon=",
      DX,
      ", dLat=",
      DY
    )
  }

  LON0 <- min(kg$Lon, na.rm = TRUE)
  LAT0 <- min(kg$Lat, na.rm = TRUE)

  kg <- kg %>%
    mutate(
      xi = as.integer(round((Lon - LON0) / DX)),
      yi = as.integer(round((Lat - LAT0) / DY))
    ) %>%
    distinct(xi, yi, .keep_all = TRUE)

  east <- kg %>%
    select(xi, yi, Lon, Lat, Cls) %>%
    left_join(
      kg %>%
        transmute(
          xi = xi - 1L,
          yi = yi,
          Cls_east = Cls
        ),
      by = c("xi", "yi")
    ) %>%
    filter(!is.na(Cls_east), Cls != Cls_east) %>%
    transmute(
      x = Lon + DX / 2,
      xend = Lon + DX / 2,
      y = Lat - DY / 2,
      yend = Lat + DY / 2
    )

  north <- kg %>%
    select(xi, yi, Lon, Lat, Cls) %>%
    left_join(
      kg %>%
        transmute(
          xi = xi,
          yi = yi - 1L,
          Cls_north = Cls
        ),
      by = c("xi", "yi")
    ) %>%
    filter(!is.na(Cls_north), Cls != Cls_north) %>%
    transmute(
      x = Lon - DX / 2,
      xend = Lon + DX / 2,
      y = Lat + DY / 2,
      yend = Lat + DY / 2
    )

  kg_segments <- bind_rows(east, north) %>%
    mutate(
      x = pmax(MAP_XMIN, pmin(MAP_XMAX, x)),
      xend = pmax(MAP_XMIN, pmin(MAP_XMAX, xend)),
      y = pmax(MAP_YMIN, pmin(MAP_YMAX, y)),
      yend = pmax(MAP_YMIN, pmin(MAP_YMAX, yend)),
      step = sqrt((xend - x)^2 + (yend - y)^2)
    ) %>%
    filter(
      is.finite(x), is.finite(xend),
      is.finite(y), is.finite(yend),
      step > 0,
      step <= 0.5000001
    ) %>%
    distinct(x, y, xend, yend)

  message(
    "[check] Köppen-Geiger boundary segments = ",
    format(nrow(kg_segments), big.mark = ",")
  )
}

# -------------------------
# Difference map
# -------------------------
p <- ggplot() +
  geom_point(
    data = cmp,
    aes(
      x = Longitude,
      y = Latitude,
      color = diff_Air_C
    ),
    size = 0.70,
    alpha = 0.94
  )

if (!is.null(kg_segments) && nrow(kg_segments) > 0) {
  p <- p +
    geom_segment(
      data = kg_segments,
      inherit.aes = FALSE,
      aes(x = x, y = y, xend = xend, yend = yend),
      color = "gray72",
      linewidth = 0.16,
      alpha = 0.90,
      lineend = "butt"
    )
}

p <- p +
  geom_segment(
    data = coast_segments,
    inherit.aes = FALSE,
    aes(x = x, y = y, xend = xend, yend = yend),
    color = "black",
    linewidth = 0.22,
    alpha = 0.90,
    lineend = "butt"
  ) +
  scale_color_gradient2(
    low = "#2166AC",
    mid = "#D9D9D9",
    high = "#B2182B",
    midpoint = 0,
    limits = c(-diff_lim, diff_lim),
    oob = scales::squish,
    name = expression("Student - Teacher air threshold (" * degree * "C)")
  ) +
  coord_quickmap(
    xlim = c(MAP_XMIN, MAP_XMAX),
    ylim = c(MAP_YMIN, MAP_YMAX),
    expand = FALSE,
    clip = "on"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    axis.text = element_blank(),
    axis.title = element_blank(),
    axis.ticks = element_blank(),
    panel.grid = element_blank(),
    panel.border = element_rect(
      color = "black",
      fill = NA,
      linewidth = 0.7
    ),
    panel.background = element_rect(
      fill = "gray96",
      color = NA
    ),
    legend.position = "bottom",
    legend.direction = "horizontal",
    plot.background = element_rect(
      fill = "white",
      color = "white"
    ),
    plot.margin = grid::unit(rep(5, 4), "pt")
  )

out_png <- file.path(
  FIG_DIR,
  "map_air_threshold_difference_student_minus_teacher.png"
)

ggsave(
  out_png,
  p,
  width = 7.2,
  height = 3.0,
  dpi = 600,
  bg = "white"
)

message("[ok] ", out_png)
message("")
message("PASS: FIGURE 4 DIFFERENCE MAP COMPLETED")
message("No model fitting was performed.")
message("Difference definition: VC-Student minus DCut-Teacher.")

'@

try {
    Write-Utf8NoBom -Path $TmpPython   -Text $PythonSource
    Write-Utf8NoBom -Path $TmpContextR -Text $ContextRSource
    Write-Utf8NoBom -Path $TmpMapsR    -Text $MapsRSource
    Write-Utf8NoBom -Path $TmpDiffR    -Text $DifferenceRSource

    Set-Location $Project

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "[1/4] Final non-map statistical figures"
    Write-Host "============================================================"
    & $Python $TmpPython
    if ($LASTEXITCODE -ne 0) {
        throw "Final non-map figure generation failed with exit code $LASTEXITCODE."
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "[2/4] Figure 1 climate context + Figure 6 climate facets"
    Write-Host "============================================================"
    & $Rscript $TmpContextR
    if ($LASTEXITCODE -ne 0) {
        throw "Climate/context figure generation failed with exit code $LASTEXITCODE."
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "[3/4] Artifact-safe manuscript maps"
    Write-Host "============================================================"
    & $Rscript $TmpMapsR
    if ($LASTEXITCODE -ne 0) {
        throw "Map generation failed with exit code $LASTEXITCODE."
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "[4/4] Student-minus-teacher difference map"
    Write-Host "============================================================"
    & $Rscript $TmpDiffR
    if ($LASTEXITCODE -ne 0) {
        throw "Difference-map generation failed with exit code $LASTEXITCODE."
    }

    # ------------------------------------------------------------------------
    # Curate EXACTLY the assets referenced by the final manuscript.
    # ------------------------------------------------------------------------
    $FinalAssets = @(
        "koppen.png",
        "split_coverage_map__distill_spatial_blocked_latlon.png",
        "map_air_threshold__distill.png",
        "R1_threshold_distributions.png",
        "map_air_threshold_student_distill.png",
        "map_air_threshold_difference_student_minus_teacher.png",
        "Figure_05_student_teacher_wbt_by_climate.png",
        "Figure_06_index_weights.png",
        "Figure_16_air_index_distillation_comparison.png",
        "map_index_Iz.png",
        "Figure_09_probability_vs_index.png",
        "distill__R3_thr_Air_C_vs_all_drivers_facets.png",
        "index_effects_per1SD.png",
        "distill__profile_corr_heatmap.png",
        "distill__R4_beta_vs_index_facets.pdf",
        "Figure_13_roc_precision_recall.png",
        "Figure_14_reliability.png",
        "map_air_threshold__nodistill.png"
    )

    $MissingBuild = @()
    foreach ($Name in $FinalAssets) {
        $Src = Join-Path $WorkFigDir $Name
        if (-not (Test-Path -LiteralPath $Src)) {
            $MissingBuild += $Src
        } else {
            Copy-Item -LiteralPath $Src -Destination (Join-Path $FinalFigDir $Name) -Force
        }
    }

    if ($MissingBuild.Count -gt 0) {
        Write-Host ""
        Write-Host "[ERROR] Missing final manuscript asset(s):"
        $MissingBuild | ForEach-Object { Write-Host "  $_" }
        throw "Final-asset curation failed."
    }

    # Exact-folder audit: figures_final must contain no stale/obsolete figures.
    $Actual = @(Get-ChildItem -LiteralPath $FinalFigDir -File | Select-Object -ExpandProperty Name | Sort-Object)
    $Expected = @($FinalAssets | Sort-Object)
    $Unexpected = @($Actual | Where-Object { $_ -notin $Expected })
    $StillMissing = @($Expected | Where-Object { $_ -notin $Actual })

    if ($Unexpected.Count -gt 0 -or $StillMissing.Count -gt 0) {
        if ($Unexpected.Count -gt 0) {
            Write-Host "[ERROR] Unexpected files in figures_final:"
            $Unexpected | ForEach-Object { Write-Host "  $_" }
        }
        if ($StillMissing.Count -gt 0) {
            Write-Host "[ERROR] Missing files in figures_final:"
            $StillMissing | ForEach-Object { Write-Host "  $_" }
        }
        throw "Final-folder exactness check failed."
    }

    # Basic non-empty-file check.
    $ZeroSize = @(Get-ChildItem -LiteralPath $FinalFigDir -File | Where-Object { $_.Length -le 0 })
    if ($ZeroSize.Count -gt 0) {
        throw "One or more final figure files are empty."
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "PASS: FINAL GITHUB FIGURE SET IS CLEAN"
    Write-Host "============================================================"
    Write-Host "Final folder contains exactly $($FinalAssets.Count) manuscript assets:"
    Write-Host "  $FinalFigDir"
    Write-Host ""
    Write-Host "Important final-revision checks:"
    Write-Host "  - Figure 8 contains NO Pearson-correlation annotation."
    Write-Host "  - Obsolete Figures 17/18 density-cloud outputs are NOT curated."
    Write-Host "  - Old contribution-share and dewpoint-index figures are NOT curated."
    Write-Host "  - No model fitting was performed by this script."
    Write-Host ""
    Write-Host "For GitHub, keep this PowerShell file plus the curated figures_final folder."
    Write-Host "You may rename figures_final to figures after you remove/replace the old figure folder."
}
finally {
    foreach ($Tmp in @($TmpPython, $TmpContextR, $TmpMapsR, $TmpDiffR)) {
        if (Test-Path -LiteralPath $Tmp) {
            Remove-Item -LiteralPath $Tmp -Force -ErrorAction SilentlyContinue
        }
    }
    if (Test-Path -LiteralPath $WorkFigDir) {
        Remove-Item -LiteralPath $WorkFigDir -Recurse -Force -ErrorAction SilentlyContinue
    }

    $env:AOAS_PROJECT_ROOT = $OldProjectEnv
    $env:AOAS_FIG_WORK = $OldFigEnv
}
