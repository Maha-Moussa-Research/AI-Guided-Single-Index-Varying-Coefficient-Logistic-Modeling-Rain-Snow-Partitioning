# AI-Guided Single-Index Varying-Coefficient Logistic Modeling: Rain–Snow Partitioning

Research code accompanying the manuscript submitted to *The Annals of Applied Statistics* (AOAS2512-017).

## Overview

This project fits an interpretable single-index varying-coefficient (VC) logistic student model to precipitation-phase observations. A higher-capacity DCut teacher supplies temperature-cutpoint targets for a structured distillation penalty. The student models rain probability using context-dependent coefficients and physically ordered dewpoint, wet-bulb and air-temperature cutpoints. The analysis includes chronological and geographically blocked evaluation, an ablation without distillation, student–teacher comparisons, and ten stability refits.

## Repository layout

- `code/00_preflight.py`: check required inputs and Python packages
- `code/01_train_chronological.py`: chronological VC–Student training
- `code/02_train_spatial_blocked.py`: geographically blocked evaluation
- `code/03_run_stability.py`, `04_align_stability_signs.py`: stability analysis and sign alignment
- `code/05_train_chronological_nodistill.py`: no-distillation comparison
- `code/05_make_main_figures.py`: main-figure generation
- `code/06_export_teacher_outputs.py` through `11_build_manuscript_tables.py`: teacher exports, station summaries, comparisons, baselines, contribution analysis and tables
- `code/pipeline_utils.py`: shared utilities
- `RUN_ALL_INDEX_ONLY_OLDPAPER.ps1`: legacy pipeline orchestrator (see limitations)
- `requirements.txt`, `R_REQUIREMENTS.txt`: declared dependencies

## Input data

The preflight script expects the following files in `data/`:

- `point_data_with_atmospheric_predictors.csv` (multi-GB input; **not included**)
- `best.weights.h5` (pretrained DCut–Teacher weights; include only if distribution is authorized)
- `Koeppen-Geiger-ASCII.txt` (climate-map background, where available)

Required input columns include `Rain_Phase`, `Cls`, `Air_Temp`, `Wet_Bulb_Temp`, `Dewpoint`, `RH`, `gridded_data_pres`, `Longitude`, `Latitude`, `Elevation`, `lapse_rate`, `Z0`, `temp_gradient`, and `ApT`. A recognized date/timestamp column is also required. The underlying data draw on NCEP ADP station reports and ERA5-derived atmospheric predictors. Consult the manuscript and original data providers for access conditions; this repository does not redistribute the full event-level dataset.

## Environment and preflight

On Windows PowerShell, from the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe code\00_preflight.py
```

The declared dependencies use minimum version bounds, not a tested lockfile. R is needed for the legacy map-generation stages; see `R_REQUIREMENTS.txt`. For a full run, ensure adequate free disk space (the preflight recommends at least 35 GB).

## Execution and reproducibility status

The existing PowerShell runner supports `-Mode models`, `-Mode postprocess`, and `-Mode figures`, in addition to `all`. It runs preflight first and writes logs to `outputs/logs/`. **Do not assume that `-Mode all` works from the active repository yet:** the runner references `code/12_make_oldpaper_figures.py`, `code/13_make_oldpaper_maps.R`, `code/14_make_climate_facets.R`, and `code/15_verify_pipeline.py`, which were absent from the active code inventory reviewed on 2026-11-04. Some versions may exist in archived project directories. Restore and validate the exact required scripts before advertising end-to-end reproduction. The `code/MAKE_FINAL_PAPER_FIGURES.ps1` runner also requires a separate dependency/path audit.

To run only the active model stages after supplying inputs and installing dependencies:

```powershell
.\RUN_ALL_INDEX_ONLY_OLDPAPER.ps1 -Mode models
```

This command is **documented, not independently executed or verified** in the preparation of this README. Training can be computationally expensive and can overwrite saved outputs. Preserve existing results before rerunning.

## Archived results (not independently rerun)

The inspected `outputs/data/tables/11_key_manuscript_values.json` reports:

| Analysis | Test AUC | Test accuracy |
| --- | ---: | ---: |
| Chronological VC–Student with distillation | 0.971000 | 92.6905% |
| Spatial-blocked VC–Student with distillation | 0.975011 | 92.5562% |
| Chronological VC–Student without distillation | 0.974529 | 92.8454% |

The saved chronological full-fit station-mean air-cutpoint student–teacher comparison has Pearson correlation 0.70118 and MAE 0.14042 °C. These full-fit quantities are distinct from the stability-refit wet-bulb correlations. The archived verification JSON reports 52/52 checks passed; that summary is not an independent reproduction certificate.

Distillation is evaluated as a trade-off between predictive accuracy, teacher-cutpoint agreement, and interpretability; the no-distillation fit has higher AUC in the archived chronological comparison.

## Figures and outputs

Intermediate event-level predictions, thresholds, and coefficient tables can each exceed GitHub's file-size limits and are excluded by `.gitignore`. The repository should contain scripts and suitably small, reviewed summary tables/figures, not large generated output directories. Any included figure must be traceable to an archived result or executable script.

## Citation

*AI-Guided Single-Index Varying-Coefficient Logistic Modeling: A Rain–Snow Partitioning Case Study*. Manuscript AOAS2512-017, submitted to *The Annals of Applied Statistics*. Full bibliographic details and DOI will be added when available.

## Code and data availability

Source code and execution guidance are provided here. The full observational input and large derived artifacts are not included. Data-access and redistribution rights must be checked against the original providers. This README does not assert that the entire pipeline has been independently reproduced on a clean environment.

## License

No open-source license is asserted until all contributors approve one. In the absence of a license, default copyright restrictions apply.
