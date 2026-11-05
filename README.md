# AI-Guided Single-Index Varying-Coefficient Logistic Modeling for Rain–Snow Partitioning

**Research code accompanying** *AI-Guided Single-Index Varying-Coefficient Logistic Modeling: A Rain–Snow Partitioning Case Study* (Annals of Applied Statistics, manuscript AOAS2512-017).

This repository implements the interpretable **VC–Student** precipitation-phase classifier and the analyses supporting the accompanying manuscript. It includes chronological and geographically blocked evaluations, a no-distillation comparison, station-level teacher–student comparisons, and ten stability refits.

> **Reproducibility status:** The source code is public. Reproducing the full numerical study requires the processed observational dataset and pretrained teacher weights, which are **not included** in this repository. The commands below describe the intended workflow; a complete, clean-environment rerun from the public repository has not been independently verified.

## Method

The VC–Student uses a learned single index of atmospheric and site-related predictors to govern varying logistic coefficients and ordered temperature thresholds for dewpoint, wet-bulb, and air temperature. A higher-capacity **DCut–Teacher** provides cutpoint targets for the distillation objective. The no-distillation fit is provided separately to evaluate the predictive/interpretability trade-off.

## Repository contents

| Path | Purpose |
| --- | --- |
| `code/00_preflight.py` | Verify Python imports and expected input files/columns |
| `code/01_train_chronological.py` | Train and evaluate the distilled VC–Student on the chronological split |
| `code/02_train_spatial_blocked.py` | Train and evaluate the geographically blocked experiment |
| `code/03_run_stability.py` | Run ten station-subsampling stability refits |
| `code/04_align_stability_signs.py` | Align single-index directions across stability refits |
| `code/05_train_chronological_nodistill.py` | Chronological ablation without distillation |
| `code/05_make_main_figures.py` | Generate analysis figures from saved outputs |
| `code/06_export_teacher_outputs.py` | Export teacher predictions and station summaries |
| `code/07_build_station_summaries.py` | Create student station-level summaries |
| `code/08_build_student_teacher_comparisons.py` | Compare student and teacher cutpoints |
| `code/09_compare_simple_rules.py` | Evaluate fixed-temperature baseline rules |
| `code/10_compute_contribution_analysis.py` | Evaluate component contributions and prediction reconstruction |
| `code/11_build_manuscript_tables.py` | Assemble manuscript summary tables |
| `code/MAKE_FINAL_PAPER_FIGURES.ps1` | Generate final manuscript figures from existing analysis outputs; requires Python and R |
| `code/pipeline_utils.py` | Shared data-processing and numerical utilities |
| `requirements.txt`, `R_REQUIREMENTS.txt` | Declared Python and R dependencies |

Generated predictions, intermediate coefficient/threshold tables, and model outputs are excluded from Git history. Selected research figures are published in [figure_outputs/](figure_outputs/). The historical `OLDPAPER` orchestrator and archived experiments are not part of the published source tree.

## Published figure outputs

Selected manuscript-related and diagnostic figures are available in [figure_outputs/](figure_outputs/).

- `Figure_05_student_teacher_wbt_by_climate.png` — student–teacher wet-bulb threshold comparisons.
- `Figure_06_index_weights.png` — single-index weights.
- `Figure_09_probability_vs_index.png` — predicted probability versus index.
- `Figure_13_roc_precision_recall.png` — ROC and precision–recall curves.
- `Figure_14_reliability.png` — reliability plot.
- `Figure_16_air_index_distillation_comparison.png` — distillation comparison.
- Additional PNG and PDF files — spatial maps and diagnostic visualizations.

Some diagnostic files do not correspond directly to numbered figures in the manuscript.

## Required inputs

Place these files in `data/` **before** running preflight or any model training:

| Local file | Purpose | Supplied in this repository? |
| --- | --- | --- |
| `data/point_data_with_atmospheric_predictors.csv` | Observation-level analysis dataset with atmospheric predictors | **No** |
| `data/best.weights.h5` | Pretrained DCut–Teacher model weights used for distillation | **No** |
| `data/Koeppen-Geiger-ASCII.txt` | Optional Köppen–Geiger background for climate maps | **No** |

The observational analysis uses NCEP ADP surface reports and ERA5-derived predictors. Access to the public source archives alone does not automatically recreate the exact processed analysis CSV: the necessary extraction, joining, filtering, and feature-construction choices must also be reproduced. The present repository does not yet provide a verified public download or complete data-construction workflow for that input.

Required input columns are checked in `code/00_preflight.py` and include `Rain_Phase`, `Cls`, `Air_Temp`, `Wet_Bulb_Temp`, `Dewpoint`, `RH`, `gridded_data_pres`, `Longitude`, `Latitude`, `Elevation`, `lapse_rate`, `Z0`, `temp_gradient`, and `ApT`, plus a recognized date/timestamp field. See the source code for dataset-specific preprocessing and split choices.

A related teacher-model code repository is available at [DCut–Teacher / DeepCut rain–snow](https://github.com/Mahamoussa321/deepcut-rainsnow). That source repository is **not** a substitute for the specific trained weights expected here.

## Software setup

The scripts target Python with TensorFlow and the packages listed in `requirements.txt`. The Python dependencies currently have minimum-version constraints rather than a verified environment lockfile. For figure maps, an R installation and the packages listed in `R_REQUIREMENTS.txt` are also required.

From the repository root in **Windows PowerShell**:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe code\00_preflight.py
```

Preflight requires both the analysis CSV and teacher-weight file. It checks their presence and the dataset columns, but it does not establish that a complete numerical rerun will succeed. A full run writes many large generated files to `outputs/`; the preflight script recommends at least 35 GB of free disk space.

## Analysis workflow

**Important:** Review file paths, available disk space, and existing results before running training. Scripts may overwrite files under `outputs/`. In particular, the training scripts currently change to a no-distillation run if teacher weights are missing; ensure that the weights are present for any intended distilled experiment.

The following is the **intended stage order**, not a certified one-command rerun:

```powershell
python code/00_preflight.py
python code/01_train_chronological.py
python code/05_train_chronological_nodistill.py
python code/02_train_spatial_blocked.py
python code/03_run_stability.py
python code/04_align_stability_signs.py
python code/06_export_teacher_outputs.py
python code/07_build_station_summaries.py
python code/08_build_student_teacher_comparisons.py
python code/09_compare_simple_rules.py
python code/10_compute_contribution_analysis.py
python code/11_build_manuscript_tables.py
```

The figure script below uses **precomputed** results; it does not retrain the student or teacher:

```powershell
powershell -ExecutionPolicy Bypass -File ".\code\MAKE_FINAL_PAPER_FIGURES.ps1"
```

It writes to `figures_final/`. **Back up that folder first:** the script removes an existing `figures_final/` before regenerating it. It also uses R for maps and has a machine-specific preferred Rscript path with a `PATH` fallback; on another machine, verify that `Rscript` is on `PATH`.

## Reported results

The following numbers are **from saved project summaries**, not newly reproduced in a clean environment.

| Experiment | Test AUC | Test accuracy |
| --- | ---: | ---: |
| Chronological, distilled VC–Student | 0.971000 | 92.6905% |
| Spatial-blocked, distilled VC–Student | 0.975011 | 92.5562% |
| Chronological, no distillation | 0.974529 | 92.8454% |

The saved full-fit student–teacher **air-temperature** station-mean cutpoint correlation is 0.70118, with a mean absolute error of 0.14042 °C. These figures differ in target and protocol from the stability-refit **wet-bulb** correlations. The previously archived verification report indicated 52/52 checks passed; that record is not an independent reproduction certificate.

The no-distillation model has a slightly higher AUC and accuracy in the archived chronological comparison. Distillation is therefore discussed as an interpretability and teacher-agreement trade-off, not an across-the-board improvement in predictive performance.

## Data and code availability

This public release provides source code and workflow documentation. It does **not** distribute the full processed observation table, pretrained teacher weights, multi-gigabyte event-level outputs, or a verified end-to-end data acquisition pipeline. Data reuse and model-weight redistribution depend on their respective source permissions. When a shareable, documented input-reconstruction method becomes available, it should be linked here.

## Citation

*AI-Guided Single-Index Varying-Coefficient Logistic Modeling: A Rain–Snow Partitioning Case Study*. Annals of Applied Statistics, manuscript AOAS2512-017. Add author list, year, publication details, and DOI when formally available.

## License

**No open-source license has been granted for this repository at present.** Public visibility does not imply permission to reuse or redistribute the code. A license may be added after all code owners and collaborators have approved the terms.
