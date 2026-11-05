from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pandas as pd

from pipeline_utils import ensure_dirs, project_root

ROOT = project_root(__file__)
PATHS = ensure_dirs(ROOT)
DATA = ROOT / "data" / "point_data_with_atmospheric_predictors.csv"
TEACHER = ROOT / "data" / "best.weights.h5"
KG = ROOT / "data" / "Koeppen-Geiger-ASCII.txt"

required_packages = ["numpy", "pandas", "sklearn", "tensorflow", "matplotlib", "h5py"]
missing_packages = [p for p in required_packages if importlib.util.find_spec(p) is None]
if missing_packages:
    raise RuntimeError("Missing Python packages: " + ", ".join(missing_packages))

for required in [DATA, TEACHER]:
    if not required.exists():
        raise FileNotFoundError(f"Required input not found: {required}")

header = pd.read_csv(DATA, nrows=5, low_memory=False)
required_columns = {
    "Rain_Phase", "Cls", "Air_Temp", "Wet_Bulb_Temp", "Dewpoint",
    "RH", "gridded_data_pres", "Longitude", "Latitude", "Elevation",
    "lapse_rate", "Z0", "temp_gradient", "ApT",
}
missing_columns = sorted(required_columns.difference(header.columns))
if missing_columns:
    raise KeyError("Input CSV is missing required columns: " + ", ".join(missing_columns))

if not any(c in header.columns for c in ["Date", "Timestamp", "date", "timestamp", "datetime"]):
    raise KeyError("Input CSV has no recognized date column.")

free = shutil.disk_usage(ROOT).free / (1024 ** 3)
print("INDEX-ONLY OLD-PAPER PIPELINE PREFLIGHT")
print("Project root:", ROOT)
print("Data:", DATA)
print("Teacher weights:", TEACHER)
print("Koppen-Geiger grid:", KG, "(found)" if KG.exists() else "(optional map background missing)")
print(f"Free disk space: {free:.1f} GB")
if free < 35:
    print("[warning] A full rerun produces several multi-GB event-level CSV files. At least 35 GB free is recommended.")
print("[ok] Python packages, inputs, and required columns passed preflight.")
