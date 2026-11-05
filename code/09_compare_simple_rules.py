from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from pipeline_utils import ensure_dirs, project_root

ROOT = project_root(__file__)
PATHS = ensure_dirs(ROOT)
INPUT = PATHS["teacher"] / "06_simple_rule_test_input.csv"
OUT = PATHS["tables"] / "09_simple_rule_baselines.csv"

if not INPUT.exists():
    raise FileNotFoundError(f"Missing teacher-exported test input: {INPUT}")

df = pd.read_csv(INPUT)
y = pd.to_numeric(df["y_true"], errors="coerce")
tair = pd.to_numeric(df["T_air"], errors="coerce")
twbt = pd.to_numeric(df["T_wbt"], errors="coerce")
valid = y.notna() & tair.notna() & twbt.notna()
y = y[valid].astype(int).to_numpy()
tair = tair[valid].to_numpy(float)
twbt = twbt[valid].to_numpy(float)


def result(name: str, score: np.ndarray, threshold: float) -> dict:
    pred = (score > threshold).astype(int)
    return {
        "model": name,
        "threshold_C": threshold,
        "n": int(len(y)),
        "auc": float(roc_auc_score(y, score)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision_rain": float(precision_score(y, pred, zero_division=0)),
        "recall_rain": float(recall_score(y, pred, zero_division=0)),
        "f1_rain": float(f1_score(y, pred, zero_division=0)),
        "snow_recall": float(((pred == 0) & (y == 0)).sum() / max(1, (y == 0).sum())),
    }

rows = [
    result("Fixed air-temperature rule (0 C)", tair, 0.0),
    result("Fixed wet-bulb rule (1.0 C)", twbt, 1.0),
    result("Fixed wet-bulb rule (1.1 C; sensitivity)", twbt, 1.1),
]
pd.DataFrame(rows).to_csv(OUT, index=False)
print(pd.DataFrame(rows).to_string(index=False))
print("[ok] Saved:", OUT)
