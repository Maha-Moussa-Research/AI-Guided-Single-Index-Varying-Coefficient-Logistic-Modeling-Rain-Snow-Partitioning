from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def project_root(file: str | Path) -> Path:
    return Path(file).resolve().parents[1]


def ensure_dirs(root: Path) -> dict[str, Path]:
    output = root / "outputs"
    paths = {
        "output": output,
        "figures": output / "figures",
        "logs": output / "logs",
        "data": output / "data",
        "models": output / "data" / "models",
        "predictions": output / "data" / "predictions",
        "tables": output / "data" / "tables",
        "station": output / "data" / "station_summaries",
        "teacher": output / "data" / "teacher",
        "comparison": output / "data" / "comparison",
        "stability": output / "data" / "stability",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=True), encoding="utf-8")


def read_csv_tail(path: Path, n_rows: int, usecols: Sequence[str], chunksize: int = 400_000) -> pd.DataFrame:
    """Read the final n_rows from a large CSV without loading the full file."""
    if n_rows <= 0:
        return pd.DataFrame(columns=list(usecols))
    pieces: deque[pd.DataFrame] = deque()
    total = 0
    for chunk in pd.read_csv(path, usecols=list(usecols), chunksize=chunksize, low_memory=False):
        pieces.append(chunk)
        total += len(chunk)
        while len(pieces) > 1 and total - len(pieces[0]) >= n_rows:
            total -= len(pieces.popleft())
    if not pieces:
        raise RuntimeError(f"No rows were read from {path}")
    out = pd.concat(list(pieces), ignore_index=True)
    if len(out) < n_rows:
        raise RuntimeError(f"{path} has {len(out):,} rows, fewer than requested tail length {n_rows:,}")
    return out.tail(n_rows).reset_index(drop=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    ex = np.exp(x[~positive])
    out[~positive] = ex / (1.0 + ex)
    return out


def softplus(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def pearson(x: Iterable[float], y: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=float)
    b = np.asarray(list(y), dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return math.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def spearman(x: Iterable[float], y: Iterable[float]) -> float:
    a = pd.Series(list(x), dtype=float)
    b = pd.Series(list(y), dtype=float)
    m = a.notna() & b.notna()
    if m.sum() < 2:
        return math.nan
    return pearson(a[m].rank(method="average"), b[m].rank(method="average"))


def first_existing(candidates: Sequence[Path], description: str) -> Path:
    for p in candidates:
        if p.exists():
            return p
    details = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(f"Could not find {description}. Checked:\n{details}")
