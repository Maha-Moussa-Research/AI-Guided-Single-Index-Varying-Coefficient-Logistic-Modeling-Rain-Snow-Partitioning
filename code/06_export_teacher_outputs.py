from __future__ import annotations

import json
import os
import random
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "2025")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras import layers, models

from pipeline_utils import ensure_dirs, json_dump, project_root

SEED = 2025
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

ROOT = project_root(__file__)
PATHS = ensure_dirs(ROOT)
DATA_PATH = ROOT / "data" / "point_data_with_atmospheric_predictors.csv"
TEACHER_WEIGHTS = ROOT / "data" / "best.weights.h5"
OUT = PATHS["teacher"]
SAVE_EVENT_LEVEL = os.environ.get("VC_SAVE_TEACHER_EVENT_LEVEL", "1").strip().lower() not in {"0", "false", "no"}

Y = "Rain_Phase"
REGION = "Cls"
X_VARS = ["Air_Temp", "Wet_Bulb_Temp", "Dewpoint"]
Z_ALL = ["RH", "gridded_data_pres", "Longitude", "Latitude", "Elevation", "lapse_rate", "Z0", "temp_gradient", "ApT"]
DATES = ["Date", "Timestamp", "date", "timestamp", "datetime"]
TEST_YEARS = 2
VAL_FRACTION = 0.20


def build_teacher(x_dim: int, z_dim: int, r_dim: int, gate_type: str = "relu", gate_k: float = 5.0, dropout_rate: float = 0.30):
    x_input = layers.Input(shape=(x_dim,), name="x_input")
    z_input = layers.Input(shape=(z_dim,), name="z_input")
    r_input = layers.Input(shape=(r_dim,), name="region_input")

    r_embed = layers.Dense(64, activation="relu", name="r_embed_1")(r_input)
    r_embed = layers.BatchNormalization(name="r_embed_bn")(r_embed)
    r_embed = layers.Dense(32, activation="relu", name="r_embed_2")(r_embed)

    zr = layers.Concatenate(name="zr_concat")([z_input, r_embed])
    h = layers.Dense(128, activation="relu", name="cut_h1")(zr)
    h = layers.BatchNormalization(name="cut_bn1")(h)
    h = layers.Dropout(dropout_rate, seed=SEED, name="cut_do1")(h)
    h = layers.Dense(64, activation="relu", name="cut_h2")(h)
    h = layers.BatchNormalization(name="cut_bn2")(h)
    h = layers.Dropout(dropout_rate, seed=SEED, name="cut_do2")(h)

    dew_base = layers.Dense(1, activation="linear", name="dew_base")(h)
    wb_off = layers.Dense(1, activation="linear", name="wb_off")(h)
    air_off = layers.Dense(1, activation="linear", name="air_off")(h)
    wb_thr = layers.Lambda(lambda t: t[0] + tf.nn.softplus(t[1]), name="wb_thr")([dew_base, wb_off])
    air_thr = layers.Lambda(lambda t: t[0] + tf.nn.softplus(t[1]), name="air_thr")([wb_thr, air_off])
    cutting_points = layers.Concatenate(name="cutting_points")([air_thr, wb_thr, dew_base])

    def gate_pair(x, c):
        if gate_type == "relu":
            return tf.nn.relu(x - c)
        if gate_type == "softplus":
            return tf.math.softplus(gate_k * (x - c)) / gate_k
        return tf.sigmoid(gate_k * (x - c)) * (x - c)

    thresholded_x = layers.Lambda(lambda t: gate_pair(t[0], t[1]), name="thresholded_x")([x_input, cutting_points])
    all_feat = layers.Concatenate(name="clf_concat")([thresholded_x, z_input, r_embed])
    h = layers.Dense(128, activation="relu", name="clf_h1")(all_feat)
    h = layers.BatchNormalization(name="clf_bn1")(h)
    h = layers.Dropout(dropout_rate, seed=SEED, name="clf_do1")(h)
    h = layers.Dense(64, activation="relu", name="clf_h2")(h)
    h = layers.BatchNormalization(name="clf_bn2")(h)
    h = layers.Dropout(dropout_rate, seed=SEED, name="clf_do2")(h)
    h = layers.Dense(32, activation="relu", name="clf_h3")(h)
    y_out = layers.Dense(1, activation="sigmoid", name="output")(h)
    return models.Model([x_input, z_input, r_input], [y_out, cutting_points], name="DeepCutRegionEmbed")


if not DATA_PATH.exists() or not TEACHER_WEIGHTS.exists():
    raise FileNotFoundError("Data CSV or teacher weights are missing from data/.")

raw = pd.read_csv(DATA_PATH, low_memory=False)
raw[REGION] = raw[REGION].astype(str)
z_vars = [c for c in Z_ALL if c in raw.columns]
raw[X_VARS] = raw[X_VARS].apply(pd.to_numeric, errors="coerce")
raw[z_vars] = raw[z_vars].apply(pd.to_numeric, errors="coerce")
raw = raw.dropna(subset=[Y, REGION]).reset_index(drop=True)

date_col = next((c for c in DATES if c in raw.columns), None)
if date_col is None:
    raise RuntimeError("No recognized date column was found.")
raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
raw = raw.dropna(subset=[date_col, Y, REGION]).sort_values(date_col).reset_index(drop=True)
cutoff = raw[date_col].max() - pd.DateOffset(years=TEST_YEARS)
trainval = raw[raw[date_col] <= cutoff].copy()
test = raw[raw[date_col] > cutoff].copy()
n_val = max(1, int(len(trainval) * VAL_FRACTION))
val = trainval.tail(n_val).copy()
train = trainval.iloc[:-n_val].copy()

x_median = train[X_VARS].median(numeric_only=True)
z_median = train[z_vars].median(numeric_only=True)
z_vars = [c for c in z_vars if pd.notna(z_median.get(c, np.nan)) and train[c].std(skipna=True) != 0]
for frame in [train, val, test]:
    frame[X_VARS] = frame[X_VARS].fillna(x_median)
    frame[z_vars] = frame[z_vars].fillna(z_median[z_vars])

x_scaler = StandardScaler().fit(train[X_VARS].to_numpy())
z_scaler = StandardScaler().fit(train[z_vars].to_numpy())
try:
    r_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
except TypeError:
    r_encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
r_encoder.fit(train[[REGION]])


def arrays(frame: pd.DataFrame):
    x = x_scaler.transform(frame[X_VARS].to_numpy()).astype("float32")
    z = z_scaler.transform(frame[z_vars].to_numpy()).astype("float32")
    r = r_encoder.transform(frame[[REGION]]).toarray().astype("float32")
    return x, z, r

x_test, z_test, r_test = arrays(test)
y_test = test[Y].to_numpy(dtype="float32")
teacher = build_teacher(len(X_VARS), len(z_vars), r_test.shape[1])
teacher.load_weights(TEACHER_WEIGHTS)
print("[ok] Loaded teacher weights")

p_test, c_test = teacher.predict([x_test, z_test, r_test], batch_size=4096, verbose=1)
p_test = p_test.reshape(-1)
metrics = {
    "analysis": "DeepCut_teacher_chronological_test",
    "n_test": int(len(y_test)),
    "auc": float(roc_auc_score(y_test, p_test)),
    "accuracy": float(accuracy_score(y_test, p_test >= 0.5)),
    "average_precision": float(average_precision_score(y_test, p_test)),
    "brier": float(brier_score_loss(y_test, p_test)),
    "log_loss": float(log_loss(y_test, np.clip(p_test, 1e-7, 1 - 1e-7))),
}
pd.DataFrame({"y_true": y_test.astype(int), "y_prob": p_test}).to_csv(OUT / "06_teacher_test_predictions.csv", index=False)
pd.DataFrame([metrics]).to_csv(OUT / "06_teacher_test_metrics.csv", index=False)
json_dump(OUT / "06_teacher_test_metrics.json", metrics)

simple = test[[Y, "Air_Temp", "Wet_Bulb_Temp"]].copy()
simple.columns = ["y_true", "T_air", "T_wbt"]
for c in [REGION, date_col]:
    if c in test.columns:
        simple[c] = test[c].to_numpy()
simple.to_csv(OUT / "06_simple_rule_test_input.csv", index=False)

full = pd.concat([train, val, test], ignore_index=True)
x_all, z_all, r_all = arrays(full)
_, cuts = teacher.predict([x_all, z_all, r_all], batch_size=4096, verbose=1)
means, scales = x_scaler.mean_, x_scaler.scale_
thresholds = pd.DataFrame(cuts, columns=["thr_Air_std", "thr_WBT_std", "thr_Dew_std"])
thresholds["thr_Air_C"] = cuts[:, 0] * scales[0] + means[0]
thresholds["thr_WBT_C"] = cuts[:, 1] * scales[1] + means[1]
thresholds["thr_Dew_C"] = cuts[:, 2] * scales[2] + means[2]
thresholds["thr_WBT_C"] = np.maximum(thresholds["thr_WBT_C"], thresholds["thr_Dew_C"])
thresholds["thr_Air_C"] = np.maximum(thresholds["thr_Air_C"], thresholds["thr_WBT_C"])
meta = [c for c in [REGION, "Longitude", "Latitude", "Elevation"] if c in full.columns]
teacher_all = pd.concat([full[meta].reset_index(drop=True), thresholds], axis=1)

# Station means are the only required teacher artifact for the manuscript comparisons.
def station_mode(s: pd.Series):
    m = s.dropna().astype(str).mode()
    return m.iloc[0] if len(m) else np.nan

agg = {"thr_Air_C": "mean", "thr_WBT_C": "mean", "thr_Dew_C": "mean"}
if REGION in teacher_all.columns:
    agg[REGION] = station_mode
if "Elevation" in teacher_all.columns:
    agg["Elevation"] = "mean"
station = teacher_all.groupby(["Longitude", "Latitude"], as_index=False).agg(agg)
station.to_csv(OUT / "06_teacher_station_means.csv", index=False)

if SAVE_EVENT_LEVEL:
    teacher_all.to_csv(OUT / "06_teacher_thresholds_all.csv", index=False)
    print("[ok] Saved event-level teacher thresholds")
else:
    print("[note] Event-level teacher threshold export disabled by VC_SAVE_TEACHER_EVENT_LEVEL=0")

json_dump(OUT / "06_teacher_preprocessing.json", {
    "x_input_vars": X_VARS,
    "z_input_vars": z_vars,
    "x_mean": means.tolist(),
    "x_scale": scales.tolist(),
    "z_mean": z_scaler.mean_.tolist(),
    "z_scale": z_scaler.scale_.tolist(),
    "date_column": date_col,
})
print("[ok] Teacher export complete:", OUT)
