# ======== START: HEADER ========
"""
S2 — TUNING HYPERPARAMETER (HOLDOUT 15%)
Revisi Paper 1 SNN-AE — Scientific Reports

Replikasi fungsi tuning v8.3 (0-snn-ae-unified-v8-3-fair-benchmark):
  - SNN-AE: 50 trial (threshold, decay, surrogate_scale, learning_rate)
  - Dense-AE, LSTM-AE, CNN-AE: 50 trial (learning_rate)
  - TPESampler(seed=42), MedianPruner(8, 5), pruning berbasis -val_loss
  - Objective: F1 pada persentil 99 error rekonstruksi tune_calib,
    diukur pada tune_dev

Perbedaan terhadap v8.3 (terdokumentasi):
  1) Data tuning memakai split group-aware beku dari S1 (R2.5).
  2) Fallback threshold (mean + 2 std saat prediksi runtuh) DIHAPUS;
     kejadian runtuh dicatat (konsistensi dengan naskah; R1.1, R3.5).
  3) Baseline IF, OCSVM, LOF di-tuning dengan grid kecil pada holdout
     yang sama (R1.3, R3.6).
  Non-metodologis: Optuna SQLite (resume), hash skrip, --mode, batch
  prediksi 4096 (hasil inferensi identik, hanya lebih cepat).

Keamanan memori:
  - Satu proses = satu dataset (tidak ada perpindahan dataset di proses
    yang sama). Jalankan dua proses paralel di dua sesi tmux.
  - Hanya baris holdout yang disimpan di memori.
  - Pemeriksaan MemAvailable sebelum setiap trial; proses berhenti aman
    jika memori menipis dan dapat dilanjutkan (resume).

Contoh:
  sudo -u jupyteruser -H bash -c 'MALLOC_ARENA_MAX=2 \\
    /home/jupyteruser/jupyter_env/bin/python -u s2_tuning.py \\
    --dataset edge_iiotset --mode validation 2>&1 | tee logs/s2_edge_validation.log'
  (MALLOC_ARENA_MAX harus diset di shell; glibc membacanya saat proses mulai.)
"""
# ======== END: HEADER ========


# ======== START: ARGUMEN DAN ENVIRONMENT (SEBELUM IMPORT TF) ========
import argparse
import os

parser = argparse.ArgumentParser(description="S2 tuning")
parser.add_argument("--dataset", required=True,
                    choices=["edge_iiotset", "ciciot2023"])
parser.add_argument("--mode", default="validation",
                    choices=["validation", "full"])
parser.add_argument("--threads-intra", type=int, default=14)
parser.add_argument("--threads-inter", type=int, default=2)
parser.add_argument("--min-free-gb", type=float, default=3.0)
ARGS, _ = parser.parse_known_args()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = str(ARGS.threads_intra)
# ======== END: ARGUMEN DAN ENVIRONMENT (SEBELUM IMPORT TF) ========


# ======== START: IMPORT ========
import gc
import hashlib
import json
import random
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tensorflow as tf
import optuna
from optuna.trial import TrialState

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Layer, Dense, BatchNormalization, Input,
    LSTM, RepeatVector, TimeDistributed,
    Conv1D, MaxPooling1D, UpSampling1D, Flatten, Reshape, Cropping1D,
)
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

tf.config.threading.set_intra_op_parallelism_threads(ARGS.threads_intra)
tf.config.threading.set_inter_op_parallelism_threads(ARGS.threads_inter)
# ======== END: IMPORT ========


# ======== START: KONFIGURASI ========
DATASET = ARGS.dataset
MODE = ARGS.mode
VALIDATION = MODE == "validation"
VALIDATION_TRIALS = 2            # trial per model AE pada mode validasi

BASE = Path("/home/jupyteruser/nix-ext-backup/0-eksperimen-final")
WORK = BASE / "0-snn-ae-revision"
CONFIG_PATH = WORK / "00-config" / "config_revision.json"
S1_DIR = WORK / "02-splits" / "full" / DATASET
OUT = WORK / "03-tuning" / MODE / DATASET
OUT.mkdir(parents=True, exist_ok=True)

CONFIG_SECTIONS = {
    "training": {
        "epochs": 50,
        "batch_size": 512,
        "patience": 8,
        "grad_clip": 1.0,
        "latent_dim": 32,
        "lstm_timesteps": 1,
        "snn_base_config": {
            "latent_dim": 32, "time_steps": 3, "threshold": 0.5,
            "decay": 0.7, "surrogate_scale": 10.0, "learning_rate": 1e-4,
        },
        "predict_batch_size": 4096,
        "quantile_scaler": {"n_quantiles": 1000,
                            "output_distribution": "normal",
                            "subsample": 200000},
    },
    "tuning": {
        "models_ae": ["SNN_AE", "Dense_AE", "LSTM_AE", "CNN_AE"],
        "n_trials_snn": 50,
        "n_trials_ann": 50,
        "threshold_percentile": 99,
        "snn_search_space": {
            "threshold": [0.30, 0.90],
            "decay": [0.30, 0.95],
            "surrogate_scale": [2.0, 20.0],
            "learning_rate": [1e-4, 1e-3],
        },
        "ann_lr_range": [1e-4, 1e-3],
        "sampler": "TPESampler(seed=42)",
        "pruner": "MedianPruner(n_startup_trials=8, n_warmup_steps=5)",
        "trial_seed": "random_state + trial.number",
        "objective": "F1 at p99 of tune_calib RE, evaluated on tune_dev; no fallback",
        "baseline_grid": {
            "IF": {"n_estimators": [100, 200], "max_samples": [256, 1024]},
            "OCSVM": {"nu": [0.001, 0.01, 0.05], "gamma": ["scale", 0.01, 0.1]},
            "LOF": {"n_neighbors": [10, 20, 50]},
        },
        "baseline_train_subsample": {"IF": None, "OCSVM": 20000, "LOF": 20000},
        "baseline_score": "-score_samples (higher = more anomalous)",
        "baseline_selection": "max F1 at p99, tie-break by AUC",
    },
}
# ======== END: KONFIGURASI ========


# ======== START: UTILITAS ========
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{now_str()}] [{DATASET}|{MODE}] {msg}", flush=True)


def section_hash(section):
    payload = json.dumps(section, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def file_hash(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except Exception:
        return "interactive"


def atomic_write_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    tmp.replace(path)


def load_json(path, default=None):
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {} if default is None else default


def sync_config(sections):
    """Menambahkan bagian baru; menolak bila bagian yang ada berbeda."""
    cfg = load_json(CONFIG_PATH)
    if "data" not in cfg or "split" not in cfg:
        raise RuntimeError("Config S1 belum ada. Jalankan S1 terlebih dahulu.")
    cfg.setdefault("hashes", {})
    for name, sec in sections.items():
        if name in cfg and section_hash(cfg[name]) != section_hash(sec):
            raise RuntimeError(f"Config bagian '{name}' berbeda dengan yang tersimpan.")
        cfg[name] = sec
        cfg["hashes"][name] = section_hash(sec)
    atomic_write_json(CONFIG_PATH, cfg)
    return cfg


def mem_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 ** 2
    return float("inf")


def guard_memory(where):
    free = mem_available_gb()
    if free < ARGS.min_free_gb:
        raise MemoryError(
            f"MemAvailable {free:.1f} GB < {ARGS.min_free_gb} GB di {where}. "
            f"Proses dihentikan aman; jalankan ulang untuk melanjutkan."
        )


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
# ======== END: UTILITAS ========


# ======== START: MANIFEST RUN DAN CHECKPOINT ========
cfg = sync_config(CONFIG_SECTIONS)
FEATURES = cfg["data"]["features"]
SPLIT = cfg["split"]
TRAIN = cfg["training"]
TUNE = cfg["tuning"]
RS = SPLIT["random_state"]

s1_done = load_json(S1_DIR / "done.json")
if not s1_done or s1_done["run_hash"].get("validation_mode", True):
    raise RuntimeError(f"S1 penuh untuk {DATASET} belum tersedia di {S1_DIR}")

MANIFEST = {
    "dataset": DATASET,
    "mode": MODE,
    "config_hashes": {k: cfg["hashes"][k] for k in
                      ["data", "split", "training", "tuning"]},
    "script_hash": file_hash(globals().get("__file__", "")),
    "s1_sample_hash": s1_done["sample_hash"],
    "s1_splits_hash": s1_done["splits_hash"],
}
manifest_path = OUT / "run_manifest.json"
if manifest_path.exists():
    old = load_json(manifest_path)
    if old != MANIFEST:
        raise RuntimeError(
            f"Manifest berbeda dengan run sebelumnya di {OUT}.\n"
            f"Lama: {old}\nBaru: {MANIFEST}\n"
            f"Hapus folder tersebut secara sadar bila ingin memulai ulang."
        )
else:
    atomic_write_json(manifest_path, MANIFEST)
log(f"Manifest OK | threads intra={ARGS.threads_intra} inter={ARGS.threads_inter} "
    f"| MemAvailable={mem_available_gb():.1f} GB")
# ======== END: MANIFEST RUN DAN CHECKPOINT ========


# ======== START: MUAT DATA HOLDOUT (HEMAT MEMORI) ========
def make_scaler():
    stype = cfg["data"]["scaler_type"][DATASET]
    if stype == "quantile":
        q = TRAIN["quantile_scaler"]
        return QuantileTransformer(n_quantiles=q["n_quantiles"],
                                   output_distribution=q["output_distribution"],
                                   random_state=RS, subsample=q["subsample"])
    if stype == "minmax":
        return MinMaxScaler()
    raise ValueError(stype)


def load_tuning_data():
    guard_memory("load data")
    splits = np.load(S1_DIR / "splits.npz")
    idx_tr = splits["tune_train"]
    idx_cal = splits["tune_calib"]
    idx_dev = splits["tune_dev"]
    splits.close()

    table = pq.read_table(S1_DIR / "sample.parquet",
                          columns=["sample_id", "y"] + FEATURES)
    sid = table.column("sample_id").to_numpy()
    if not np.array_equal(sid, np.arange(len(sid))):
        raise RuntimeError("sample_id tidak berurutan")
    y_all = table.column("y").to_numpy().astype(np.int64)
    X_all = np.column_stack([
        table.column(c).to_numpy(zero_copy_only=False) for c in FEATURES
    ]).astype(np.float64)
    del table, sid
    gc.collect()

    # Pembersihan identik dengan v8.3: inf -> NaN -> 0, float32.
    X_all[~np.isfinite(X_all)] = 0.0
    X_all = X_all.astype(np.float32)

    Xtr, Xcal, Xdev = X_all[idx_tr], X_all[idx_cal], X_all[idx_dev]
    ytr, ycal, ydev = y_all[idx_tr], y_all[idx_cal], y_all[idx_dev]
    del X_all, y_all
    gc.collect()

    if (ytr != 0).any() or (ycal != 0).any():
        raise RuntimeError("KEBOCORAN: tune_train/tune_calib memuat serangan")
    if Xtr.shape[1] != len(FEATURES):
        raise RuntimeError("Dimensi fitur tidak sesuai")

    scaler = make_scaler()
    Xtr = scaler.fit_transform(Xtr).astype(np.float32)   # fit hanya train-normal
    Xcal = scaler.transform(Xcal).astype(np.float32)
    Xdev = scaler.transform(Xdev).astype(np.float32)

    log(f"Data tuning: train_normal={len(Xtr):,} calib_normal={len(Xcal):,} "
        f"dev={len(Xdev):,} (attack_ratio={ydev.mean():.3f}) "
        f"| MemAvailable={mem_available_gb():.1f} GB")
    return {"Xtr": Xtr, "Xcal": Xcal, "Xdev": Xdev, "ydev": ydev,
            "input_dim": int(Xtr.shape[1])}
# ======== END: MUAT DATA HOLDOUT (HEMAT MEMORI) ========


# ======== START: LAPISAN SNN DAN MODEL (SALINAN v8.3) ========
@tf.keras.utils.register_keras_serializable()
class SpikingLayer(Layer):
    def __init__(self, units, threshold=0.5, decay=0.7, surrogate_scale=10.0,
                 time_steps=3, return_sequences=False, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.threshold = float(threshold)
        self.decay = float(decay)
        self.surrogate_scale = float(surrogate_scale)
        self.time_steps = int(time_steps)
        self.return_sequences = bool(return_sequences)

    def build(self, input_shape):
        self.input_dim = int(input_shape[-1])
        self.w = self.add_weight(shape=(self.input_dim, self.units),
                                 initializer=tf.keras.initializers.HeNormal(),
                                 trainable=True, name="weights")
        self.b = self.add_weight(shape=(self.units,), initializer="zeros",
                                 trainable=True, name="bias")
        super().build(input_shape)

    @tf.custom_gradient
    def spike_function(self, membrane, threshold):
        spikes = tf.cast(tf.greater(membrane, threshold), tf.float32)

        def grad(dy):
            diff = membrane - threshold
            sg = self.surrogate_scale / tf.square(
                self.surrogate_scale * tf.abs(diff) + 1.0)
            return dy * sg, None
        return spikes, grad

    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        membrane = tf.zeros((batch_size, self.units), dtype=inputs.dtype)
        spikes = []
        is_seq = len(inputs.shape) == 3
        for t in range(self.time_steps):
            cur_in = inputs[:, t, :] if is_seq else inputs
            cur = tf.matmul(cur_in, self.w) + self.b
            cur = cur / tf.sqrt(tf.cast(tf.shape(cur_in)[-1], tf.float32))
            membrane = self.decay * membrane + cur
            spk = self.spike_function(membrane, self.threshold)
            membrane = membrane - spk * self.threshold
            spikes.append(spk)
        all_spikes = tf.stack(spikes, axis=1)
        if self.return_sequences:
            return all_spikes
        return tf.reduce_mean(all_spikes, axis=1)

    def get_config(self):
        c = super().get_config()
        c.update({"units": self.units, "threshold": self.threshold,
                  "decay": self.decay, "surrogate_scale": self.surrogate_scale,
                  "time_steps": self.time_steps,
                  "return_sequences": self.return_sequences})
        return c


def build_variant(input_dim, c, enc_spiking=True, dec_spiking=True,
                  use_bn=True, name="variant"):
    L = int(c["latent_dim"])
    inp = Input(shape=(input_dim,), name="input")
    x = BatchNormalization(name="batch_norm")(inp) if use_bn else inp
    if enc_spiking:
        x = SpikingLayer(L, threshold=c["threshold"], decay=c["decay"],
                         surrogate_scale=c["surrogate_scale"],
                         time_steps=c["time_steps"],
                         return_sequences=dec_spiking, name="encoder")(x)
    else:
        x = Dense(L, activation="relu", name="encoder")(x)
    if dec_spiking:
        x = SpikingLayer(L, threshold=c["threshold"], decay=c["decay"],
                         surrogate_scale=c["surrogate_scale"],
                         time_steps=c["time_steps"],
                         return_sequences=False, name="decoder")(x)
    else:
        x = Dense(L, activation="relu", name="decoder")(x)
    out = Dense(input_dim, activation="linear", name="output")(x)
    return Model(inp, out, name=name)


def build_lstm_ae(input_dim, latent_dim, timesteps=1):
    if input_dim % timesteps != 0:
        raise ValueError("input_dim harus habis dibagi timesteps")
    nf = input_dim // timesteps
    inp = Input(shape=(timesteps, nf), name="input")
    x = LSTM(64, activation="relu", return_sequences=True, name="enc_lstm1")(inp)
    x = LSTM(latent_dim, activation="relu", return_sequences=False, name="enc_lstm2")(x)
    x = RepeatVector(timesteps, name="repeat")(x)
    x = LSTM(latent_dim, activation="relu", return_sequences=True, name="dec_lstm1")(x)
    x = LSTM(64, activation="relu", return_sequences=True, name="dec_lstm2")(x)
    out = TimeDistributed(Dense(nf), name="output")(x)
    return Model(inp, out, name="LSTM_AE")


def build_cnn_ae(input_dim, latent_dim):
    pooled_len_1 = int((input_dim + 1) // 2)
    pooled_len_2 = int((pooled_len_1 + 1) // 2)
    crop_right = int(pooled_len_2 * 4) - input_dim
    if crop_right < 0:
        raise ValueError("Panjang decoder CNN lebih pendek dari input_dim")
    inp = Input(shape=(input_dim, 1), name="input")
    x = Conv1D(32, 3, activation="relu", padding="same", name="conv1")(inp)
    x = MaxPooling1D(2, padding="same", name="pool1")(x)
    x = Conv1D(16, 3, activation="relu", padding="same", name="conv2")(x)
    x = MaxPooling1D(2, padding="same", name="pool2")(x)
    x = Flatten(name="flatten")(x)
    x = Dense(latent_dim, activation="relu", name="latent")(x)
    x = Dense(16 * pooled_len_2, activation="relu", name="dense_expand")(x)
    x = Reshape((pooled_len_2, 16), name="reshape")(x)
    x = Conv1D(16, 3, activation="relu", padding="same", name="deconv1")(x)
    x = UpSampling1D(2, name="up1")(x)
    x = Conv1D(32, 3, activation="relu", padding="same", name="deconv2")(x)
    x = UpSampling1D(2, name="up2")(x)
    x = Conv1D(1, 3, activation="linear", padding="same", name="deconv_out")(x)
    if crop_right > 0:
        x = Cropping1D(cropping=(0, crop_right), name="crop_to_input_length")(x)
    out = Flatten(name="output")(x)
    return Model(inp, out, name="CNN_AE")
# ======== END: LAPISAN SNN DAN MODEL (SALINAN v8.3) ========


# ======== START: ERROR REKONSTRUKSI DAN METRIK ========
def reshape_for_model(X, kind):
    if kind == "lstm":
        T = TRAIN["lstm_timesteps"]
        nf = X.shape[1] // T
        return X[:, :T * nf].reshape(-1, T, nf)
    if kind == "cnn":
        return X.reshape(-1, X.shape[1], 1)
    return X


def recon_errors(model, X, kind):
    Xr = reshape_for_model(X, kind)
    pred = model.predict(Xr, batch_size=TRAIN["predict_batch_size"], verbose=0)
    if kind == "lstm":
        return np.mean((Xr - pred) ** 2, axis=(1, 2))
    if kind == "cnn":
        pf = pred.reshape(pred.shape[0], -1)
        md = min(X.shape[1], pf.shape[1])
        return np.mean((X[:, :md] - pf[:, :md]) ** 2, axis=1)
    return np.mean((X - pred) ** 2, axis=1)


def metrics_at_percentile(calib_scores, dev_scores, y_dev, p):
    """Threshold dari data kalibrasi normal saja. Tanpa fallback."""
    thr = float(np.percentile(calib_scores, p))
    y_pred = (dev_scores > thr).astype(int)
    collapsed = bool(len(np.unique(y_pred)) == 1)
    tn, fp, fn, tp = confusion_matrix(y_dev, y_pred, labels=[0, 1]).ravel()
    try:
        auc = float(roc_auc_score(y_dev, dev_scores))
    except Exception:
        auc = float("nan")
    return {
        "threshold": thr,
        "accuracy": float(accuracy_score(y_dev, y_pred)),
        "precision": float(precision_score(y_dev, y_pred, zero_division=0)),
        "recall": float(recall_score(y_dev, y_pred, zero_division=0)),
        "f1": float(f1_score(y_dev, y_pred, zero_division=0)),
        "auc": auc, "collapsed": collapsed,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
# ======== END: ERROR REKONSTRUKSI DAN METRIK ========


# ======== START: OBJECTIVE OPTUNA (REPLIKASI v8.3) ========
class OptunaPruneCallback(Callback):
    def __init__(self, trial):
        super().__init__()
        self.trial = trial

    def on_epoch_end(self, epoch, logs=None):
        val_loss = float((logs or {}).get("val_loss", 0.0))
        self.trial.report(-val_loss, epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


def fit_and_score(trial, model, kind, data, clip, lr):
    opt = Adam(lr, clipnorm=TRAIN["grad_clip"]) if clip else Adam(lr)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    Xtr = reshape_for_model(data["Xtr"], kind)
    Xcal = reshape_for_model(data["Xcal"], kind)
    ytr = data["Xtr"] if kind == "cnn" else Xtr
    ycal = data["Xcal"] if kind == "cnn" else Xcal
    cbs = [EarlyStopping(monitor="val_loss", patience=TRAIN["patience"],
                         restore_best_weights=True, verbose=0),
           OptunaPruneCallback(trial)]
    model.fit(Xtr, ytr, validation_data=(Xcal, ycal), epochs=TRAIN["epochs"],
              batch_size=TRAIN["batch_size"], callbacks=cbs, verbose=0)
    ve = recon_errors(model, data["Xcal"], kind)
    te = recon_errors(model, data["Xdev"], kind)
    m = metrics_at_percentile(ve, te, data["ydev"], TUNE["threshold_percentile"])
    for k in ["auc", "precision", "recall", "collapsed"]:
        trial.set_user_attr(k, m[k])
    trial.set_user_attr("f1", m["f1"])
    return m["f1"]


def objective(trial, model_id, data):
    guard_memory(f"{model_id} trial {trial.number}")
    set_global_seed(RS + trial.number)
    try:
        if model_id == "SNN_AE":
            ss = TUNE["snn_search_space"]
            c = dict(TRAIN["snn_base_config"])
            c.update({
                "threshold": trial.suggest_float("threshold", *ss["threshold"]),
                "decay": trial.suggest_float("decay", *ss["decay"]),
                "surrogate_scale": trial.suggest_float("surrogate_scale", *ss["surrogate_scale"]),
                "learning_rate": trial.suggest_float("learning_rate", *ss["learning_rate"], log=True),
            })
            model = build_variant(data["input_dim"], c, True, True, name="SNN_AE_tune")
            return fit_and_score(trial, model, "snn", data, clip=True,
                                 lr=c["learning_rate"])

        lr = trial.suggest_float("learning_rate", *TUNE["ann_lr_range"], log=True)
        if model_id == "Dense_AE":
            c = dict(TRAIN["snn_base_config"])
            c["learning_rate"] = lr
            model, kind, clip = build_variant(data["input_dim"], c, False, False,
                                              name="Dense_AE_tune"), "snn", True
        elif model_id == "LSTM_AE":
            model, kind, clip = build_lstm_ae(data["input_dim"], TRAIN["latent_dim"],
                                              TRAIN["lstm_timesteps"]), "lstm", False
        elif model_id == "CNN_AE":
            model, kind, clip = build_cnn_ae(data["input_dim"], TRAIN["latent_dim"]), "cnn", False
        else:
            raise ValueError(model_id)
        return fit_and_score(trial, model, kind, data, clip=clip, lr=lr)
    finally:
        tf.keras.backend.clear_session()
        gc.collect()
# ======== END: OBJECTIVE OPTUNA (REPLIKASI v8.3) ========


# ======== START: TUNING AE DENGAN RESUME ========
def run_ae_tuning(data, best_all):
    storage = f"sqlite:///{OUT / 'optuna.db'}"
    for model_id in TUNE["models_ae"]:
        key = f"{DATASET}|{model_id}"
        target = VALIDATION_TRIALS if VALIDATION else (
            TUNE["n_trials_snn"] if model_id == "SNN_AE" else TUNE["n_trials_ann"])

        study = optuna.create_study(
            study_name=key, storage=storage, direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RS),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=5),
            load_if_exists=True)

        # Trial RUNNING yang tertinggal dari proses terputus ditandai FAIL.
        for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
            study._storage.set_trial_state_values(t._trial_id, state=TrialState.FAIL)

        done = len(study.get_trials(deepcopy=False,
                                    states=(TrialState.COMPLETE, TrialState.PRUNED)))
        remaining = max(0, target - done)
        log(f"{model_id}: {done}/{target} trial selesai, sisa {remaining}")

        t0 = time.time()
        if remaining:
            study.optimize(lambda t, m=model_id: objective(t, m, data),
                           n_trials=remaining, gc_after_trial=True)
        elapsed = time.time() - t0

        df = study.trials_dataframe()
        df.to_csv(OUT / f"{model_id}_optuna_trials.csv", index=False)
        n_collapsed = int(sum(bool(t.user_attrs.get("collapsed", False))
                              for t in study.get_trials(deepcopy=False)))

        best = study.best_trial
        if model_id == "SNN_AE":
            params = dict(TRAIN["snn_base_config"])
            params.update(best.params)
        else:
            params = {"learning_rate": float(best.params["learning_rate"])}
        best_all[key] = {
            "model": model_id, "dataset": DATASET,
            "best_f1": float(best.value),
            "best_auc": best.user_attrs.get("auc"),
            "best_trial_number": int(best.number),
            "best_params": params,
            "n_trials_used": len([t for t in study.get_trials(deepcopy=False)
                                  if t.state in (TrialState.COMPLETE, TrialState.PRUNED)]),
            "n_trials_collapsed": n_collapsed,
            "manifest": MANIFEST,
        }
        atomic_write_json(OUT / "best_hparams.json", best_all)
        per_trial = elapsed / remaining if remaining else float("nan")
        log(f"{model_id}: best F1={best.value:.4f} params={params} "
            f"| {elapsed / 60:.1f} menit ({per_trial:.0f} s/trial) "
            f"| MemAvailable={mem_available_gb():.1f} GB")
        if VALIDATION and remaining:
            full_n = TUNE["n_trials_snn"] if model_id == "SNN_AE" else TUNE["n_trials_ann"]
            log(f"{model_id}: estimasi run penuh ≈ {per_trial * full_n / 3600:.1f} jam "
                f"(batas atas; pruning mempercepat)")
    return best_all
# ======== END: TUNING AE DENGAN RESUME ========


# ======== START: TUNING BASELINE (IF, OCSVM, LOF) ========
def grid_list(grid):
    keys = list(grid)
    combos = [{}]
    for k in keys:
        combos = [dict(c, **{k: v}) for c in combos for v in grid[k]]
    return combos


def make_baseline(name, params):
    if name == "IF":
        return IsolationForest(n_estimators=params["n_estimators"],
                               max_samples=params["max_samples"],
                               random_state=RS, n_jobs=ARGS.threads_intra)
    if name == "OCSVM":
        return OneClassSVM(kernel="rbf", nu=params["nu"], gamma=params["gamma"],
                           cache_size=1000)
    if name == "LOF":
        return LocalOutlierFactor(n_neighbors=params["n_neighbors"], novelty=True,
                                  n_jobs=ARGS.threads_intra)
    raise ValueError(name)


def run_baseline_tuning(data, best_all):
    grid_path = OUT / "baselines_grid.json"
    results = load_json(grid_path, default={})
    sub_cfg = TUNE["baseline_train_subsample"]

    for name, grid in TUNE["baseline_grid"].items():
        combos = grid_list(grid)
        if VALIDATION:
            combos = combos[:1]
        n_sub = sub_cfg.get(name)
        if n_sub and n_sub < len(data["Xtr"]):
            idx = np.sort(np.random.RandomState(RS).choice(
                len(data["Xtr"]), size=n_sub, replace=False))
            Xfit = data["Xtr"][idx]
        else:
            Xfit = data["Xtr"]

        for params in combos:
            key = f"{name}|{json.dumps(params, sort_keys=True)}"
            if key in results:
                continue
            guard_memory(f"baseline {key}")
            t0 = time.time()
            model = make_baseline(name, params)
            model.fit(Xfit)
            fit_s = time.time() - t0
            cal = -model.score_samples(data["Xcal"])
            dev = -model.score_samples(data["Xdev"])
            m = metrics_at_percentile(cal, dev, data["ydev"],
                                      TUNE["threshold_percentile"])
            extra = {}
            if name == "OCSVM":
                extra["n_support_vectors"] = int(model.support_vectors_.shape[0])
            results[key] = {"model": name, "params": params,
                            "n_train": int(len(Xfit)), "fit_seconds": fit_s,
                            "total_seconds": time.time() - t0, **m, **extra}
            atomic_write_json(grid_path, results)
            log(f"{key}: F1={m['f1']:.4f} AUC={m['auc']:.4f} "
                f"collapsed={m['collapsed']} | {time.time() - t0:.0f} s")
            del model
            gc.collect()

        rows = [r for r in results.values() if r["model"] == name]
        rows.sort(key=lambda r: (r["f1"], -1 if np.isnan(r["auc"]) else r["auc"]),
                  reverse=True)
        best = rows[0]
        best_all[f"{DATASET}|{name}"] = {
            "model": name, "dataset": DATASET, "best_f1": best["f1"],
            "best_auc": best["auc"], "best_params": best["params"],
            "n_train": best["n_train"], "n_grid": len(rows),
            "manifest": MANIFEST,
        }
        atomic_write_json(OUT / "best_hparams.json", best_all)

    pd.DataFrame(list(results.values())).to_csv(OUT / "baselines_grid.csv", index=False)
    return best_all
# ======== END: TUNING BASELINE (IF, OCSVM, LOF) ========


# ======== START: EKSEKUSI ========
t_start = time.time()
log("S2 mulai")
data = load_tuning_data()
best_all = load_json(OUT / "best_hparams.json", default={})
best_all = run_ae_tuning(data, best_all)
best_all = run_baseline_tuning(data, best_all)

atomic_write_json(OUT / "done.json", {
    "dataset": DATASET, "mode": MODE, "manifest": MANIFEST,
    "finished_at": now_str(),
    "elapsed_hours": round((time.time() - t_start) / 3600, 2),
})
log(f"S2 selesai | {(time.time() - t_start) / 60:.1f} menit | output: {OUT}")
# ======== END: EKSEKUSI ========
