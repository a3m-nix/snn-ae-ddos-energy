# ======== START: HEADER ========
"""
S3 — EVALUASI FINAL (3 SEED x 10 FOLD)
Revisi Paper 1 SNN-AE — Scientific Reports

Replikasi run_final_evaluation dan train_eval_job v8.3, dengan:
  - Split beku group-aware dari S1 (R2.5)
  - Hyperparameter hasil S2
  - Job lama: SNN_AE, LSTM_AE, CNN_AE, SNN_Dense, Dense_SNN, Dense_AE,
    SNN_AE_T2/T5/T7
  - Job baru: SNN_AE_noBN (R3.9, R2.14), IF/OCSVM/LOF (R1.3, R3.6)
  - Tanpa fallback threshold; kejadian runtuh dicatat
  - Grid persentil p95–p99.9 + FPR test aktual (R1.1, R2.12, R3.5)
  - AUC-PR dan recall per subtipe (R2.8)
  - Spike rate encoder/decoder per neuron, normal vs attack (R3.4, R3.2b)
  - Sampel latent spike SNN_AE (R3.10)
  - Seluruh model, scaler, error per sampel disimpan (tanpa run ulang)
Latensi di S3 hanya arsip; latensi naskah diukur di S5 tanpa beban.

Eksekusi (satu proses = satu dataset, unit kerja = (seed, fold)):
  --worker i --n-workers k  membagi 30 unit secara bergiliran.
  Rekomendasi: Edge 2 worker, CIC 4 worker, 5 thread per worker.
  Jalankan dengan MALLOC_ARENA_MAX=2 di shell.
"""
# ======== END: HEADER ========


# ======== START: ARGUMEN DAN ENVIRONMENT (SEBELUM IMPORT TF) ========
import argparse
import os

parser = argparse.ArgumentParser(description="S3 final evaluation")
parser.add_argument("--dataset", required=True,
                    choices=["edge_iiotset", "ciciot2023"])
parser.add_argument("--mode", default="validation",
                    choices=["validation", "full"])
parser.add_argument("--worker", type=int, default=0)
parser.add_argument("--n-workers", type=int, default=1)
parser.add_argument("--threads-intra", type=int, default=5)
parser.add_argument("--threads-inter", type=int, default=1)
parser.add_argument("--min-free-gb", type=float, default=2.5)
ARGS, _ = parser.parse_known_args()
if not 0 <= ARGS.worker < ARGS.n_workers:
    raise SystemExit("--worker harus 0 <= worker < n-workers")

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

import joblib
import numpy as np
import pyarrow.parquet as pq
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Layer, Dense, BatchNormalization, Input,
    LSTM, RepeatVector, TimeDistributed,
    Conv1D, MaxPooling1D, UpSampling1D, Flatten, Reshape, Cropping1D,
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

warnings.filterwarnings("ignore")
tf.config.threading.set_intra_op_parallelism_threads(ARGS.threads_intra)
tf.config.threading.set_inter_op_parallelism_threads(ARGS.threads_inter)
# ======== END: IMPORT ========


# ======== START: KONFIGURASI ========
DATASET = ARGS.dataset
MODE = ARGS.mode
VALIDATION = MODE == "validation"

BASE = Path("/home/jupyteruser/nix-ext-backup/0-eksperimen-final")
WORK = BASE / "0-snn-ae-revision"
CONFIG_PATH = WORK / "00-config" / "config_revision.json"
S1_DIR = WORK / "02-splits" / "full" / DATASET
S2_DIR = WORK / "03-tuning" / "full" / DATASET
OUT = WORK / "04-final-eval" / MODE / DATASET
OUT.mkdir(parents=True, exist_ok=True)

AE_JOBS = [
    # id, model, enc, dec, T (None = pakai konfigurasi), use_bn, study
    ("SNN_AE",      "SNN_AE",    True,  True,  3,    True,  "benchmark+arch+timestep"),
    ("LSTM_AE",     "LSTM_AE",   None,  None,  None, None,  "benchmark"),
    ("CNN_AE",      "CNN_AE",    None,  None,  None, None,  "benchmark"),
    ("SNN_Dense",   "SNN_Dense", True,  False, 3,    True,  "arch"),
    ("Dense_SNN",   "Dense_SNN", False, True,  3,    True,  "arch"),
    ("Dense_AE",    "Dense_AE",  False, False, 3,    True,  "benchmark+arch"),
    ("SNN_AE_T2",   "SNN_AE",    True,  True,  2,    True,  "timestep"),
    ("SNN_AE_T5",   "SNN_AE",    True,  True,  5,    True,  "timestep"),
    ("SNN_AE_T7",   "SNN_AE",    True,  True,  7,    True,  "timestep"),
    ("SNN_AE_noBN", "SNN_AE",    True,  True,  3,    False, "ablation_bn"),
]
BASELINE_JOBS = ["IF", "OCSVM", "LOF"]

CONFIG_SECTIONS = {
    "evaluation": {
        "ae_jobs": [list(j) for j in AE_JOBS],
        "baseline_jobs": BASELINE_JOBS,
        "percentile_grid": [95, 96, 97, 98, 99, 99.5, 99.9],
        "operating_percentile": 99,
        "threshold_source": "validation-normal only; no fallback",
        "model_seed": "random_state + 1000*seed_idx + fold",
        "spike_subset_n": 1000,
        "spike_neuron_n": 2000,
        "latent_n_per_class": 1000,
        "latency": {"batch": 100, "repeats": 50,
                    "note": "archival only; manuscript latency from S5"},
        "brian2": {"enabled": True, "sample_size": 256,
                   "record_neurons": 8, "run_for": ["SNN_AE"]},
    },
    "energy": {
        "E_AC": 0.9, "E_MAC": 4.6, "E_LIF": 0.1,
        "model": "Horowitz 45 nm, arithmetic only (as v8.3); "
                 "BN cost 0 when BN removed",
    },
}
# ======== END: KONFIGURASI ========


# ======== START: UTILITAS ========
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    tag = f"{DATASET}|{MODE}|w{ARGS.worker}/{ARGS.n_workers}"
    print(f"[{now_str()}] [{tag}] {msg}", flush=True)


def section_hash(section):
    payload = json.dumps(section, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def file_hash(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except Exception:
        return "interactive"


def to_serial(obj):
    """Nilai JSON ketat: NaN/Inf menjadi null."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): to_serial(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_serial(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_serial(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def atomic_write_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(to_serial(data), indent=2, ensure_ascii=False,
                              allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def atomic_savez(path, **arrays):
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def atomic_joblib(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    joblib.dump(obj, tmp)
    tmp.replace(path)


def atomic_keras_save(path, model):
    path = Path(path)
    tmp = path.with_name("tmp_" + path.name)
    model.save(tmp)
    tmp.replace(path)


def load_json(path, default=None):
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {} if default is None else default


def sync_config(sections):
    cfg = load_json(CONFIG_PATH)
    for need in ["data", "split", "training", "tuning"]:
        if need not in cfg:
            raise RuntimeError(f"Config bagian '{need}' belum ada (S1/S2).")
    cfg.setdefault("hashes", {})
    for name, sec in sections.items():
        if name in cfg and section_hash(cfg[name]) != section_hash(sec):
            raise RuntimeError(f"Config bagian '{name}' berbeda dengan yang tersimpan.")
        cfg[name] = sec
        cfg["hashes"][name] = section_hash(sec)
    atomic_write_json(CONFIG_PATH, cfg)
    return json.loads(json.dumps(cfg))


def mem_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 ** 2
    return float("inf")


def guard_memory(where):
    free = mem_available_gb()
    if free < ARGS.min_free_gb:
        raise MemoryError(f"MemAvailable {free:.1f} GB < {ARGS.min_free_gb} GB "
                          f"di {where}. Berhenti aman; jalankan ulang untuk lanjut.")


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
# ======== END: UTILITAS ========


# ======== START: MANIFEST RUN ========
cfg = sync_config(CONFIG_SECTIONS)
FEATURES = cfg["data"]["features"]
SPLIT, TRAIN, TUNE = cfg["split"], cfg["training"], cfg["tuning"]
EVAL, ENERGY = cfg["evaluation"], cfg["energy"]
RS = SPLIT["random_state"]
PB = TRAIN["predict_batch_size"]

s1_done = load_json(S1_DIR / "done.json")
s2_done = load_json(S2_DIR / "done.json")
if not s1_done or s1_done["run_hash"].get("validation_mode", True):
    raise RuntimeError(f"S1 penuh belum tersedia: {S1_DIR}")
if not s2_done:
    raise RuntimeError(f"S2 penuh belum selesai: {S2_DIR}")
BEST = load_json(S2_DIR / "best_hparams.json")

MANIFEST = {
    "dataset": DATASET, "mode": MODE,
    "config_hashes": {k: cfg["hashes"][k] for k in
                      ["data", "split", "training", "tuning", "evaluation", "energy"]},
    "script_hash": file_hash(globals().get("__file__", "")),
    "s1_sample_hash": s1_done["sample_hash"],
    "s1_splits_hash": s1_done["splits_hash"],
    "s2_best_hparams_hash": file_hash(S2_DIR / "best_hparams.json"),
}
manifest_path = OUT / "run_manifest.json"
if manifest_path.exists():
    if load_json(manifest_path) != to_serial(MANIFEST):
        raise RuntimeError(f"Manifest berbeda dengan run sebelumnya di {OUT}. "
                           f"Hapus folder secara sadar bila ingin memulai ulang.")
else:
    atomic_write_json(manifest_path, MANIFEST)
log(f"Manifest OK | threads {ARGS.threads_intra}/{ARGS.threads_inter} "
    f"| MemAvailable={mem_available_gb():.1f} GB")
# ======== END: MANIFEST RUN ========


# ======== START: MUAT DATA ========
def load_dataset():
    guard_memory("load data")
    table = pq.read_table(S1_DIR / "sample.parquet",
                          columns=["sample_id", "y", "label_type"] + FEATURES)
    sid = table.column("sample_id").to_numpy()
    if not np.array_equal(sid, np.arange(len(sid))):
        raise RuntimeError("sample_id tidak berurutan")
    y = table.column("y").to_numpy().astype(np.int64)
    label_type = np.asarray(table.column("label_type").to_pylist(), dtype=object)
    X = np.column_stack([table.column(c).to_numpy(zero_copy_only=False)
                         for c in FEATURES]).astype(np.float64)
    del table, sid
    gc.collect()
    X[~np.isfinite(X)] = 0.0                  # identik v8.3: inf/NaN -> 0
    return X.astype(np.float32), y, label_type


X_ALL, Y_ALL, LT_ALL = load_dataset()
SPLITS = dict(np.load(S1_DIR / "splits.npz"))
HOLDOUT = SPLITS["holdout"]
log(f"Data: {X_ALL.shape} | MemAvailable={mem_available_gb():.1f} GB")
# ======== END: MUAT DATA ========


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
    p1 = int((input_dim + 1) // 2)
    p2 = int((p1 + 1) // 2)
    crop_right = int(p2 * 4) - input_dim
    inp = Input(shape=(input_dim, 1), name="input")
    x = Conv1D(32, 3, activation="relu", padding="same", name="conv1")(inp)
    x = MaxPooling1D(2, padding="same", name="pool1")(x)
    x = Conv1D(16, 3, activation="relu", padding="same", name="conv2")(x)
    x = MaxPooling1D(2, padding="same", name="pool2")(x)
    x = Flatten(name="flatten")(x)
    x = Dense(latent_dim, activation="relu", name="latent")(x)
    x = Dense(16 * p2, activation="relu", name="dense_expand")(x)
    x = Reshape((p2, 16), name="reshape")(x)
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
    pred = model.predict(Xr, batch_size=PB, verbose=0)
    if kind == "lstm":
        return np.mean((Xr - pred) ** 2, axis=(1, 2))
    if kind == "cnn":
        pf = pred.reshape(pred.shape[0], -1)
        md = min(X.shape[1], pf.shape[1])
        return np.mean((X[:, :md] - pf[:, :md]) ** 2, axis=1)
    return np.mean((X - pred) ** 2, axis=1)


def evaluate_scores(val_s, test_s, y_te, lt_te):
    """Threshold hanya dari validation-normal. Tanpa fallback."""
    per = {}
    for p in EVAL["percentile_grid"]:
        thr = float(np.percentile(val_s, p))
        pred = (test_s > thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
        per[str(p)] = {
            "threshold": thr,
            "accuracy": float(accuracy_score(y_te, pred)),
            "precision": float(precision_score(y_te, pred, zero_division=0)),
            "recall": float(recall_score(y_te, pred, zero_division=0)),
            "f1": float(f1_score(y_te, pred, zero_division=0)),
            "fpr_test": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
            "fpr_target": round(1.0 - p / 100.0, 6),
            "collapsed": bool(len(np.unique(pred)) == 1),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }
    try:
        auc = float(roc_auc_score(y_te, test_s))
        auc_pr = float(average_precision_score(y_te, test_s))
    except Exception:
        auc, auc_pr = float("nan"), float("nan")

    thr_op = per[str(EVAL["operating_percentile"])]["threshold"]
    subtype = {}
    att = y_te == 1
    for name in sorted(set(lt_te[att])):
        m = att & (lt_te == name)
        subtype[str(name)] = {"n": int(m.sum()),
                              "recall_op": float((test_s[m] > thr_op).mean())}
    return per, auc, auc_pr, subtype


def score_summary(val_s, test_s, y_te):
    def stats(prefix, a):
        a = np.asarray(a, dtype=float)
        if a.size == 0:
            return {f"{prefix}_n": 0}
        return {f"{prefix}_n": int(a.size), f"{prefix}_mean": float(a.mean()),
                f"{prefix}_std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
                f"{prefix}_min": float(a.min()),
                f"{prefix}_p50": float(np.percentile(a, 50)),
                f"{prefix}_p95": float(np.percentile(a, 95)),
                f"{prefix}_p99": float(np.percentile(a, 99)),
                f"{prefix}_max": float(a.max())}
    out = {}
    out.update(stats("re_val_normal", val_s))
    out.update(stats("re_test_mixed", test_s))
    out.update(stats("re_test_normal", test_s[y_te == 0]))
    out.update(stats("re_test_attack", test_s[y_te == 1]))
    return out


def measure_latency(predict_fn, X):
    """Arsip saja; nilai naskah diukur di S5 tanpa beban."""
    b, r = EVAL["latency"]["batch"], EVAL["latency"]["repeats"]
    Xb = X[:max(b, 10)]
    for _ in range(5):
        predict_fn(Xb[:10])
    lat = []
    for _ in range(r):
        t0 = time.time()
        predict_fn(Xb[:b])
        lat.append((time.time() - t0) * 1000.0 / b)
    lat = np.array(lat)
    return {"mean_ms": float(lat.mean()), "std_ms": float(lat.std(ddof=1)),
            "throughput_sps": float(1000.0 / lat.mean()),
            "note": "archival; measured under parallel load"}
# ======== END: ERROR REKONSTRUKSI DAN METRIK ========


# ======== START: SPIKE DAN ENERGI (REPLIKASI v8.3 + PER NEURON) ========
def spiking_rates(model, layer_name, X):
    """Laju spike per sampel per neuron (rata-rata atas time step)."""
    names = [l.name for l in model.layers]
    if layer_name not in names:
        return None
    layer = model.get_layer(layer_name)
    if not isinstance(layer, SpikingLayer):
        return None
    sub = Model(model.input, layer.output)
    out = sub.predict(X, batch_size=PB, verbose=0)
    del sub
    return out.mean(axis=1) if out.ndim == 3 else out


def spike_analysis(model, X_te, y_te, lt_te):
    n_e = EVAL["spike_subset_n"]
    n_n = EVAL["spike_neuron_n"]
    subsets = {"mixed": X_te[:n_e], "normal": X_te[y_te == 0][:n_e],
               "attack": X_te[y_te == 1][:n_e]}
    rates, per_neuron = {}, {}
    for layer in ["encoder", "decoder"]:
        for name, Xs in subsets.items():
            r = spiking_rates(model, layer, Xs) if len(Xs) else None
            rates[f"{layer}_rate_{name}"] = float(r.mean()) if r is not None else float("nan")
        for cls, yv in [("normal", 0), ("attack", 1)]:
            Xs = X_te[y_te == yv][:n_n]
            r = spiking_rates(model, layer, Xs) if len(Xs) else None
            if r is not None:
                per_neuron[f"{layer}_{cls}_per_sample"] = r.astype(np.float32)
    return rates, per_neuron


def snn_energy(model, c, enc, dec, use_bn, rates):
    """Rincian energi v8.3; BN = 0 bila BN dihapus."""
    E_MAC, E_AC, E_LIF = ENERGY["E_MAC"], ENERGY["E_AC"], ENERGY["E_LIF"]
    T, L = int(c["time_steps"]), int(c["latent_dim"])
    din = int(model.input_shape[-1])
    dout = int(model.output_shape[-1])
    bn = din * 4 * E_MAC if use_bn else 0.0
    input_mac = din * L * E_MAC
    enc_rate = rates.get("encoder_rate_mixed", float("nan"))
    fan_ac = enc_rate * L * T * L * E_AC if (enc and np.isfinite(enc_rate)) else 0.0
    if enc and dec:
        internal_mac = 0.0
    elif enc and not dec:
        internal_mac = L * L * E_MAC
    elif (not enc) and dec:
        internal_mac = L * L * T * E_MAC
    else:
        internal_mac = L * L * E_MAC
    lif = (int(enc) + int(dec)) * L * T * E_LIF
    out_mac = L * dout * E_MAC
    total = bn + input_mac + fan_ac + internal_mac + lif + out_mac
    return {"batchnorm_pj": bn, "input_mac_pj": input_mac,
            "internal_ac_pj": fan_ac, "encoder_spike_fanout_ac_pj": fan_ac,
            "internal_dense_mac_pj": internal_mac, "lif_update_pj": lif,
            "dense_output_mac_pj": out_mac, "total_pj": total,
            "spike_rate": enc_rate,
            "activation_sparsity": 1.0 - enc_rate if np.isfinite(enc_rate) else float("nan"),
            **rates}


def lstm_energy(input_dim):
    T = TRAIN["lstm_timesteps"]
    nf = input_dim // T

    def m(n_in, n_h, t):
        return 4 * (n_in + n_h) * n_h * t
    macs = m(nf, 64, T) + m(64, 32, T) + m(32, 32, T) + m(32, 64, T) + 64 * nf * T
    return float(macs * ENERGY["E_MAC"])


def cnn_energy(input_dim):
    def conv(length, cin, k, cout):
        return length * k * cin * cout
    L = TRAIN["latent_dim"]
    p1 = int((input_dim + 1) // 2)
    p2 = int((p1 + 1) // 2)
    u1, u2 = p2 * 2, p2 * 4
    macs = (conv(input_dim, 1, 3, 32) + conv(p1, 32, 3, 16) + p2 * 16 * L
            + L * p2 * 16 + conv(p2, 16, 3, 16) + conv(u1, 16, 3, 32)
            + conv(u2, 32, 3, 1))
    return float(macs * ENERGY["E_MAC"])
# ======== END: SPIKE DAN ENERGI (REPLIKASI v8.3 + PER NEURON) ========


# ======== START: BRIAN2 (REPLIKASI v8.3) ========
def brian2_simulation(model, X_sample, c, outdir):
    from brian2 import (start_scope, defaultclock, NeuronGroup,
                        SpikeMonitor, StateMonitor, ms, run)
    b2 = EVAL["brian2"]
    enc = model.get_layer("encoder")
    Xb = X_sample[:min(b2["sample_size"], len(X_sample))].astype(np.float32)
    W, b = enc.get_weights()
    units, din = W.shape[1], W.shape[0]
    T, beta, vth = int(c["time_steps"]), float(c["decay"]), float(c["threshold"])
    I_all = (Xb @ W + b) / np.sqrt(float(din))

    counts, rates = [], []
    first = {}
    for i in range(len(I_all)):
        start_scope()
        defaultclock.dt = 1 * ms
        eqs = """
        dv/dt = ((beta - 1) * v + I) / dt_sim : 1
        I : 1
        beta : 1
        vth : 1
        dt_sim : second
        """
        G = NeuronGroup(units, model=eqs, threshold="v > vth",
                        reset="v -= vth", method="euler")
        G.v = 0
        G.I = I_all[i].astype(np.float64)
        G.beta = beta
        G.vth = vth
        G.dt_sim = 1 * ms
        sp = SpikeMonitor(G)
        st = StateMonitor(G, "v", record=list(range(min(b2["record_neurons"], units))))
        run(T * ms)
        n = int(sp.num_spikes)
        counts.append(n)
        rates.append(n / float(units * T))
        if i == 0:
            first = {"spike_i": np.asarray(sp.i), "spike_t": np.asarray(sp.t / ms),
                     "state_t": np.asarray(st.t / ms), "state_v": np.asarray(st.v)}

    counts = np.asarray(counts, dtype=float)
    rates = np.asarray(rates, dtype=float)
    L = int(c["latent_dim"])
    outdir.mkdir(parents=True, exist_ok=True)
    atomic_savez(outdir / "brian2_arrays.npz", spike_counts=counts,
                 spike_rates=rates, **first)
    return {"n_samples": int(len(Xb)),
            "mean_spike_count": float(counts.mean()),
            "mean_spike_rate": float(rates.mean()),
            "std_spike_rate": float(rates.std(ddof=1)) if len(rates) > 1 else 0.0,
            "mean_activation_sparsity": float(1.0 - rates.mean()),
            "mean_synops_encoder_to_decoder": float(counts.mean() * L),
            "mean_ac_energy_encoder_to_decoder_pj": float(counts.mean() * L * ENERGY["E_AC"]),
            "note": "Brian2 uses trained Keras encoder weights; LIF inspection only"}
# ======== END: BRIAN2 (REPLIKASI v8.3) ========


# ======== START: JOB AUTOENCODER ========
def snn_cfg():
    c = dict(BEST[f"{DATASET}|SNN_AE"]["best_params"])
    c["latent_dim"] = TRAIN["latent_dim"]
    c["time_steps"] = TRAIN["snn_base_config"]["time_steps"]
    return c


def run_ae_job(job, data, seed, jdir):
    job_id, model_id, enc, dec, T, use_bn, study = job
    c = snn_cfg()
    if T is not None:
        c["time_steps"] = int(T)
    set_global_seed(seed)
    din = data["Xtr"].shape[1]

    if model_id in ["SNN_AE", "SNN_Dense", "Dense_SNN"]:
        model = build_variant(din, c, enc, dec, use_bn=use_bn, name=job_id)
        lr, clip, kind = float(c["learning_rate"]), True, "snn"
    elif model_id == "Dense_AE":
        lr = float(BEST[f"{DATASET}|Dense_AE"]["best_params"]["learning_rate"])
        c["learning_rate"] = lr
        model = build_variant(din, c, False, False, use_bn=True, name=job_id)
        clip, kind = True, "snn"
    elif model_id == "LSTM_AE":
        lr = float(BEST[f"{DATASET}|LSTM_AE"]["best_params"]["learning_rate"])
        model = build_lstm_ae(din, TRAIN["latent_dim"], TRAIN["lstm_timesteps"])
        clip, kind = False, "lstm"
    elif model_id == "CNN_AE":
        lr = float(BEST[f"{DATASET}|CNN_AE"]["best_params"]["learning_rate"])
        model = build_cnn_ae(din, TRAIN["latent_dim"])
        clip, kind = False, "cnn"
    else:
        raise ValueError(model_id)

    opt = Adam(lr, clipnorm=TRAIN["grad_clip"]) if clip else Adam(lr)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    Xtr = reshape_for_model(data["Xtr"], kind)
    Xvl = reshape_for_model(data["Xval"], kind)
    ytr = data["Xtr"] if kind == "cnn" else Xtr
    yvl = data["Xval"] if kind == "cnn" else Xvl

    es = EarlyStopping(monitor="val_loss", patience=TRAIN["patience"],
                       restore_best_weights=True, verbose=0)
    t0 = time.time()
    hist = model.fit(Xtr, ytr, validation_data=(Xvl, yvl), epochs=TRAIN["epochs"],
                     batch_size=TRAIN["batch_size"], callbacks=[es], verbose=0)
    train_time = time.time() - t0

    val_s = recon_errors(model, data["Xval"], kind)
    test_s = recon_errors(model, data["Xte"], kind)
    per, auc, auc_pr, subtype = evaluate_scores(val_s, test_s, data["yte"], data["lte"])

    extra = {}
    if kind == "snn":
        rates, per_neuron = spike_analysis(model, data["Xte"], data["yte"], data["lte"])
        energy = snn_energy(model, c, enc, dec, use_bn, rates)
        if per_neuron:
            atomic_savez(jdir / "spikes.npz", **per_neuron)
        if job_id == "SNN_AE":
            nl = EVAL["latent_n_per_class"]
            ids = np.concatenate([np.flatnonzero(data["yte"] == 0)[:nl],
                                  np.flatnonzero(data["yte"] == 1)[:nl]])
            enc_model = Model(model.input, model.get_layer("encoder").output)
            lat = enc_model.predict(data["Xte"][ids], batch_size=PB, verbose=0)
            del enc_model
            atomic_savez(jdir / "latent.npz",
                         spike_counts=lat.sum(axis=1).astype(np.uint8),
                         y=data["yte"][ids], label_type=data["lte"][ids].astype(str),
                         test_positions=ids, sample_ids=data["te_ids"][ids])
            if EVAL["brian2"]["enabled"]:
                try:
                    extra["brian2_summary"] = brian2_simulation(
                        model, data["Xte"], c, jdir / "brian2")
                except Exception as e:
                    extra["brian2_summary"] = {"error": repr(e)}
    elif kind == "lstm":
        energy = {"total_pj": lstm_energy(din)}
    else:
        energy = {"total_pj": cnn_energy(din)}

    latency = measure_latency(
        lambda Xb: model.predict(reshape_for_model(Xb, kind), verbose=0), data["Xte"])
    atomic_keras_save(jdir / "model.keras", model)
    atomic_savez(jdir / "errors.npz", val_errors=val_s.astype(np.float32),
                 test_errors=test_s.astype(np.float32))

    h = hist.history
    result = {
        "job_id": job_id, "model": model_id, "study": study, "kind": kind,
        "T": int(c["time_steps"]) if kind == "snn" else None,
        "enc_spiking": enc, "dec_spiking": dec, "use_bn": use_bn,
        "lr_used": lr,
        "snn_threshold": c.get("threshold"), "snn_decay": c.get("decay"),
        "snn_surrogate_scale": c.get("surrogate_scale"),
        "per_pct": per, "auc": auc, "auc_pr": auc_pr, "subtype_recall": subtype,
        "energy": energy, "latency_archival": latency,
        "params": int(model.count_params()), "train_time_sec": train_time,
        "best_epoch": int(np.argmin(h["val_loss"]) + 1),
        "final_val_loss": float(np.min(h["val_loss"])),
        "history": {"loss": h["loss"], "val_loss": h["val_loss"]},
        "reconstruction_error_summary": score_summary(val_s, test_s, data["yte"]),
        **extra,
    }
    del model, hist
    return result
# ======== END: JOB AUTOENCODER ========


# ======== START: JOB BASELINE ========
def run_baseline_job(name, data, seed, jdir):
    p = BEST[f"{DATASET}|{name}"]["best_params"]
    n_sub = TUNE["baseline_train_subsample"].get(name)
    Xfit = data["Xtr"]
    if n_sub and n_sub < len(Xfit):
        idx = np.sort(np.random.RandomState(seed).choice(len(Xfit), n_sub, replace=False))
        Xfit = Xfit[idx]

    if name == "IF":
        model = IsolationForest(n_estimators=p["n_estimators"], max_samples=p["max_samples"],
                                random_state=seed, n_jobs=ARGS.threads_intra)
    elif name == "OCSVM":
        model = OneClassSVM(kernel="rbf", nu=p["nu"], gamma=p["gamma"], cache_size=1000)
    elif name == "LOF":
        model = LocalOutlierFactor(n_neighbors=p["n_neighbors"], novelty=True,
                                   n_jobs=ARGS.threads_intra)
    else:
        raise ValueError(name)

    t0 = time.time()
    model.fit(Xfit)
    train_time = time.time() - t0
    val_s = -model.score_samples(data["Xval"])
    test_s = -model.score_samples(data["Xte"])
    per, auc, auc_pr, subtype = evaluate_scores(val_s, test_s, data["yte"], data["lte"])

    cost = {"n_train": int(len(Xfit)), "n_features": int(Xfit.shape[1])}
    if name == "OCSVM":
        cost["n_support_vectors"] = int(model.support_vectors_.shape[0])
    elif name == "LOF":
        cost["n_neighbors"] = int(p["n_neighbors"])
    elif name == "IF":
        cost["n_estimators"] = int(p["n_estimators"])
        cost["max_samples"] = int(model.max_samples_)
        cost["mean_tree_depth"] = float(np.mean([e.get_depth() for e in model.estimators_]))

    latency = measure_latency(lambda Xb: model.score_samples(Xb), data["Xte"])
    atomic_joblib(jdir / "model.pkl", model)
    atomic_savez(jdir / "errors.npz", val_errors=val_s.astype(np.float32),
                 test_errors=test_s.astype(np.float32))
    result = {"job_id": name, "model": name, "study": "baseline", "kind": "baseline",
              "params": p, "per_pct": per, "auc": auc, "auc_pr": auc_pr,
              "subtype_recall": subtype, "cost": cost, "latency_archival": latency,
              "train_time_sec": train_time,
              "reconstruction_error_summary": score_summary(val_s, test_s, data["yte"])}
    del model
    return result
# ======== END: JOB BASELINE ========


# ======== START: UNIT (SEED, FOLD) ========
def make_scaler():
    stype = cfg["data"]["scaler_type"][DATASET]
    if stype == "quantile":
        q = TRAIN["quantile_scaler"]
        return QuantileTransformer(n_quantiles=q["n_quantiles"],
                                   output_distribution=q["output_distribution"],
                                   random_state=RS, subsample=q["subsample"])
    return MinMaxScaler()


def run_unit(seed_idx, fold):
    key = f"s{seed_idx}_f{fold}"
    udir = OUT / key
    udir.mkdir(parents=True, exist_ok=True)
    tr, va, te = SPLITS[f"{key}_train"], SPLITS[f"{key}_val"], SPLITS[f"{key}_test"]

    # Assert anti-kebocoran per fold.
    if (Y_ALL[tr] != 0).any() or (Y_ALL[va] != 0).any():
        raise RuntimeError(f"KEBOCORAN {key}: train/val memuat serangan")
    for a, b, n in [(tr, va, "train-val"), (tr, te, "train-test"), (va, te, "val-test")]:
        if len(np.intersect1d(a, b)):
            raise RuntimeError(f"KEBOCORAN {key}: irisan {n}")
    if np.isin(np.concatenate([tr, va, te]), HOLDOUT).any():
        raise RuntimeError(f"KEBOCORAN {key}: memuat sampel holdout")

    scaler_path = udir / "scaler.pkl"
    scaler = make_scaler()
    Xtr = scaler.fit_transform(X_ALL[tr]).astype(np.float32)   # fit train-normal saja
    if not scaler_path.exists():
        atomic_joblib(scaler_path, scaler)
        atomic_savez(udir / "indices.npz", train=tr, val=va, test=te)
    data = {"Xtr": Xtr,
            "Xval": scaler.transform(X_ALL[va]).astype(np.float32),
            "Xte": scaler.transform(X_ALL[te]).astype(np.float32),
            "yte": Y_ALL[te], "lte": LT_ALL[te], "te_ids": te}
    seed = RS + 1000 * seed_idx + fold

    jobs = [("ae", j) for j in AE_JOBS] + [("bl", b) for b in BASELINE_JOBS]
    for kind, job in jobs:
        job_id = job[0] if kind == "ae" else job
        jdir = udir / job_id
        if (jdir / "result.json").exists():
            continue
        jdir.mkdir(parents=True, exist_ok=True)
        guard_memory(f"{key}/{job_id}")
        t0 = time.time()
        try:
            r = (run_ae_job(job, data, seed, jdir) if kind == "ae"
                 else run_baseline_job(job, data, seed, jdir))
        finally:
            tf.keras.backend.clear_session()
            gc.collect()
        r.update({"dataset": DATASET, "seed_idx": seed_idx, "fold": fold,
                  "n_train_normal": int(len(tr)), "n_val_normal": int(len(va)),
                  "n_test": int(len(te)), "n_test_normal": int((Y_ALL[te] == 0).sum()),
                  "n_test_attack": int((Y_ALL[te] == 1).sum()),
                  "wall_seconds": time.time() - t0, "finished_at": now_str()})
        atomic_write_json(jdir / "result.json", r)        # ditulis terakhir
        op = r["per_pct"][str(EVAL["operating_percentile"])]
        e = r.get("energy", {}).get("total_pj")
        log(f"{key} {job_id:<12} F1={op['f1']:.4f} P={op['precision']:.4f} "
            f"R={op['recall']:.4f} FPR={op['fpr_test']:.4f} AUC={r['auc']:.4f} "
            + (f"E={e:,.0f}pJ " if e else "")
            + f"| {r['wall_seconds']:.0f} s | Mem={mem_available_gb():.1f} GB")
    del data, Xtr, scaler
    gc.collect()
# ======== END: UNIT (SEED, FOLD) ========


# ======== START: EKSEKUSI WORKER ========
units = ([(0, 0)] if VALIDATION else
         [(s, f) for s in range(len(SPLIT["seeds"])) for f in range(SPLIT["n_folds"])])
mine = units[ARGS.worker::ARGS.n_workers]
log(f"S3 mulai | unit worker ini: {len(mine)} dari {len(units)}")

t_start = time.time()
for i, (s, f) in enumerate(mine, 1):
    tu = time.time()
    run_unit(s, f)
    done_t = time.time() - t_start
    log(f"Unit s{s}_f{f} selesai ({time.time() - tu:.0f} s) | {i}/{len(mine)} | "
        f"estimasi sisa {done_t / i * (len(mine) - i) / 3600:.1f} jam")

atomic_write_json(OUT / f"worker{ARGS.worker}_of{ARGS.n_workers}_done.json", {
    "units": [f"s{s}_f{f}" for s, f in mine], "manifest": MANIFEST,
    "finished_at": now_str(), "elapsed_hours": (time.time() - t_start) / 3600})
log(f"S3 worker selesai | {(time.time() - t_start) / 3600:.2f} jam | output: {OUT}")
# ======== END: EKSEKUSI WORKER ========
