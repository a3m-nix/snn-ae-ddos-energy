# ======== START: HEADER ========
"""
S5 — ANALISIS PASCA-RUN (TANPA TRAINING)
Revisi Paper 1 SNN-AE — Scientific Reports

Bagian (jalankan terpisah, satu proses per bagian):
  --part brian2  --dataset <ds> : validasi dinamika LIF encoder SNN_AE di Brian2
                                  dengan input = output BatchNormalization (inference),
                                  dibandingkan bit per bit dengan spike Keras per time step.
                                  Seed 0, 10 fold, 256 sampel/fold (128 normal + 128 serangan).
  --part cross   --source <ds>  : model dataset sumber (seed 0, 10 fold; scaler & threshold
                                  sumber) diuji pada test set dataset target (fold sama).
  --part payload --dataset <ds> : paket RPi dari seed 0 / fold 0: model, scaler, threshold,
                                  fitur, test set mentah, referensi skor server, TFLite (opsional),
                                  manifest hash.
Output: 06-posthoc/{brian2,cross,payload}/...
"""
# ======== END: HEADER ========


# ======== START: ARGUMEN DAN ENVIRONMENT ========
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--part", required=True, choices=["brian2", "cross", "payload"])
parser.add_argument("--dataset", choices=["edge_iiotset", "ciciot2023"])
parser.add_argument("--source", choices=["edge_iiotset", "ciciot2023"])
parser.add_argument("--mode", default="full", choices=["validation", "full"])
parser.add_argument("--threads", type=int, default=8)
ARGS, _ = parser.parse_known_args()
if ARGS.part in ("brian2", "payload") and not ARGS.dataset:
    raise SystemExit("--dataset wajib untuk brian2/payload")
if ARGS.part == "cross" and not ARGS.source:
    raise SystemExit("--source wajib untuk cross")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = str(ARGS.threads)
# ======== END: ARGUMEN DAN ENVIRONMENT ========


# ======== START: IMPORT ========
import gc
import hashlib
import json
import platform
import shutil
import time
import warnings
from datetime import datetime
from importlib import metadata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")
tf.config.threading.set_intra_op_parallelism_threads(ARGS.threads)
tf.config.threading.set_inter_op_parallelism_threads(2)
# ======== END: IMPORT ========


# ======== START: KONFIGURASI DAN UTILITAS ========
WORK = Path("/home/jupyteruser/nix-ext-backup/0-eksperimen-final/0-snn-ae-revision")
CFG = json.loads((WORK / "00-config" / "config_revision.json").read_text())
OUT = WORK / "06-posthoc"
FEATURES = CFG["data"]["features"]
OP = CFG["evaluation"]["operating_percentile"]
GRID = [90, 95, 96, 97, 98, 99, 99.5, 99.9, 99.99]
AE_MAIN = ["SNN_AE", "Dense_AE", "LSTM_AE", "CNN_AE"]
BASELINES = CFG["evaluation"]["baseline_jobs"]
KIND = {"SNN_AE": "snn", "Dense_AE": "snn", "LSTM_AE": "lstm", "CNN_AE": "cnn"}
FOLDS = [0] if ARGS.mode == "validation" else list(range(CFG["split"]["n_folds"]))
PB = 8192
OTHER = {"edge_iiotset": "ciciot2023", "ciciot2023": "edge_iiotset"}


def log(m):
    tag = ARGS.dataset or ARGS.source
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [S5|{ARGS.part}|{tag}] {m}", flush=True)


def array_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).view(np.uint8))
    return h.hexdigest()[:16]


def file_sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def atomic_json(path, data):
    path = Path(path); tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=float)); tmp.replace(path)


def versions():
    v = {"python": platform.python_version()}
    for pkg in ["numpy", "pandas", "scikit-learn", "tensorflow", "joblib", "pyarrow", "brian2"]:
        try:
            v[pkg] = metadata.version(pkg)
        except Exception:
            v[pkg] = None
    return v


def check_provenance(ds):
    s1 = WORK / "02-splits" / "full" / ds
    done = json.loads((s1 / "done.json").read_text())
    t = pq.read_table(s1 / "sample.parquet", columns=["row_idx", "y"])
    if array_hash(t.column("row_idx").to_numpy(), t.column("y").to_numpy()) != done["sample_hash"]:
        raise RuntimeError(f"{ds}: sample_hash tidak cocok")
    sp = dict(np.load(s1 / "splits.npz"))
    if array_hash(*[sp[k] for k in sorted(sp)]) != done["splits_hash"]:
        raise RuntimeError(f"{ds}: splits_hash tidak cocok")
    man = json.loads((WORK / "04-final-eval" / "full" / ds / "run_manifest.json").read_text())
    if man["s1_sample_hash"] != done["sample_hash"] or man["s1_splits_hash"] != done["splits_hash"]:
        raise RuntimeError(f"{ds}: manifest S3 tidak merujuk S1 yang sama")
    return sp, done


def load_raw(ds):
    t = pq.read_table(WORK / "02-splits" / "full" / ds / "sample.parquet",
                      columns=["sample_id", "y", "label_type"] + FEATURES)
    y = t.column("y").to_numpy().astype(int)
    lt = np.asarray(t.column("label_type").to_pylist(), dtype=object)
    X = np.column_stack([t.column(c).to_numpy(zero_copy_only=False) for c in FEATURES]).astype(np.float64)
    del t
    X[~np.isfinite(X)] = 0.0                          # identik S3
    return X, y, lt                                   # float64 mentah (untuk payload)


def prep(X):
    """Jalur input identik S3: cast float32 SEBELUM scaler (wajib; dtype mengubah hasil QuantileTransformer)."""
    return np.asarray(X, dtype=np.float32)


def unit_dir(ds, fold, seed=0):
    return WORK / "04-final-eval" / "full" / ds / f"s{seed}_f{fold}"
# ======== END: KONFIGURASI DAN UTILITAS ========


# ======== START: SPIKINGLAYER (IDENTIK S3) DAN SKOR ========
@tf.keras.utils.register_keras_serializable()
class SpikingLayer(Layer):
    def __init__(self, units, threshold=0.5, decay=0.7, surrogate_scale=10.0,
                 time_steps=3, return_sequences=False, **kw):
        super().__init__(**kw)
        self.units = int(units); self.threshold = float(threshold); self.decay = float(decay)
        self.surrogate_scale = float(surrogate_scale); self.time_steps = int(time_steps)
        self.return_sequences = bool(return_sequences)

    def build(self, s):
        self.input_dim = int(s[-1])
        self.w = self.add_weight(shape=(self.input_dim, self.units),
                                 initializer=tf.keras.initializers.HeNormal(), trainable=True, name="weights")
        self.b = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True, name="bias")
        super().build(s)

    @tf.custom_gradient
    def spike_function(self, m, th):
        s = tf.cast(tf.greater(m, th), tf.float32)

        def grad(dy):
            return dy * self.surrogate_scale / tf.square(self.surrogate_scale * tf.abs(m - th) + 1.0), None
        return s, grad

    def call(self, x, training=None):
        mem = tf.zeros((tf.shape(x)[0], self.units), dtype=x.dtype)
        out, seq = [], len(x.shape) == 3
        for t in range(self.time_steps):
            xi = x[:, t, :] if seq else x
            cur = (tf.matmul(xi, self.w) + self.b) / tf.sqrt(tf.cast(tf.shape(xi)[-1], tf.float32))
            mem = self.decay * mem + cur
            s = self.spike_function(mem, self.threshold)
            mem = mem - s * self.threshold
            out.append(s)
        a = tf.stack(out, axis=1)
        return a if self.return_sequences else tf.reduce_mean(a, axis=1)

    def get_config(self):
        c = super().get_config()
        c.update({"units": self.units, "threshold": self.threshold, "decay": self.decay,
                  "surrogate_scale": self.surrogate_scale, "time_steps": self.time_steps,
                  "return_sequences": self.return_sequences})
        return c


def load_keras(path):
    return tf.keras.models.load_model(path, compile=False, custom_objects={"SpikingLayer": SpikingLayer})


def recon_errors(model, X, kind):
    """Identik S3 (CNN: dimensi output diverifikasi = input)."""
    if kind == "lstm":
        Xr = X.reshape(-1, 1, X.shape[1])
        p = model.predict(Xr, batch_size=PB, verbose=0)
        return np.mean((Xr - p) ** 2, axis=(1, 2))
    if kind == "cnn":
        p = model.predict(X.reshape(-1, X.shape[1], 1), batch_size=PB, verbose=0).reshape(len(X), -1)
        if p.shape[1] != X.shape[1]:
            raise RuntimeError("Dimensi output CNN != input")
        return np.mean((X - p) ** 2, axis=1)
    p = model.predict(X, batch_size=PB, verbose=0)
    return np.mean((X - p) ** 2, axis=1)


def scores(job, path_dir, X):
    if job in BASELINES:
        return -joblib.load(path_dir / "model.pkl").score_samples(X)
    m = load_keras(path_dir / "model.keras")
    e = recon_errors(m, X, KIND[job])
    del m
    tf.keras.backend.clear_session()
    return e


def metrics(val_s, test_s, y):
    out = {}
    for p in GRID:
        thr = float(np.percentile(val_s, p))
        pred = test_s > thr
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
        out[str(p)] = {"f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                       "precision": prec, "recall": rec,
                       "fpr_test": fp / (fp + tn) if fp + tn else np.nan}
    try:
        auc, ap = roc_auc_score(y, test_s), average_precision_score(y, test_s)
    except ValueError:
        auc, ap = np.nan, np.nan
    return out, float(auc), float(ap)
# ======== END: SPIKINGLAYER (IDENTIK S3) DAN SKOR ========


# ======== START: BAGIAN A — BRIAN2 ========
def part_brian2(ds):
    from brian2 import prefs, start_scope, defaultclock, NeuronGroup, SpikeMonitor, ms, run
    prefs.codegen.target = "numpy"
    sp, _ = check_provenance(ds)
    X, y, _ = load_raw(ds)
    out = OUT / "brian2" / ds
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in FOLDS:
        u = unit_dir(ds, f)
        te = sp[f"s0_f{f}_test"]
        rng = np.random.RandomState(42 + f)
        pos = np.concatenate([rng.choice(np.flatnonzero(y[te] == c), 128, replace=False) for c in (0, 1)])
        ids = te[pos]
        Xs = joblib.load(u / "scaler.pkl").transform(prep(X[ids])).astype(np.float32)
        model = load_keras(u / "SNN_AE" / "model.keras")
        enc = model.get_layer("encoder")
        T, vth, beta, units = enc.time_steps, enc.threshold, enc.decay, enc.units
        bn_out = Model(model.input, model.get_layer("batch_norm").output).predict(Xs, verbose=0)
        k_spk = Model(model.input, enc.output).predict(Xs, verbose=0)          # (n, T, units)
        if k_spk.ndim != 3:
            raise RuntimeError("Encoder SNN_AE harus return_sequences=True")
        W, b = enc.get_weights()
        I = ((bn_out.astype(np.float64) @ W.astype(np.float64)) + b) / np.sqrt(W.shape[0])
        n = len(ids)

        start_scope()
        defaultclock.dt = 1 * ms
        G = NeuronGroup(n * units, """dv/dt = ((beta - 1) * v + I) / dt_sim : 1
                                     I : 1
                                     beta : 1
                                     vth : 1
                                     dt_sim : second""",
                        threshold="v > vth", reset="v -= vth", method="euler")
        G.v = 0; G.I = I.ravel(); G.beta = beta; G.vth = vth; G.dt_sim = 1 * ms
        mon = SpikeMonitor(G)
        run(T * ms, namespace={})   # cegah konflik nama dengan variabel Python
        b_spk = np.zeros((n, T, units), dtype=np.uint8)
        idx = np.asarray(mon.i); step = np.rint(np.asarray(mon.t / ms)).astype(int)
        b_spk[idx // units, step, idx % units] = 1

        agree = float((b_spk == k_spk.astype(np.uint8)).mean())
        kc, bc = k_spk.sum(axis=(1, 2)), b_spk.sum(axis=(1, 2))
        corr = float(np.corrcoef(kc, bc)[0, 1]) if kc.std() > 0 and bc.std() > 0 else np.nan
        mem_margin = np.abs(I).min()
        rows.append({"dataset": ds, "fold": f, "n_samples": n, "T": T, "units": units,
                     "bit_agreement": agree, "n_bit_mismatch": int((b_spk != k_spk).sum()),
                     "spike_count_corr": corr,
                     "k_keras": float(k_spk.sum(axis=1).mean()), "k_brian2": float(b_spk.sum(axis=1).mean()),
                     "k_keras_normal": float(k_spk[:128].sum(axis=1).mean()),
                     "k_keras_attack": float(k_spk[128:].sum(axis=1).mean()),
                     "min_abs_current": float(mem_margin)})
        np.savez_compressed(out / f"s0_f{f}.npz", sample_ids=ids, keras_spikes=k_spk.astype(np.uint8),
                            brian2_spikes=b_spk, y=y[ids])
        log(f"fold {f}: kesesuaian bit={agree:.6f} mismatch={rows[-1]['n_bit_mismatch']} "
            f"k_keras={rows[-1]['k_keras']:.3f} k_brian2={rows[-1]['k_brian2']:.3f}")
        del model
        tf.keras.backend.clear_session(); gc.collect()
    pd.DataFrame(rows).to_csv(out / "brian2_validation.csv", index=False)
    atomic_json(out / "summary.json", {"versions": versions(), "folds": FOLDS,
                                       "input": "BatchNormalization output (inference mode)",
                                       "mean_bit_agreement": float(np.mean([r["bit_agreement"] for r in rows]))})
# ======== END: BAGIAN A — BRIAN2 ========


# ======== START: BAGIAN B — CROSS-DATASET ========
def part_cross(src):
    tgt = OTHER[src]
    check_provenance(src)
    sp_t, _ = check_provenance(tgt)
    Xt, yt, ltt = load_raw(tgt)
    out = OUT / "cross" / f"{src}_to_{tgt}"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in FOLDS:
        u = unit_dir(src, f)
        scaler = joblib.load(u / "scaler.pkl")
        te = sp_t[f"s0_f{f}_test"]
        Xte = scaler.transform(prep(Xt[te])).astype(np.float32)
        y = yt[te]
        for job in AE_MAIN + BASELINES:
            val_s = np.load(u / job / "errors.npz")["val_errors"].astype(np.float64)
            test_s = scores(job, u / job, Xte).astype(np.float64)
            if not np.isfinite(test_s).all():
                raise RuntimeError(f"{job} fold {f}: skor non-finite")
            per, auc, ap = metrics(val_s, test_s, y)
            rows.append({"source": src, "target": tgt, "fold": f, "job": job, "auc": auc, "ap": ap,
                         **{f"{k}_p{p}": v for p, d in per.items() for k, v in d.items()}})
        log(f"fold {f}: " + " ".join(f"{r['job']}:F1={r[f'f1_p{OP}']:.3f}/AUC={r['auc']:.3f}"
                                     for r in rows if r["fold"] == f))
        gc.collect()
    df = pd.DataFrame(rows)
    df.to_csv(out / "cross_folds.csv", index=False)
    cols = [f"f1_p{OP}", f"recall_p{OP}", f"fpr_test_p{OP}", "auc", "ap", "f1_p95", "fpr_test_p95"]
    summ = df.groupby("job")[cols].agg(["mean", "std"]).round(4)
    summ.to_csv(out / "cross_summary.csv")
    print(summ.to_string())
# ======== END: BAGIAN B — CROSS-DATASET ========


# ======== START: BAGIAN C — PAYLOAD RPI ========
def part_payload(ds):
    sp, done = check_provenance(ds)
    X, y, lt = load_raw(ds)
    u = unit_dir(ds, 0)
    out = OUT / "payload" / ds
    if out.exists():
        raise RuntimeError(f"{out} sudah ada; hapus secara sadar bila ingin membuat ulang")
    (out / "models").mkdir(parents=True)
    te = sp["s0_f0_test"]
    ind = np.load(u / "indices.npz")
    if not np.array_equal(ind["test"], te):
        raise RuntimeError("indices test berbeda dari S1")

    shutil.copy2(u / "scaler.pkl", out / "scaler.pkl")
    scaler = joblib.load(out / "scaler.pkl")
    Xte_s = scaler.transform(prep(X[te])).astype(np.float32)
    raw = pd.DataFrame(X[te], columns=FEATURES)
    raw.insert(0, "label_type", lt[te].astype(str)); raw.insert(0, "y", y[te]); raw.insert(0, "sample_id", te)
    pq.write_table(pa.Table.from_pandas(raw, preserve_index=False), out / "test_raw.parquet")

    thresholds, ref, tflite = {}, {}, {}
    for job in AE_MAIN + BASELINES:
        src = u / job
        e = np.load(src / "errors.npz")
        val_s, test_s = e["val_errors"].astype(np.float64), e["test_errors"]
        thresholds[job] = {str(p): float(np.percentile(val_s, p)) for p in GRID}
        r = json.loads((src / "result.json").read_text())
        if job in BASELINES:
            shutil.copy2(src / "model.pkl", out / "models" / f"{job}.pkl")
        else:
            shutil.copy2(src / "model.keras", out / "models" / f"{job}.keras")
            # Verifikasi: skor ulang == errors.npz server
            m = load_keras(src / "model.keras")
            re_s = recon_errors(m, Xte_s, KIND[job])
            diff = float(np.max(np.abs(re_s - test_s)))
            if not np.allclose(re_s, test_s, rtol=1e-4, atol=1e-6):
                raise RuntimeError(f"{job}: skor ulang berbeda dari S3 (max diff {diff:.2e})")
            try:
                conv = tf.lite.TFLiteConverter.from_keras_model(m)
                conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
                blob = conv.convert()
                (out / "models" / f"{job}.tflite").write_bytes(blob)
                it = tf.lite.Interpreter(model_content=blob); it.allocate_tensors()
                inp, outd = it.get_input_details()[0], it.get_output_details()[0]
                xs = {"lstm": Xte_s[:256].reshape(-1, 1, 33), "cnn": Xte_s[:256].reshape(-1, 33, 1)}.get(KIND[job], Xte_s[:256])
                it.resize_tensor_input(inp["index"], xs.shape); it.allocate_tensors()
                it.set_tensor(inp["index"], xs.astype(np.float32)); it.invoke()
                tl = it.get_tensor(outd["index"]).reshape(256, -1)
                kr = m.predict(xs, verbose=0).reshape(256, -1)
                tflite[job] = {"ok": True, "max_abs_diff_vs_keras": float(np.max(np.abs(tl - kr)))}
            except Exception as ex:
                tflite[job] = {"ok": False, "error": repr(ex)[:300]}
            del m
            tf.keras.backend.clear_session()
        ref[job] = test_s
        log(f"{job}: F1@p{OP}={r['per_pct'][str(OP)]['f1']:.4f} | tflite={tflite.get(job, {}).get('ok', 'n/a')}")

    np.savez_compressed(out / "server_reference_scores.npz", sample_ids=te, y=y[te], **ref)
    atomic_json(out / "thresholds.json", thresholds)
    atomic_json(out / "features.json", FEATURES)
    files = {p.relative_to(out).as_posix(): file_sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
    atomic_json(out / "manifest.json", {
        "dataset": ds, "unit": "seed 0 / fold 0 (dipilih a priori)",
        "created_at": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "versions_server": versions(),
        "s1_sample_hash": done["sample_hash"], "s1_splits_hash": done["splits_hash"],
        "scaler_type": CFG["data"]["scaler_type"][ds], "operating_percentile": OP,
        "n_test": int(len(te)), "n_test_normal": int((y[te] == 0).sum()), "n_test_attack": int((y[te] == 1).sum()),
        "score_definition": {"AE": "MSE rekonstruksi pada fitur terskala",
                             "baseline": "-score_samples (lebih tinggi = lebih anomali)"},
        "input_pipeline": "33 fitur mentah (test_raw.parquet, float64) -> non-finite = 0 -> CAST float32 -> scaler.pkl -> float32 -> model (cast float32 sebelum scaler WAJIB, identik training)",
        "tflite": tflite, "sha256": files})
    log(f"Payload selesai: {out}")
# ======== END: BAGIAN C — PAYLOAD RPI ========


# ======== START: EKSEKUSI ========
t0 = time.time()
if ARGS.part == "brian2":
    part_brian2(ARGS.dataset)
elif ARGS.part == "cross":
    part_cross(ARGS.source)
else:
    part_payload(ARGS.dataset)
log(f"Selesai | {(time.time() - t0) / 60:.1f} menit")
# ======== END: EKSEKUSI ========
