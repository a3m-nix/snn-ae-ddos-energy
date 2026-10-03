# ======== START: HEADER ========
"""
S6 — ENERGI TIGA TINGKAT (TANPA TRAINING)
Revisi Paper 1 SNN-AE — Scientific Reports

Dua tahap (jalankan terpisah; tidak ada penulisan file bersama yang paralel):
  --stage measure   --dataset <ds>   : ukur dari model S3 tersimpan pada SELURUH
                                       sampel test tiap fold:
                                       k (spike/neuron/sampel) per lapisan, laju per
                                       time step, per neuron, per kelas; sparsitas
                                       ReLU lapisan Dense. Checkpoint per unit.
  --stage aggregate                  : energi T1/T2/T3 (Em grid), baseline,
                                       titik impas, uji Wilcoxon, tabel.

Model energi per sampel (BN dilipat ke bobot untuk semua model):
  Lapisan dengan input real -> MAC; input spike -> AC per spike (SOP).
  Input konstan antar-time step dihitung sekali.
  T1: konvensi SOP.  T2: T1 + zero-skipping ReLU pada model Dense.
  T3: T2 + akses memori: baca bobot (per spike untuk SNN), trafik aktivasi
      (32-bit untuk Dense, 1-bit untuk spike), state membran (opt=register,
      pess=SRAM).  Em = energi per akses kata 32-bit.
  v1 (arsip S3, rumus v8.3) dilaporkan berdampingan.
Masukan audit yang ditangani: tanpa sampling "n pertama" (seluruh test),
verifikasi provenance & tipe scaler, skor finite, aktivitas per time step,
hitungan ganda SNN_Dense diperbaiki, tidak menulis config bersama.
"""
# ======== END: HEADER ========


# ======== START: ARGUMEN DAN ENVIRONMENT ========
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--stage", required=True, choices=["measure", "aggregate"])
parser.add_argument("--dataset", choices=["edge_iiotset", "ciciot2023"])
parser.add_argument("--mode", default="full", choices=["validation", "full"])
parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--bootstrap", type=int, default=10_000)
ARGS, _ = parser.parse_known_args()
if ARGS.stage == "measure" and not ARGS.dataset:
    raise SystemExit("--dataset wajib untuk --stage measure")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = str(ARGS.threads)
# ======== END: ARGUMEN DAN ENVIRONMENT ========


# ======== START: IMPORT ========
import gc
import hashlib
import json
import platform
import time
import warnings
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

warnings.filterwarnings("ignore")
# ======== END: IMPORT ========


# ======== START: KONFIGURASI ========
WORK = Path("/home/jupyteruser/nix-ext-backup/0-eksperimen-final/0-snn-ae-revision")
CFG = json.loads((WORK / "00-config" / "config_revision.json").read_text())
OUT = WORK / "07-energy"
MEAS = OUT / ("measure_validation" if ARGS.mode == "validation" else "measure")
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = CFG["data"]["features"]
SEEDS = list(range(len(CFG["split"]["seeds"])))
N_FOLDS = CFG["split"]["n_folds"]
E_MAC = CFG["energy"]["E_MAC"]          # 4.6 pJ (Horowitz 45 nm, FP32 MAC)
E_AC = CFG["energy"]["E_AC"]            # 0.9 pJ (FP32 add)
E_LIF = CFG["energy"]["E_LIF"]          # 0.1 pJ
E_CMP = E_AC                            # perbandingan IF ~ satu add FP32
EM_GRID = [2.5, 5.0, 10.0, 20.0]        # pJ per akses kata 32-bit
EM_MAIN = 5.0
SPIKE_JOBS = ["SNN_AE", "SNN_AE_T2", "SNN_AE_T5", "SNN_AE_T7", "SNN_AE_noBN",
              "SNN_Dense", "Dense_SNN"]
MEASURE_JOBS = SPIKE_JOBS + ["Dense_AE"]
ALL_AE = [j[0] for j in CFG["evaluation"]["ae_jobs"]]
BASELINES = CFG["evaluation"]["baseline_jobs"]
DS_LABEL = {"edge_iiotset": "Edge-IIoTset", "ciciot2023": "CICIoT2023"}
PB = 8192


def log(m):
    tag = ARGS.dataset or "all"
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [S6|{ARGS.stage}|{tag}] {m}", flush=True)


def array_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).view(np.uint8))
    return h.hexdigest()[:16]


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=float))
    tmp.replace(path)
# ======== END: KONFIGURASI ========


# ======== START: PROVENANCE ========
def check_provenance(ds):
    """Hash S1 dihitung ulang; manifest S3 harus merujuk hash yang sama."""
    s1 = WORK / "02-splits" / "full" / ds
    done = json.loads((s1 / "done.json").read_text())
    t = pq.read_table(s1 / "sample.parquet", columns=["row_idx", "y"])
    if array_hash(t.column("row_idx").to_numpy(), t.column("y").to_numpy()) != done["sample_hash"]:
        raise RuntimeError(f"{ds}: sample_hash tidak cocok dengan S1")
    sp = dict(np.load(s1 / "splits.npz"))
    if array_hash(*[sp[k] for k in sorted(sp)]) != done["splits_hash"]:
        raise RuntimeError(f"{ds}: splits_hash tidak cocok dengan S1")
    man = json.loads((WORK / "04-final-eval" / "full" / ds / "run_manifest.json").read_text())
    if man["s1_sample_hash"] != done["sample_hash"] or man["s1_splits_hash"] != done["splits_hash"]:
        raise RuntimeError(f"{ds}: manifest S3 tidak merujuk S1 yang sama")
    if len(FEATURES) != 33 or "frame.time_epoch" in FEATURES:
        raise RuntimeError("Daftar fitur config tidak valid")
    if CFG["data"]["scaler_type"] != {"edge_iiotset": "quantile", "ciciot2023": "minmax"}:
        raise RuntimeError("Pemetaan scaler config tidak valid")
    return sp
# ======== END: PROVENANCE ========


# ======== START: STAGE MEASURE ========
def run_measure(ds):
    import joblib
    import tensorflow as tf
    from tensorflow.keras.layers import Layer
    from tensorflow.keras.models import Model
    from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

    tf.config.threading.set_intra_op_parallelism_threads(ARGS.threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)

    # ---- SpikingLayer identik S3 (nama registrasi sama) ----
    @tf.keras.utils.register_keras_serializable()
    class SpikingLayer(Layer):
        def __init__(self, units, threshold=0.5, decay=0.7, surrogate_scale=10.0,
                     time_steps=3, return_sequences=False, **kw):
            super().__init__(**kw)
            self.units = int(units); self.threshold = float(threshold)
            self.decay = float(decay); self.surrogate_scale = float(surrogate_scale)
            self.time_steps = int(time_steps); self.return_sequences = bool(return_sequences)

        def build(self, s):
            self.input_dim = int(s[-1])
            self.w = self.add_weight(shape=(self.input_dim, self.units),
                                     initializer=tf.keras.initializers.HeNormal(),
                                     trainable=True, name="weights")
            self.b = self.add_weight(shape=(self.units,), initializer="zeros",
                                     trainable=True, name="bias")
            super().build(s)

        @tf.custom_gradient
        def spike_function(self, m, th):
            s = tf.cast(tf.greater(m, th), tf.float32)

            def grad(dy):
                return dy * self.surrogate_scale / tf.square(
                    self.surrogate_scale * tf.abs(m - th) + 1.0), None
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

    sp = check_provenance(ds)
    s1 = WORK / "02-splits" / "full" / ds
    s3 = WORK / "04-final-eval" / "full" / ds
    t = pq.read_table(s1 / "sample.parquet", columns=["y"] + FEATURES)
    Y = t.column("y").to_numpy().astype(int)
    X = np.column_stack([t.column(c).to_numpy(zero_copy_only=False) for c in FEATURES]).astype(np.float64)
    del t
    X[~np.isfinite(X)] = 0.0
    X = X.astype(np.float32)
    expected_scaler = {"quantile": QuantileTransformer, "minmax": MinMaxScaler}[CFG["data"]["scaler_type"][ds]]

    mdir = MEAS / ds
    mdir.mkdir(parents=True, exist_ok=True)
    units = [(0, 0)] if ARGS.mode == "validation" else [(s, f) for s in SEEDS for f in range(N_FOLDS)]
    t_start = time.time()

    for i, (s, f) in enumerate(units, 1):
        key = f"s{s}_f{f}"
        outp = mdir / f"{key}.json"
        if outp.exists():
            continue
        te = sp[f"{key}_test"]
        ind = np.load(s3 / key / "indices.npz")
        if not np.array_equal(ind["test"], te):
            raise RuntimeError(f"{key}: indices test berbeda dari S1")
        scaler = joblib.load(s3 / key / "scaler.pkl")
        if not isinstance(scaler, expected_scaler):
            raise RuntimeError(f"{key}: tipe scaler {type(scaler).__name__} tidak sesuai config")
        Xte = scaler.transform(X[te]).astype(np.float32)
        yte = Y[te]
        masks = {"mixed": np.ones(len(yte), bool), "normal": yte == 0, "attack": yte == 1}
        unit_res = {"dataset": ds, "seed": s, "fold": f, "n_test": int(len(te)),
                    "n_normal": int(masks["normal"].sum()), "n_attack": int(masks["attack"].sum()),
                    "jobs": {}}

        for job in MEASURE_JOBS:
            path = s3 / key / job / "model.keras"
            model = tf.keras.models.load_model(path, compile=False,
                                               custom_objects={"SpikingLayer": SpikingLayer})
            din = int(model.input_shape[-1]); dout = int(model.output_shape[-1])
            if din != 33 or dout != 33:
                raise RuntimeError(f"{key}/{job}: dimensi model {din}->{dout}")
            outs, meta = [], []
            for lname in ["encoder", "decoder"]:
                layer = model.get_layer(lname)
                if isinstance(layer, SpikingLayer):
                    cfg = layer.get_config()
                    cfg.update({"return_sequences": True, "name": f"{lname}_seq"})
                    clone = SpikingLayer.from_config(cfg)
                    outs.append(clone(layer.input))
                    clone.set_weights(layer.get_weights())
                    meta.append((lname, "spike", int(layer.units), int(layer.time_steps)))
                else:
                    outs.append(layer.output)
                    meta.append((lname, "relu", int(layer.units), None))
            probe = Model(model.input, outs)
            preds = probe.predict(Xte, batch_size=PB, verbose=0)

            # Verifikasi: klon return_sequences harus mereproduksi output lapisan asli.
            spk_names = [m[0] for m in meta if m[1] == "spike"]
            if spk_names:
                orig = Model(model.input, [model.get_layer(n).output for n in spk_names])
                o = orig.predict(Xte[:2048], batch_size=PB, verbose=0)
                o = o if isinstance(o, list) else [o]
                for n, ov in zip(spk_names, o):
                    cv = preds[[m[0] for m in meta].index(n)][:2048]
                    cv = cv if ov.ndim == 3 else cv.mean(axis=1)
                    if not np.allclose(cv, ov, atol=1e-6):
                        raise RuntimeError(f"{key}/{job}/{n}: klon tidak identik dengan lapisan asli")
                del orig
            jr = {"din": din, "dout": dout, "params": int(model.count_params()), "layers": {}}
            for (lname, kind, n_units, T), a in zip(meta, preds):
                if not np.isfinite(a).all():
                    raise RuntimeError(f"{key}/{job}/{lname}: output non-finite")
                if kind == "spike":
                    # a: (B, T, units) biner
                    rec = {"type": "spike", "units": n_units, "T": T, "k": {}, "rate_t": {},
                           "per_neuron_k": {}}
                    for cls, m in masks.items():
                        sub = a[m]
                        rec["k"][cls] = float(sub.sum(axis=1).mean())           # spike/neuron/sampel
                        rec["rate_t"][cls] = sub.mean(axis=(0, 2)).tolist()      # laju per time step
                        rec["per_neuron_k"][cls] = sub.sum(axis=1).mean(axis=0).tolist()
                else:
                    rec = {"type": "relu", "units": n_units, "zero_frac": {}}
                    for cls, m in masks.items():
                        rec["zero_frac"][cls] = float((a[m] == 0).mean())
                jr["layers"][lname] = rec
            unit_res["jobs"][job] = jr
            del model, probe, preds
            tf.keras.backend.clear_session()
            gc.collect()

        atomic_json(outp, unit_res)
        el = time.time() - t_start
        sa = unit_res["jobs"]["SNN_AE"]["layers"]
        log(f"{key} | SNN_AE k_enc={sa['encoder']['k']['mixed']:.3f} "
            f"k_dec={sa['decoder']['k']['mixed']:.3f} | Dense_AE s_enc="
            f"{unit_res['jobs']['Dense_AE']['layers']['encoder']['zero_frac']['mixed']:.3f} | "
            f"{i}/{len(units)} | sisa ~{el / i * (len(units) - i) / 60:.1f} menit")
    log(f"Selesai measure | {(time.time() - t_start) / 60:.1f} menit | {mdir}")
# ======== END: STAGE MEASURE ========


# ======== START: MODEL ENERGI ========
def energy_spiking(job, L, din, dout, T, k_enc, k_dec, s_enc, s_dec, em):
    """
    Energi per sampel varian SNN. k_* = spike/neuron/sampel; s_* = fraksi nol ReLU.
    Mengembalikan dict T1, T2, T3_opt, T3_pess, beserta komponen.
    """
    wi, wint, wo = din * L, L * L, L * dout
    if job == "SNN_Dense":                     # enc spiking -> decoder Dense ReLU -> output
        n_lif = L * T
        t1 = wi * E_MAC + k_enc * wint * E_AC + wo * E_MAC + n_lif * E_LIF
        t2 = wi * E_MAC + k_enc * wint * E_AC + (1 - s_dec) * wo * E_MAC + n_lif * E_LIF
        reads = wi + k_enc * wint + (1 - s_dec) * wo
        act = din + dout + 2 * L * T / 32 + 2 * L        # input, output, spike enc, act dec
        state = 2 * L * T
    elif job == "Dense_SNN":                   # encoder Dense ReLU -> decoder spiking
        n_lif = L * T
        t1 = wi * E_MAC + wint * E_MAC + k_dec * wo * E_AC + n_lif * E_LIF
        t2 = wi * E_MAC + (1 - s_enc) * wint * E_MAC + k_dec * wo * E_AC + n_lif * E_LIF
        reads = wi + (1 - s_enc) * wint + k_dec * wo
        act = din + dout + 2 * L + 2 * L * T / 32
        state = 2 * L * T
    else:                                      # SNN_AE, T2/T5/T7, noBN
        n_lif = 2 * L * T
        t1 = wi * E_MAC + k_enc * wint * E_AC + k_dec * wo * E_AC + n_lif * E_LIF
        t2 = t1
        reads = wi + k_enc * wint + k_dec * wo
        act = din + dout + 2 * 2 * L * T / 32
        state = 2 * 2 * L * T
    mem_opt = em * (reads + act)
    return {"T1": t1, "T2": t2, "T3_opt": t2 + mem_opt, "T3_pess": t2 + mem_opt + em * state,
            "weight_reads": reads}


def energy_dense(L, din, dout, s_enc, s_dec, em):
    wi, wint, wo = din * L, L * L, L * dout
    t1 = (wi + wint + wo) * E_MAC
    t2 = wi * E_MAC + (1 - s_enc) * wint * E_MAC + (1 - s_dec) * wo * E_MAC
    reads = wi + (1 - s_enc) * wint + (1 - s_dec) * wo
    act = din + dout + 2 * 2 * L
    t3 = t2 + em * (reads + act)
    return {"T1": t1, "T2": t2, "T3_opt": t3, "T3_pess": t3, "weight_reads": reads}


def energy_large(e_v1, params, din, dout, em):
    """LSTM/CNN: aritmetika dari v1 (tanpa BN); bobot dibaca sekali (reuse ideal)."""
    t3 = e_v1 + em * (params + din + dout)
    return {"T1": e_v1, "T2": e_v1, "T3_opt": t3, "T3_pess": t3, "weight_reads": params}


def energy_baseline(name, cost, din, em):
    """Batas bawah, setara MAC; exp OCSVM dihitung terpisah."""
    if name == "OCSVM":
        n = cost["n_support_vectors"]
        ar = n * (din + 1) * E_MAC
        reads = n * (din + 1)
        extra = {"n_exp": n}
    elif name == "LOF":
        n = cost["n_train"]
        ar = n * din * E_MAC
        reads = n * din
        extra = {"note": "jarak brute-force; pengurutan k-NN tidak dihitung"}
    else:  # IF
        cmp_ = cost["n_estimators"] * cost["mean_tree_depth"]
        ar = cmp_ * E_CMP
        reads = 2 * cmp_
        extra = {"n_compare": cmp_}
    t3 = ar + em * (reads + din)
    return {"T1": ar, "T2": ar, "T3_opt": t3, "T3_pess": t3, "weight_reads": reads, **extra}
# ======== END: MODEL ENERGI ========


# ======== START: STAGE AGGREGATE ========
def bootstrap_ci(x, rng, n):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan, np.nan
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def holm(p):
    p = np.asarray(p, float); out = np.full_like(p, np.nan)
    order = np.argsort(p); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - r) * p[i])); out[i] = run
    return out


def run_aggregate():
    from scipy.stats import wilcoxon
    rng = np.random.default_rng(42)
    rows, meas_rows, tstep_rows, neuron = [], [], [], {}

    for ds in ["edge_iiotset", "ciciot2023"]:
        mdir = MEAS / ds
        files = sorted(mdir.glob("s*_f*.json"))
        if len(files) != len(SEEDS) * N_FOLDS:
            log(f"{ds}: measure belum lengkap ({len(files)} unit) -> dilewati")
            continue
        check_provenance(ds)
        s3 = WORK / "04-final-eval" / "full" / ds
        for fp in files:
            u = json.loads(fp.read_text())
            s, f = u["seed"], u["fold"]
            key = f"s{s}_f{f}"
            dense = u["jobs"]["Dense_AE"]["layers"]
            s_enc_d, s_dec_d = dense["encoder"]["zero_frac"]["mixed"], dense["decoder"]["zero_frac"]["mixed"]
            for job in ALL_AE + BASELINES:
                r = json.loads((s3 / key / job / "result.json").read_text())
                op = r["per_pct"][str(CFG["evaluation"]["operating_percentile"])]
                base = {"dataset": DS_LABEL[ds], "job": job, "seed": s, "fold": f,
                        "f1": op["f1"], "auc": r["auc"], "ap": r["auc_pr"],
                        "energy_v1_pj": (r.get("energy") or {}).get("total_pj", np.nan)}
                if job in MEASURE_JOBS:
                    jl = u["jobs"][job]["layers"]
                    L, din, dout = jl["encoder"]["units"], u["jobs"][job]["din"], u["jobs"][job]["dout"]
                    get_k = lambda n: jl[n]["k"]["mixed"] if jl[n]["type"] == "spike" else 0.0
                    get_s = lambda n: jl[n]["zero_frac"]["mixed"] if jl[n]["type"] == "relu" else 0.0
                    k_enc, k_dec = get_k("encoder"), get_k("decoder")
                    s_enc, s_dec = get_s("encoder"), get_s("decoder")
                    T = jl["encoder"].get("T") or jl["decoder"].get("T")
                    meas_rows.append({**base, "T": T, "k_enc": k_enc, "k_dec": k_dec,
                                      "s_enc": s_enc, "s_dec": s_dec,
                                      **{f"k_enc_{c}": jl["encoder"]["k"][c] for c in ["normal", "attack"]
                                         if jl["encoder"]["type"] == "spike"},
                                      **{f"k_dec_{c}": jl["decoder"]["k"][c] for c in ["normal", "attack"]
                                         if jl["decoder"]["type"] == "spike"}})
                    for lname in ["encoder", "decoder"]:
                        if jl[lname]["type"] == "spike":
                            for c in ["normal", "attack"]:
                                for t, v in enumerate(jl[lname]["rate_t"][c]):
                                    tstep_rows.append({"dataset": DS_LABEL[ds], "job": job, "seed": s,
                                                       "fold": f, "layer": lname, "class": c,
                                                       "t": t + 1, "rate": v})
                                neuron.setdefault(f"{ds}|{job}|{lname}|{c}", []).append(
                                    jl[lname]["per_neuron_k"][c])
                for em in EM_GRID:
                    if job in SPIKE_JOBS:
                        e = energy_spiking(job, L, din, dout, T, k_enc, k_dec, s_enc, s_dec, em)
                    elif job == "Dense_AE":
                        e = energy_dense(L, din, dout, s_enc_d, s_dec_d, em)
                    elif job in ("LSTM_AE", "CNN_AE"):
                        e = energy_large(base["energy_v1_pj"], r["params"], 33, 33, em)
                    else:
                        e = energy_baseline(job, r["cost"], 33, em)
                    rows.append({**base, "Em": em, **{k: v for k, v in e.items()
                                                       if k in ("T1", "T2", "T3_opt", "T3_pess", "weight_reads")}})

    if not rows:
        raise SystemExit("Tidak ada dataset dengan measure lengkap.")
    E = pd.DataFrame(rows)
    M = pd.DataFrame(meas_rows)
    E.to_csv(OUT / "energy_folds.csv", index=False)
    M.to_csv(OUT / "spike_sparsity_folds.csv", index=False)
    pd.DataFrame(tstep_rows).to_csv(OUT / "spike_rate_per_timestep_folds.csv", index=False)
    np.savez_compressed(OUT / "per_neuron_k.npz", **{k: np.asarray(v) for k, v in neuron.items()})

    # ---- Ringkasan (Em utama) ----
    tiers = ["T1", "T2", "T3_opt", "T3_pess"]
    summ = []
    for (dsl, job), x in E[E.Em == EM_MAIN].groupby(["dataset", "job"], sort=False):
        d = E[(E.Em == EM_MAIN) & (E.dataset == dsl) & (E.job == "Dense_AE")].sort_values(["seed", "fold"])
        x = x.sort_values(["seed", "fold"])
        row = {"dataset": dsl, "model": job, "f1": x.f1.mean(), "auc": x.auc.mean(), "ap": x.ap.mean(),
               "energy_v1_pj": x.energy_v1_pj.mean()}
        for t in tiers:
            lo, hi = bootstrap_ci(x[t], rng, ARGS.bootstrap)
            row[f"{t}_mean"] = x[t].mean(); row[f"{t}_sd"] = x[t].std(ddof=1)
            row[f"{t}_ci95"] = f"[{lo:,.0f}, {hi:,.0f}]"
            row[f"{t}_dense_over_model"] = d[t].mean() / x[t].mean()
        summ.append(row)
    S = pd.DataFrame(summ)
    S.to_csv(OUT / "energy_summary_Em5.csv", index=False)

    # ---- Uji Wilcoxon SNN_AE vs Dense_AE per tingkat & Em ----
    tests = []
    for dsl in E.dataset.unique():
        res = []
        for em in EM_GRID:
            for t in tiers:
                a = E[(E.dataset == dsl) & (E.job == "SNN_AE") & (E.Em == em)].sort_values(["seed", "fold"])[t].to_numpy()
                b = E[(E.dataset == dsl) & (E.job == "Dense_AE") & (E.Em == em)].sort_values(["seed", "fold"])[t].to_numpy()
                dlt = a - b
                try:
                    stat, p = wilcoxon(a, b) if not np.allclose(dlt, 0) else (np.nan, 1.0)
                except ValueError:
                    stat, p = np.nan, 1.0
                res.append({"dataset": dsl, "Em": em, "tier": t, "snn_mean": a.mean(), "dense_mean": b.mean(),
                            "saving_pct": 100 * (1 - a.mean() / b.mean()),
                            "folds_snn_lower": int((a < b).sum()), "W": stat, "p": p})
        ph = holm([r["p"] for r in res])
        for r, h in zip(res, ph):
            r["p_holm"] = h
        tests.extend(res)
    pd.DataFrame(tests).to_csv(OUT / "energy_tests_snn_vs_dense.csv", index=False)

    # ---- Titik impas k* (SNN_AE vs Dense_AE, k_enc = k_dec = k) ----
    be = []
    for dsl in M.dataset.unique():
        ms = M[(M.dataset == dsl) & (M.job == "SNN_AE")]
        md = M[(M.dataset == dsl) & (M.job == "Dense_AE")]
        T = int(ms["T"].iloc[0]); L, din, dout = 32, 33, 33
        k_eff = ((ms.k_enc * L * L + ms.k_dec * L * dout) / (L * L + L * dout)).mean()
        for em in EM_GRID:
            for t in tiers:
                dense_e = np.mean([energy_dense(L, din, dout, se, sd, em)[t] for se, sd in zip(md.s_enc, md.s_dec)])
                e0 = energy_spiking("SNN_AE", L, din, dout, T, 0, 0, 0, 0, em)[t]
                e1 = energy_spiking("SNN_AE", L, din, dout, T, 1, 1, 0, 0, em)[t]
                kstar = (dense_e - e0) / (e1 - e0)
                be.append({"dataset": dsl, "Em": em, "tier": t, "k_star": kstar,
                           "k_measured_eff": k_eff, "snn_below_breakeven": bool(k_eff < kstar)})
    pd.DataFrame(be).to_csv(OUT / "break_even.csv", index=False)

    # ---- Baseline & Pareto ----
    S[S.model.isin(BASELINES)].to_csv(OUT / "baseline_costs_Em5.csv", index=False)
    S[["dataset", "model", "f1", "auc", "ap", "T1_mean", "T2_mean", "T3_opt_mean", "T3_pess_mean"]] \
        .to_csv(OUT / "pareto_points_Em5.csv", index=False)

    # ---- Titik pilot (holdout) untuk Supplementary ----
    for ds in ["edge_iiotset", "ciciot2023"]:
        p = WORK / "pilot-spike-aware" / ds / "pilot_trials.csv"
        if p.exists():
            pd.read_csv(p).assign(dataset=DS_LABEL[ds]).to_csv(OUT / f"pilot_points_{ds}.csv", index=False)

    vers = {"python": platform.python_version()}
    for pkg in ["numpy", "pandas", "scipy", "tensorflow", "scikit-learn"]:
        try:
            vers[pkg] = metadata.version(pkg)
        except Exception:
            vers[pkg] = None
    atomic_json(OUT / "summary.json", {
        "generated_at": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "versions": vers,
        "constants_pj": {"E_MAC": E_MAC, "E_AC": E_AC, "E_LIF": E_LIF, "E_CMP": E_CMP,
                         "Em_grid": EM_GRID, "Em_main": EM_MAIN},
        "assumptions": [
            "BN dilipat ke bobot untuk semua model (biaya 0).",
            "Lapisan dengan input spike dihitung AC per spike (SOP); input real dihitung MAC.",
            "Input konstan antar-time step dihitung sekali (encoder SNN, decoder Dense_SNN).",
            "T2: zero-skipping aktivasi ReLU untuk lapisan Dense berikutnya.",
            "T3: baca bobot sekali per penggunaan (Dense) atau per spike (SNN); aktivasi 32-bit (Dense) "
            "vs 1-bit (spike); state membran: opt=register, pess=SRAM (2 akses/neuron/time step).",
            "LSTM/CNN: aritmetika dari v1, tanpa kredit sparsitas; bobot dibaca sekali.",
            "Baseline: batas bawah setara MAC; exp OCSVM dan pengurutan LOF tidak dihitung.",
            "k dan s diukur pada seluruh sampel test tiap fold (campuran normal+serangan).",
        ]})

    show = S[S.model.isin(["SNN_AE", "Dense_AE", "SNN_Dense", "Dense_SNN", "LSTM_AE", "CNN_AE",
                           "OCSVM", "IF", "LOF"])]
    print(show[["dataset", "model", "f1", "T1_mean", "T2_mean", "T3_opt_mean", "T3_pess_mean",
                "T1_dense_over_model", "T2_dense_over_model", "T3_opt_dense_over_model",
                "T3_pess_dense_over_model"]].round(3).to_string(index=False))
    tt = pd.DataFrame(tests)
    print(tt[tt.Em == EM_MAIN][["dataset", "tier", "saving_pct", "folds_snn_lower", "p_holm"]]
          .round(4).to_string(index=False))
    print(pd.DataFrame(be).query("Em == @EM_MAIN").round(3).to_string(index=False))
    log(f"Selesai aggregate | output: {OUT}")
# ======== END: STAGE AGGREGATE ========


# ======== START: EKSEKUSI ========
if ARGS.stage == "measure":
    MEAS.mkdir(parents=True, exist_ok=True)
    run_measure(ARGS.dataset)
else:
    run_aggregate()
# ======== END: EKSEKUSI ========
