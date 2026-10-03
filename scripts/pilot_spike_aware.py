# ======== START: HEADER ========
"""
PILOT — SELEKSI SPIKE-AWARE DAN REGULARISASI LAJU SPIKE (HOLDOUT SAJA)
Revisi Paper 1 SNN-AE

Pertanyaan: dapatkah SNN-AE mencapai k <= 0,5 (spike per neuron per inferensi)
dengan penurunan F1 kecil, sehingga unggul atas Dense-AE pada tiga tingkat
energi (T1 SOP, T2 sparsitas simetris, T3 memory-aware)?

Desain:
  - Data: tune_train / tune_calib / tune_dev dari S1 (identik S2). Test fold
    TIDAK disentuh.
  - Optuna multi-objektif: maksimalkan F1@p99, minimalkan energi T3 SNN.
  - Ruang pencarian: V_th [0.3, 2.0] (diperluas), decay, surrogate, lr,
    lam (penalti laju spike) {0, 1e-3, 1e-2, 5e-2, 1e-1}.
  - Trial 0 = konfigurasi S2 terbaik (lam=0) sebagai acuan.
  - Dense-AE acuan dilatih sekali (LR S2) untuk mengukur sparsitas ReLU nyata.
Output: 0-snn-ae-revision/pilot-spike-aware/<dataset>/
"""
# ======== END: HEADER ========


# ======== START: ARGUMEN DAN ENVIRONMENT ========
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, choices=["edge_iiotset", "ciciot2023"])
parser.add_argument("--n-trials", type=int, default=40)
parser.add_argument("--threads-intra", type=int, default=2)
parser.add_argument("--threads-inter", type=int, default=1)
parser.add_argument("--em", type=float, default=5.0, help="pJ per baca bobot 32-bit")
ARGS, _ = parser.parse_known_args()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = str(ARGS.threads_intra)
# ======== END: ARGUMEN DAN ENVIRONMENT ========


# ======== START: IMPORT ========
import gc
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
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from sklearn.metrics import f1_score, roc_auc_score

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
tf.config.threading.set_intra_op_parallelism_threads(ARGS.threads_intra)
tf.config.threading.set_inter_op_parallelism_threads(ARGS.threads_inter)
# ======== END: IMPORT ========


# ======== START: KONFIGURASI DAN DATA ========
DS = ARGS.dataset
WORK = Path("/home/jupyteruser/nix-ext-backup/0-eksperimen-final/0-snn-ae-revision")
CFG = json.loads((WORK / "00-config" / "config_revision.json").read_text())
S1 = WORK / "02-splits" / "full" / DS
BEST = json.loads((WORK / "03-tuning" / "full" / DS / "best_hparams.json").read_text())
OUT = WORK / "pilot-spike-aware" / DS
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = CFG["data"]["features"]
TR = CFG["training"]
RS = CFG["split"]["random_state"]
T_STEPS, LAT = 3, TR["latent_dim"]
E_MAC, E_AC, E_LIF, EM = 4.6, 0.9, 0.1, ARGS.em
LAMS = [0.0, 1e-3, 1e-2, 5e-2, 1e-1]


def log(m):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [pilot|{DS}] {m}", flush=True)


def load_data():
    sp = np.load(S1 / "splits.npz")
    itr, ical, idev = sp["tune_train"], sp["tune_calib"], sp["tune_dev"]
    t = pq.read_table(S1 / "sample.parquet", columns=["y"] + FEATURES)
    y = t.column("y").to_numpy().astype(int)
    X = np.column_stack([t.column(c).to_numpy(zero_copy_only=False) for c in FEATURES]).astype(np.float64)
    X[~np.isfinite(X)] = 0.0
    X = X.astype(np.float32)
    assert (y[itr] == 0).all() and (y[ical] == 0).all()
    if CFG["data"]["scaler_type"][DS] == "quantile":
        q = TR["quantile_scaler"]
        sc = QuantileTransformer(n_quantiles=q["n_quantiles"], output_distribution=q["output_distribution"],
                                 random_state=RS, subsample=q["subsample"])
    else:
        sc = MinMaxScaler()
    Xtr = sc.fit_transform(X[itr]).astype(np.float32)
    return Xtr, sc.transform(X[ical]).astype(np.float32), sc.transform(X[idev]).astype(np.float32), y[idev]


XTR, XCAL, XDEV, YDEV = load_data()
DIN = XTR.shape[1]
log(f"train={len(XTR):,} calib={len(XCAL):,} dev={len(XDEV):,}")
# ======== END: KONFIGURASI DAN DATA ========


# ======== START: MODEL (SpikingLayer SALINAN v8.3) ========
@tf.keras.utils.register_keras_serializable()
class SpikingLayer(Layer):
    def __init__(self, units, threshold=0.5, decay=0.7, surrogate_scale=10.0,
                 time_steps=3, return_sequences=False, **kw):
        super().__init__(**kw)
        self.units, self.threshold, self.decay = int(units), float(threshold), float(decay)
        self.surrogate_scale, self.time_steps = float(surrogate_scale), int(time_steps)
        self.return_sequences = bool(return_sequences)

    def build(self, s):
        self.w = self.add_weight(shape=(int(s[-1]), self.units),
                                 initializer=tf.keras.initializers.HeNormal(), name="weights")
        self.b = self.add_weight(shape=(self.units,), initializer="zeros", name="bias")

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


def build_snn(c, lam):
    inp = Input(shape=(DIN,))
    x = BatchNormalization()(inp)
    enc = SpikingLayer(LAT, c["threshold"], c["decay"], c["surrogate_scale"], T_STEPS, True, name="encoder")(x)
    dec = SpikingLayer(LAT, c["threshold"], c["decay"], c["surrogate_scale"], T_STEPS, False, name="decoder")(enc)
    out = Dense(DIN, activation="linear", name="output")(dec)
    m = Model(inp, out)
    if lam > 0:
        m.add_loss(lam * (tf.reduce_mean(enc) + tf.reduce_mean(dec)))
    return m


def build_dense():
    inp = Input(shape=(DIN,))
    x = BatchNormalization()(inp)
    e = Dense(LAT, activation="relu", name="encoder")(x)
    d = Dense(LAT, activation="relu", name="decoder")(e)
    return Model(inp, Dense(DIN, activation="linear", name="output")(d))


def fit_eval(model, lr):
    model.compile(optimizer=Adam(lr, clipnorm=TR["grad_clip"]), loss="mse")
    model.fit(XTR, XTR, validation_data=(XCAL, XCAL), epochs=TR["epochs"], batch_size=TR["batch_size"],
              callbacks=[EarlyStopping(patience=TR["patience"], restore_best_weights=True)], verbose=0)
    ec = np.mean((XCAL - model.predict(XCAL, batch_size=4096, verbose=0)) ** 2, axis=1)
    ed = np.mean((XDEV - model.predict(XDEV, batch_size=4096, verbose=0)) ** 2, axis=1)
    thr = np.percentile(ec, 99)
    f1 = f1_score(YDEV, (ed > thr).astype(int), zero_division=0)
    try:
        auc = roc_auc_score(YDEV, ed)
    except Exception:
        auc = float("nan")
    return float(f1), float(auc)


def layer_out(model, name):
    sub = Model(model.input, model.get_layer(name).output)
    o = sub.predict(XDEV, batch_size=4096, verbose=0)
    del sub
    return o
# ======== END: MODEL (SpikingLayer SALINAN v8.3) ========


# ======== START: ENERGI TIGA TINGKAT ========
W_IN, W_INT, W_OUT = DIN * LAT, LAT * LAT, LAT * DIN


def snn_energy(k_enc, k_dec):
    t1 = W_IN * E_MAC + k_enc * W_INT * E_AC + k_dec * W_OUT * E_AC + 2 * LAT * T_STEPS * E_LIF
    reads = W_IN + k_enc * W_INT + k_dec * W_OUT
    return {"T1": t1, "T3_opt": t1 + EM * reads, "T3_pess": t1 + EM * reads + EM * 4 * LAT * T_STEPS}


def dense_energy(s_enc, s_dec):
    t1 = (W_IN + W_INT + W_OUT) * E_MAC
    t2 = W_IN * E_MAC + (1 - s_enc) * W_INT * E_MAC + (1 - s_dec) * W_OUT * E_MAC
    t3 = t2 + EM * (W_IN + (1 - s_enc) * W_INT + (1 - s_dec) * W_OUT)
    return {"T1": t1, "T2": t2, "T3": t3}
# ======== END: ENERGI TIGA TINGKAT ========


# ======== START: ACUAN DENSE-AE ========
ref_path = OUT / "dense_reference.json"
if ref_path.exists():
    REF = json.loads(ref_path.read_text())
else:
    random.seed(RS); np.random.seed(RS); tf.random.set_seed(RS)
    dm = build_dense()
    f1d, aucd = fit_eval(dm, BEST[f"{DS}|Dense_AE"]["best_params"]["learning_rate"])
    s_enc = float((layer_out(dm, "encoder") == 0).mean())
    s_dec = float((layer_out(dm, "decoder") == 0).mean())
    REF = {"f1": f1d, "auc": aucd, "relu_sparsity_enc": s_enc, "relu_sparsity_dec": s_dec,
           **dense_energy(s_enc, s_dec)}
    ref_path.write_text(json.dumps(REF, indent=2))
    tf.keras.backend.clear_session(); gc.collect()
log(f"Dense-AE acuan: F1={REF['f1']:.4f} sparsitas ReLU enc/dec={REF['relu_sparsity_enc']:.2f}/"
    f"{REF['relu_sparsity_dec']:.2f} | T1={REF['T1']:,.0f} T2={REF['T2']:,.0f} T3={REF['T3']:,.0f} pJ")
# ======== END: ACUAN DENSE-AE ========


# ======== START: OBJECTIVE MULTI-OBJEKTIF ========
def objective(trial):
    c = {"threshold": trial.suggest_float("threshold", 0.3, 2.0),
         "decay": trial.suggest_float("decay", 0.3, 0.95),
         "surrogate_scale": trial.suggest_float("surrogate_scale", 2.0, 20.0),
         "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)}
    lam = trial.suggest_categorical("lam", LAMS)
    random.seed(RS + trial.number); np.random.seed(RS + trial.number); tf.random.set_seed(RS + trial.number)
    t0 = time.time()
    try:
        m = build_snn(c, lam)
        f1, auc = fit_eval(m, c["learning_rate"])
        k_enc = float(layer_out(m, "encoder").mean() * T_STEPS)
        k_dec = float(layer_out(m, "decoder").mean() * T_STEPS)
    finally:
        tf.keras.backend.clear_session(); gc.collect()
    e = snn_energy(k_enc, k_dec)
    for k, v in {"f1": f1, "auc": auc, "k_enc": k_enc, "k_dec": k_dec, **{f"E_{a}": b for a, b in e.items()},
                 "ratio_T1": REF["T1"] / e["T1"], "ratio_T2": REF["T2"] / e["T1"],
                 "saving_T3_opt": 1 - e["T3_opt"] / REF["T3"], "saving_T3_pess": 1 - e["T3_pess"] / REF["T3"],
                 "seconds": time.time() - t0}.items():
        trial.set_user_attr(k, v)
    log(f"trial {trial.number:>2} lam={lam:<6} Vth={c['threshold']:.2f} F1={f1:.4f} AUC={auc:.4f} "
        f"k_enc={k_enc:.2f} k_dec={k_dec:.2f} | vsDense T1={REF['T1']/e['T1']:.2f}x "
        f"T2={REF['T2']/e['T1']:.2f}x T3={100*(1-e['T3_opt']/REF['T3']):+.1f}%/"
        f"{100*(1-e['T3_pess']/REF['T3']):+.1f}% | {time.time()-t0:.0f}s")
    return f1, e["T3_opt"]
# ======== END: OBJECTIVE MULTI-OBJEKTIF ========


# ======== START: EKSEKUSI DAN RINGKASAN ========
study = optuna.create_study(study_name=f"pilot|{DS}", storage=f"sqlite:///{OUT/'pilot.db'}",
                            directions=["maximize", "minimize"],
                            sampler=optuna.samplers.TPESampler(seed=RS), load_if_exists=True)
for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
    study._storage.set_trial_state_values(t._trial_id, state=TrialState.FAIL)
if len(study.trials) == 0:
    b = BEST[f"{DS}|SNN_AE"]["best_params"]
    study.enqueue_trial({"threshold": b["threshold"], "decay": b["decay"],
                         "surrogate_scale": b["surrogate_scale"], "learning_rate": b["learning_rate"], "lam": 0.0})
done = len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
study.optimize(objective, n_trials=max(0, ARGS.n_trials - done))

rows = [{"trial": t.number, **t.params, **t.user_attrs} for t in study.get_trials(states=(TrialState.COMPLETE,))]
df = pd.DataFrame(rows).sort_values("trial")
df.to_csv(OUT / "pilot_trials.csv", index=False)

f1_best = df["f1"].max()
summary = {"dense_reference": REF, "f1_best": f1_best, "trial0_S2_config": df.iloc[0].to_dict(), "selection": {}}
log(f"Acuan trial 0 (konfigurasi S2): F1={df.iloc[0]['f1']:.4f} k_enc={df.iloc[0]['k_enc']:.2f} "
    f"k_dec={df.iloc[0]['k_dec']:.2f} T3={100*df.iloc[0]['saving_T3_opt']:+.1f}%")
for delta in [0.002, 0.005, 0.01, 0.02]:
    cand = df[df["f1"] >= f1_best - delta]
    r = cand.loc[cand["E_T3_opt"].idxmin()]
    summary["selection"][str(delta)] = r.to_dict()
    log(f"dF1<={delta}: trial {int(r['trial'])} lam={r['lam']} Vth={r['threshold']:.2f} F1={r['f1']:.4f} "
        f"k_enc={r['k_enc']:.2f} k_dec={r['k_dec']:.2f} | T1={r['ratio_T1']:.2f}x T2={r['ratio_T2']:.2f}x "
        f"T3={100*r['saving_T3_opt']:+.1f}%/{100*r['saving_T3_pess']:+.1f}%")
(OUT / "pilot_summary.json").write_text(json.dumps(summary, indent=2, default=float))
log(f"Selesai | output: {OUT}")
# ======== END: EKSEKUSI DAN RINGKASAN ========
