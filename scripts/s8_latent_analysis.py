# ======== START: HEADER ========
"""
s8_latent_analysis.py — R3.10: interpretabilitas ruang laten spike SNN_AE.
Data: latent.npz S3 (1.000 normal + 1.000 serangan per fold, 30 fold).
Analisis per fold:
  (a) probe logistik CV-5 (AUC): laten SNN | laten Dense_AE (forward pass model
      tersimpan, tanpa pelatihan) | 33 fitur input terskala
  (b) silhouette di ruang asli (Euclidean), BUKAN di embedding UMAP
  (c) per neuron: laju spike normal vs serangan, Mann-Whitney (Holm) + Cliff's delta
  (d) % kode laju serangan identik dengan kode normal (pembanding: normal fold sama)
  (e) probe AUC per subtipe (normal vs subtipe)
UMAP hanya untuk visualisasi (seed0/fold0).
Jalankan satu proses per dataset:
  python s8_latent_analysis.py --dataset ciciot2023 --n-jobs 10 [--no-dense]
"""
# ======== END: HEADER ========


# ======== START: ARGUMEN DAN ENVIRONMENT ========
import argparse, os
P = argparse.ArgumentParser()
P.add_argument("--dataset", required=True, choices=["edge_iiotset", "ciciot2023"])
P.add_argument("--n-jobs", type=int, default=10)
P.add_argument("--no-dense", action="store_true")
ARGS = P.parse_args()
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[v] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json, glob, pickle, warnings
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, silhouette_score
warnings.filterwarnings("ignore")

ROOT = Path("/home/jupyteruser/nix-ext-backup/0-eksperimen-final/0-snn-ae-revision")
DS = ARGS.dataset
EVAL = ROOT / "04-final-eval/full" / DS
S1 = ROOT / "02-splits/full" / DS
OUT = ROOT / "06-posthoc/latent" / DS
OUT.mkdir(parents=True, exist_ok=True)
T = 3
DELTA_LARGE = 0.474
RS = 42
# ======== END: ARGUMEN DAN ENVIRONMENT ========


# ======== START: UTILITAS ========
def find_features():
    """Cari daftar 33 nama fitur di config_revision.json (urutan = S3)."""
    cfg = json.load(open(ROOT / "00-config/config_revision.json"))
    found = []
    def walk(o):
        if isinstance(o, dict):
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            if len(o) == 33 and all(isinstance(s, str) for s in o):
                found.append(o)
            else:
                for v in o: walk(v)
    walk(cfg)
    if not found:
        raise RuntimeError("Daftar 33 fitur tidak ditemukan di config")
    if any(f != found[0] for f in found):
        raise RuntimeError("Lebih dari satu daftar 33 fitur yang berbeda")
    return found[0]

def holm(p):
    p = np.asarray(p); m = len(p); o = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(run, 1.0)
    return adj

def probe_auc(X, y, seed):
    if len(np.unique(y)) < 2 or np.bincount(y).min() < 5:
        return np.nan
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    s = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(y, s)

def sil(X, y):
    return silhouette_score(X.astype(np.float64), y, metric="euclidean",
                            sample_size=None)
# ======== END: UTILITAS ========


# ======== START: MUAT DATA ========
FEATURES = find_features()
tab = pq.read_table(S1 / "sample.parquet", columns=["sample_id"] + FEATURES)
assert np.array_equal(tab.column("sample_id").to_numpy(), np.arange(tab.num_rows))
X_ALL = np.column_stack([tab.column(c).to_numpy(zero_copy_only=False)
                         for c in FEATURES]).astype(np.float64)
del tab
X_ALL[~np.isfinite(X_ALL)] = 0.0
X_ALL = X_ALL.astype(np.float32)          # cast float32 SEBELUM scaler

UNITS = []
for d in sorted(glob.glob(str(EVAL / "s*_f*"))):
    L = np.load(f"{d}/SNN_AE/latent.npz")
    import joblib; sc = joblib.load(f"{d}/scaler.pkl")
    Xin = sc.transform(X_ALL[L["sample_ids"]]).astype(np.float32)
    UNITS.append(dict(unit=Path(d).name, Z=L["spike_counts"].astype(np.float32),
                      y=L["y"].astype(int), lt=L["label_type"].astype(str),
                      Xin=Xin, Zd=None))
print(f"[{DS}] {len(UNITS)} unit, fitur {len(FEATURES)}", flush=True)
# ======== END: MUAT DATA ========


# ======== START: LATEN DENSE_AE (FORWARD PASS, TANPA PELATIHAN) ========
if not ARGS.no_dense:
    import tensorflow as tf
    for u in UNITS:
        mp = [p for p in glob.glob(str(EVAL / u["unit"] / "Dense_AE" / "*"))
              if p.endswith((".keras", ".h5"))]
        if len(mp) != 1:
            raise RuntimeError(f"Model Dense_AE {u['unit']}: {mp}")
        m = tf.keras.models.load_model(mp[0], compile=False)
        enc = tf.keras.Model(m.input, m.get_layer("encoder").output)
        u["Zd"] = enc.predict(u["Xin"], batch_size=1000, verbose=0)
        tf.keras.backend.clear_session()
    print(f"[{DS}] laten Dense_AE: dim {UNITS[0]['Zd'].shape[1]}", flush=True)
# ======== END: LATEN DENSE_AE ========


# ======== START: ANALISIS PER FOLD ========
def analyse(u, k):
    Z, y, lt, Xin, Zd = u["Z"], u["y"], u["lt"], u["Xin"], u["Zd"]
    seed = RS + k
    row = dict(unit=u["unit"],
               auc_latent=probe_auc(Z, y, seed), auc_input=probe_auc(Xin, y, seed),
               sil_latent=sil(Z, y), sil_input=sil(Xin, y),
               n_unique_normal=len({tuple(r) for r in Z[y == 0].astype(int)}))
    if Zd is not None:
        row.update(auc_dense=probe_auc(Zd, y, seed), sil_dense=sil(Zd, y))
    # (d) kode identik
    norm_codes = {tuple(r) for r in Z[y == 0].astype(int)}
    same = np.array([tuple(r) in norm_codes for r in Z.astype(int)])
    row["pct_attack_identical"] = 100 * same[y == 1].mean()
    # (c) per neuron
    zn, za = Z[y == 0], Z[y == 1]
    p, dl = [], []
    for j in range(Z.shape[1]):
        if np.all(Z[:, j] == Z[0, j]):
            p.append(1.0); dl.append(0.0); continue
        U, pj = mannwhitneyu(za[:, j], zn[:, j], alternative="two-sided")
        p.append(pj); dl.append(2 * U / (len(za) * len(zn)) - 1)
    neu = pd.DataFrame(dict(unit=u["unit"], neuron=np.arange(Z.shape[1]),
                            rate_normal=zn.mean(0) / T, rate_attack=za.mean(0) / T,
                            p_holm=holm(p), cliffs_delta=dl))
    # (e) per subtipe
    sub = []
    for s in sorted(set(lt[y == 1])):
        m = (y == 0) | (lt == s)
        ys = (lt[m] == s).astype(int)
        d = dict(unit=u["unit"], subtype=s, n=int(ys.sum()),
                 auc_latent=probe_auc(Z[m], ys, seed),
                 auc_input=probe_auc(Xin[m], ys, seed),
                 pct_identical=100 * same[(lt == s) & (y == 1)].mean())
        if Zd is not None:
            d["auc_dense"] = probe_auc(Zd[m], ys, seed)
        sub.append(d)
    return row, neu, pd.DataFrame(sub)

res = Parallel(n_jobs=ARGS.n_jobs)(delayed(analyse)(u, k) for k, u in enumerate(UNITS))
fold = pd.DataFrame([r[0] for r in res])
neu = pd.concat([r[1] for r in res]); sub = pd.concat([r[2] for r in res])
fold.to_csv(OUT / "per_fold.csv", index=False)
neu.to_csv(OUT / "neurons_per_fold.csv", index=False)
sub.to_csv(OUT / "subtype_per_fold.csv", index=False)
# ======== END: ANALISIS PER FOLD ========


# ======== START: RINGKASAN ========
num = fold.drop(columns="unit")
neu["large_sig"] = (neu.p_holm < 0.05) & (neu.cliffs_delta.abs() >= DELTA_LARGE)
nsum = neu.groupby("neuron").agg(rate_normal=("rate_normal", "mean"),
                                 rate_attack=("rate_attack", "mean"),
                                 delta_median=("cliffs_delta", "median"),
                                 folds_large_sig=("large_sig", "sum"))
nsum.to_csv(OUT / "neurons_summary.csv")
ssum = sub.drop(columns="unit").groupby("subtype").agg(["median",
          lambda s: s.quantile(.25), lambda s: s.quantile(.75)])
ssum.to_csv(OUT / "subtype_summary.csv")
summary = dict(dataset=DS, n_units=len(fold),
               mean=num.mean().round(4).to_dict(), sd=num.std().round(4).to_dict(),
               neurons_silent=int((nsum.rate_normal + nsum.rate_attack == 0).sum()),
               neurons_large_sig_all_folds=int((nsum.folds_large_sig == len(fold)).sum()))
json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2), flush=True)
# ======== END: RINGKASAN ========


# ======== START: UMAP SEED0/FOLD0 (VISUALISASI SAJA) ========
import umap, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
u0 = next(u for u in UNITS if u["unit"] == "s0_f0")
emb = {}
for name, X in [("latent", u0["Z"]), ("input", u0["Xin"])] + \
               ([("dense", u0["Zd"])] if u0["Zd"] is not None else []):
    emb[name] = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=RS
                          ).fit_transform(X.astype(np.float64))
np.savez(OUT / "umap_s0_f0.npz", y=u0["y"], label_type=u0["lt"], **emb)
fig, ax = plt.subplots(2, len(emb), figsize=(5 * len(emb), 9))
for c, (name, E) in enumerate(emb.items()):
    for v, lab in [(0, "normal"), (1, "attack")]:
        m = u0["y"] == v
        ax[0, c].scatter(E[m, 0], E[m, 1], s=3, alpha=.5, label=lab)
    for s in sorted(set(u0["lt"])):
        m = u0["lt"] == s
        ax[1, c].scatter(E[m, 0], E[m, 1], s=3, alpha=.5, label=s)
    ax[0, c].set_title(name); ax[0, c].legend(markerscale=4, fontsize=7)
ax[1, -1].legend(markerscale=4, fontsize=5, ncol=2, bbox_to_anchor=(1, 1))
fig.tight_layout(); fig.savefig(OUT / "umap_s0_f0.png", dpi=150)
print(f"[{DS}] selesai -> {OUT}", flush=True)
# ======== END: UMAP SEED0/FOLD0 ========
