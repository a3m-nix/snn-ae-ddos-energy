# ======== START: HEADER ========
"""
S4 — INTEGRITAS, TABEL, DAN STATISTIK (TANPA TRAINING)
Revisi Paper 1 SNN-AE — Scientific Reports

Input : 02-splits/full (S1), 04-final-eval/full (S3), 00-config
Output: 05-tables/  (CSV per tabel, workbook Excel, summary.json)

Bagian:
  A. Integritas & provenance: hash S1, config (33 fitur, scaler), indeks per unit,
     disjointness grup normal, kelengkapan 780 hasil, skor finite, konsistensi
     metrik result.json vs errors.npz, versi pustaka.
  B. Tabel deteksi (mean ± sd) + CI 95% bootstrap untuk F1, AUC, AP.
  C. Statistik: Friedman (4 AE) + Wilcoxon-Holm + Cohen's d (paired) untuk
     F1, AUC, AP; famili terpisah: SNN-AE vs baseline, dan ablasi.
  D. Grid threshold dari errors.npz: p90 ... p99.99, max-normal (F1, recall,
     FPR aktual vs target).
  E. Recall per subtipe (p99).
  F. Ablasi (BN, time step, arsitektur) + k_enc/k_dec.
  G. Variabilitas per fold (SNN-AE vs Dense-AE).
Catatan: "auc_pr" di S3 = average_precision_score -> dilaporkan sebagai AP.
Energi di S3 = arsip (rumus v8.3); energi final dihitung di S6.
"""
# ======== END: HEADER ========


# ======== START: IMPORT DAN ARGUMEN ========
import argparse
import gc
import hashlib
import json
import platform
import warnings
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import friedmanchisquare, wilcoxon

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", nargs="+", default=["edge_iiotset", "ciciot2023"])
parser.add_argument("--bootstrap", type=int, default=10_000)
parser.add_argument("--seed", type=int, default=42)
ARGS, _ = parser.parse_known_args()
# ======== END: IMPORT DAN ARGUMEN ========


# ======== START: KONFIGURASI ========
WORK = Path("/home/jupyteruser/nix-ext-backup/0-eksperimen-final/0-snn-ae-revision")
CFG = json.loads((WORK / "00-config" / "config_revision.json").read_text())
OUT = WORK / "05-tables"
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = CFG["data"]["features"]
SEEDS = list(range(len(CFG["split"]["seeds"])))
N_FOLDS = CFG["split"]["n_folds"]
AE_JOBS = [j[0] for j in CFG["evaluation"]["ae_jobs"]]
BL_JOBS = CFG["evaluation"]["baseline_jobs"]
ALL_JOBS = AE_JOBS + BL_JOBS

BENCH = ["SNN_AE", "Dense_AE", "LSTM_AE", "CNN_AE"]
BASELINES = ["OCSVM", "IF", "LOF"]
ABLATION = ["SNN_AE_noBN", "SNN_AE_T2", "SNN_AE_T5", "SNN_AE_T7", "SNN_Dense", "Dense_SNN"]
ORDER = BENCH + BASELINES + ABLATION
METRICS = ["f1", "auc", "ap"]
GRID = [90, 95, 96, 97, 98, 99, 99.5, 99.9, 99.99, "max"]
OP = CFG["evaluation"]["operating_percentile"]
DS_LABEL = {"edge_iiotset": "Edge-IIoTset", "ciciot2023": "CICIoT2023"}


def log(m):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [S4] {m}", flush=True)
# ======== END: KONFIGURASI ========


# ======== START: UTILITAS ========
def array_hash(*arrays):
    """Identik dengan S1."""
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).view(np.uint8))
    return h.hexdigest()[:16]


def fmt(m, s, d=4):
    return f"{m:.{d}f} ± {s:.{d}f}"


def bootstrap_ci(x, n, rng):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def paired_test(a, b):
    """Wilcoxon two-sided + mean diff + Cohen's d (paired)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    sd = d.std(ddof=1)
    cohen = float(d.mean() / sd) if sd > 0 else 0.0
    if np.allclose(d, 0):
        p, stat = 1.0, np.nan
    else:
        try:
            stat, p = wilcoxon(a, b, alternative="two-sided")
        except ValueError:
            stat, p = np.nan, 1.0
    return {"mean_a": a.mean(), "mean_b": b.mean(), "mean_diff": d.mean(),
            "cohen_d": cohen, "W": stat, "p": float(p)}


def versions():
    v = {"python": platform.python_version()}
    for pkg in ["numpy", "pandas", "scipy", "scikit-learn", "tensorflow",
                "optuna", "pyarrow", "joblib", "brian2"]:
        try:
            v[pkg] = metadata.version(pkg)
        except Exception:
            v[pkg] = None
    return v
# ======== END: UTILITAS ========


# ======== START: A. INTEGRITAS DAN PROVENANCE ========
def check_config():
    issues = []
    if len(FEATURES) != 33:
        issues.append(f"jumlah fitur {len(FEATURES)} != 33")
    for bad in ["frame.time_epoch", "ip.src", "ip.dst", "label", "label_type", "flow_id"]:
        if bad in FEATURES:
            issues.append(f"fitur terlarang: {bad}")
    if CFG["data"]["scaler_type"] != {"edge_iiotset": "quantile", "ciciot2023": "minmax"}:
        issues.append(f"pemetaan scaler tidak sesuai: {CFG['data']['scaler_type']}")
    return issues


def integrity(ds):
    s1 = WORK / "02-splits" / "full" / ds
    s3 = WORK / "04-final-eval" / "full" / ds
    rep = {"dataset": ds, "issues": []}
    done = json.loads((s1 / "done.json").read_text())

    t = pq.read_table(s1 / "sample.parquet", columns=["row_idx", "y", "label_type", "group_code"])
    row_idx = t.column("row_idx").to_numpy()
    y = t.column("y").to_numpy()
    lt = np.asarray(t.column("label_type").to_pylist(), dtype=object)
    grp = t.column("group_code").to_numpy()
    del t
    if array_hash(row_idx, y) != done["sample_hash"]:
        rep["issues"].append("sample_hash tidak cocok dengan S1")
    sp = dict(np.load(s1 / "splits.npz"))
    if array_hash(*[sp[k] for k in sorted(sp)]) != done["splits_hash"]:
        rep["issues"].append("splits_hash tidak cocok dengan S1")
    hold = sp["holdout"]

    n_missing, nonfinite, len_mismatch, max_diff = 0, 0, 0, {}
    for s in SEEDS:
        for f in range(N_FOLDS):
            key = f"s{s}_f{f}"
            udir = s3 / key
            tr, va, te = sp[f"{key}_train"], sp[f"{key}_val"], sp[f"{key}_test"]
            ind = np.load(udir / "indices.npz")
            for name, arr in [("train", tr), ("val", va), ("test", te)]:
                if not np.array_equal(ind[name], arr):
                    rep["issues"].append(f"{key}: indices {name} berbeda dari S1")
            if (y[tr] != 0).any() or (y[va] != 0).any():
                rep["issues"].append(f"{key}: train/val memuat serangan")
            g_tr, g_va = np.unique(grp[tr]), np.unique(grp[va])
            g_te = np.unique(grp[te[y[te] == 0]])
            for a, b, n in [(g_tr, g_va, "train-val"), (g_tr, g_te, "train-test"), (g_va, g_te, "val-test")]:
                if len(np.intersect1d(a, b)):
                    rep["issues"].append(f"{key}: irisan grup normal {n}")
            if np.isin(np.concatenate([tr, va, te]), hold).any():
                rep["issues"].append(f"{key}: memuat holdout")

            for job in ALL_JOBS:
                jdir = udir / job
                if not (jdir / "result.json").exists():
                    n_missing += 1
                    continue
                e = np.load(jdir / "errors.npz")
                ve, tes = e["val_errors"].astype(np.float64), e["test_errors"].astype(np.float64)
                if len(ve) != len(va) or len(tes) != len(te):
                    len_mismatch += 1
                if not (np.isfinite(ve).all() and np.isfinite(tes).all()):
                    nonfinite += 1
                r = json.loads((jdir / "result.json").read_text())
                thr = np.percentile(ve, OP)
                pred = tes > thr
                yt = y[te]
                tp = int((pred & (yt == 1)).sum()); fp = int((pred & (yt == 0)).sum())
                fn = int((~pred & (yt == 1)).sum())
                f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
                diff = abs(f1 - r["per_pct"][str(OP)]["f1"])
                max_diff[job] = max(max_diff.get(job, 0.0), diff)

    rep.update({"n_results_missing": n_missing, "n_nonfinite_scores": nonfinite,
                "n_length_mismatch": len_mismatch,
                "max_abs_f1_diff_result_vs_errors": max_diff,
                "n_units": len(SEEDS) * N_FOLDS, "n_jobs": len(ALL_JOBS)})
    return rep, y, lt, sp
# ======== END: A. INTEGRITAS DAN PROVENANCE ========


# ======== START: PENGUMPULAN HASIL PER FOLD ========
def collect(ds):
    rows = []
    s3 = WORK / "04-final-eval" / "full" / ds
    for p in sorted(s3.glob("s*_f*/*/result.json")):
        r = json.loads(p.read_text())
        o = r["per_pct"][str(OP)]
        e = r.get("energy") or {}
        T = r.get("T") or 0

        def rate(k):
            v = e.get(k)
            return float(v) if v is not None else np.nan
        rows.append({
            "dataset": ds, "job": r["job_id"], "seed": r["seed_idx"], "fold": r["fold"],
            "accuracy": o["accuracy"], "precision": o["precision"], "recall": o["recall"],
            "f1": o["f1"], "fpr_test": o["fpr_test"], "collapsed": o["collapsed"],
            "auc": r["auc"] if r["auc"] is not None else np.nan,
            "ap": r["auc_pr"] if r["auc_pr"] is not None else np.nan,
            "T": T or np.nan,
            "enc_rate_mixed": rate("encoder_rate_mixed"),
            "dec_rate_mixed": rate("decoder_rate_mixed"),
            "enc_rate_normal": rate("encoder_rate_normal"),
            "enc_rate_attack": rate("encoder_rate_attack"),
            "dec_rate_normal": rate("decoder_rate_normal"),
            "dec_rate_attack": rate("decoder_rate_attack"),
            "energy_archival_pj": e.get("total_pj", np.nan),
            "params": r.get("params") if isinstance(r.get("params"), int) else np.nan,
            "train_time_sec": r.get("train_time_sec"),
            "best_epoch": r.get("best_epoch"),
        })
    d = pd.DataFrame(rows)
    d["k_enc"] = d["enc_rate_mixed"] * d["T"]
    d["k_dec"] = d["dec_rate_mixed"] * d["T"]
    return d
# ======== END: PENGUMPULAN HASIL PER FOLD ========


# ======== START: B. TABEL DETEKSI + CI ========
def detection_table(d, rng):
    out = []
    for job in [j for j in ORDER if j in d["job"].unique()]:
        x = d[d["job"] == job]
        row = {"dataset": DS_LABEL[x["dataset"].iloc[0]], "model": job, "n_folds": len(x)}
        for m in ["accuracy", "precision", "recall", "f1", "auc", "ap", "fpr_test"]:
            row[m] = fmt(x[m].mean(), x[m].std(ddof=1))
        for m in METRICS:
            lo, hi = bootstrap_ci(x[m], ARGS.bootstrap, rng)
            row[f"{m}_ci95"] = f"[{lo:.4f}, {hi:.4f}]"
        row["collapsed_folds"] = int(x["collapsed"].sum())
        out.append(row)
    return pd.DataFrame(out)
# ======== END: B. TABEL DETEKSI + CI ========


# ======== START: C. STATISTIK ========
def aligned(d, job, metric):
    x = d[d["job"] == job].sort_values(["seed", "fold"])
    return x[metric].to_numpy()


def stats_tables(d):
    ds = DS_LABEL[d["dataset"].iloc[0]]
    fried, pair = [], []
    for metric in METRICS:
        mats = [aligned(d, j, metric) for j in BENCH]
        try:
            stat, p = friedmanchisquare(*mats)
        except Exception:
            stat, p = np.nan, np.nan
        fried.append({"dataset": ds, "metric": metric.upper(), "models": ", ".join(BENCH),
                      "chi2": stat, "p": p, "n_blocks": len(mats[0])})

        families = {
            "benchmark_AE": [(a, b) for i, a in enumerate(BENCH) for b in BENCH[i + 1:]],
            "SNN_vs_baseline": [("SNN_AE", b) for b in BASELINES],
            "ablation": [("SNN_AE", b) for b in ABLATION],
        }
        for fam, pairs in families.items():
            res = [dict(paired_test(aligned(d, a, metric), aligned(d, b, metric)),
                        dataset=ds, metric=metric.upper(), family=fam, model_a=a, model_b=b)
                   for a, b in pairs]
            ph = holm([r["p"] for r in res])
            for r, h in zip(res, ph):
                r["p_holm"] = h
                r["significant_0.05"] = bool(h < 0.05)
            pair.extend(res)
    cols = ["dataset", "metric", "family", "model_a", "model_b", "mean_a", "mean_b",
            "mean_diff", "cohen_d", "W", "p", "p_holm", "significant_0.05"]
    return pd.DataFrame(fried), pd.DataFrame(pair)[cols]
# ======== END: C. STATISTIK ========


# ======== START: D. GRID THRESHOLD DARI ERRORS.NPZ ========
def threshold_grid(ds, y, sp):
    s3 = WORK / "04-final-eval" / "full" / ds
    rows = []
    for s in SEEDS:
        for f in range(N_FOLDS):
            key = f"s{s}_f{f}"
            yt = y[sp[f"{key}_test"]]
            for job in ALL_JOBS:
                e = np.load(s3 / key / job / "errors.npz")
                ve = e["val_errors"].astype(np.float64)
                te = e["test_errors"].astype(np.float64)
                for p in GRID:
                    thr = float(ve.max()) if p == "max" else float(np.percentile(ve, p))
                    pred = te > thr
                    tp = int((pred & (yt == 1)).sum()); fp = int((pred & (yt == 0)).sum())
                    fn = int((~pred & (yt == 1)).sum()); tn = int((~pred & (yt == 0)).sum())
                    prec = tp / (tp + fp) if (tp + fp) else 0.0
                    rec = tp / (tp + fn) if (tp + fn) else 0.0
                    rows.append({"dataset": DS_LABEL[ds], "job": job, "seed": s, "fold": f,
                                 "percentile": str(p),
                                 "fpr_target": round(1 - p / 100, 6) if p != "max" else 0.0,
                                 "precision": prec, "recall": rec,
                                 "f1": 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0,
                                 "fpr_test": fp / (fp + tn) if (fp + tn) else np.nan})
    long = pd.DataFrame(rows)
    agg = (long.groupby(["dataset", "job", "percentile"], sort=False)
           .agg(fpr_target=("fpr_target", "first"),
                f1_mean=("f1", "mean"), f1_sd=("f1", "std"),
                recall_mean=("recall", "mean"), precision_mean=("precision", "mean"),
                fpr_test_mean=("fpr_test", "mean"), fpr_test_sd=("fpr_test", "std"))
           .reset_index())
    agg["fpr_ratio_actual_to_target"] = agg["fpr_test_mean"] / agg["fpr_target"].replace(0, np.nan)
    return long, agg
# ======== END: D. GRID THRESHOLD DARI ERRORS.NPZ ========


# ======== START: E. RECALL PER SUBTIPE ========
def subtype_table(ds):
    s3 = WORK / "04-final-eval" / "full" / ds
    rows = []
    for p in s3.glob("s*_f*/*/result.json"):
        r = json.loads(p.read_text())
        for st, v in (r.get("subtype_recall") or {}).items():
            rows.append({"dataset": DS_LABEL[ds], "job": r["job_id"], "subtype": st,
                         "n": v["n"], "recall": v["recall_op"]})
    long = pd.DataFrame(rows)
    agg = long.groupby(["dataset", "subtype", "job"]).agg(
        recall_mean=("recall", "mean"), recall_sd=("recall", "std"),
        n_mean=("n", "mean")).reset_index()
    wide = agg.pivot_table(index=["dataset", "subtype"], columns="job",
                           values="recall_mean").reset_index()
    cols = ["dataset", "subtype"] + [j for j in ORDER if j in wide.columns]
    return agg, wide[cols]
# ======== END: E. RECALL PER SUBTIPE ========


# ======== START: F. ABLASI DAN G. VARIABILITAS FOLD ========
def ablation_table(d):
    rows = []
    for job in ["SNN_AE"] + ABLATION + ["Dense_AE"]:
        x = d[d["job"] == job]
        if x.empty:
            continue
        rows.append({"dataset": DS_LABEL[x["dataset"].iloc[0]], "model": job,
                     "T": x["T"].iloc[0], "f1": fmt(x["f1"].mean(), x["f1"].std(ddof=1)),
                     "auc": fmt(x["auc"].mean(), x["auc"].std(ddof=1)),
                     "ap": fmt(x["ap"].mean(), x["ap"].std(ddof=1)),
                     "k_enc": round(x["k_enc"].mean(), 4), "k_dec": round(x["k_dec"].mean(), 4),
                     "enc_rate_normal": round(x["enc_rate_normal"].mean(), 4),
                     "enc_rate_attack": round(x["enc_rate_attack"].mean(), 4),
                     "energy_archival_pj": round(x["energy_archival_pj"].mean(), 1),
                     "params": x["params"].iloc[0]})
    return pd.DataFrame(rows)


def fold_variability(d):
    a = d[d["job"] == "SNN_AE"].set_index(["seed", "fold"])[METRICS]
    b = d[d["job"] == "Dense_AE"].set_index(["seed", "fold"])[METRICS]
    w = a.join(b, lsuffix="_SNN_AE", rsuffix="_Dense_AE").reset_index()
    for m in METRICS:
        w[f"{m}_diff"] = w[f"{m}_SNN_AE"] - w[f"{m}_Dense_AE"]
    w.insert(0, "dataset", DS_LABEL[d["dataset"].iloc[0]])
    return w
# ======== END: F. ABLASI DAN G. VARIABILITAS FOLD ========


# ======== START: EKSEKUSI ========
rng = np.random.default_rng(ARGS.seed)
cfg_issues = check_config()
summary = {"generated_at": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "versions": versions(),
           "config_issues": cfg_issues, "integrity": {}, "bootstrap": ARGS.bootstrap}
tables = {k: [] for k in ["folds_long", "detection", "friedman", "pairwise", "threshold_grid",
                          "threshold_grid_folds", "subtype_long", "subtype_wide",
                          "ablation", "fold_variability"]}

for ds in ARGS.datasets:
    log(f"{ds}: integritas ...")
    rep, y, lt, sp = integrity(ds)
    summary["integrity"][ds] = rep
    log(f"{ds}: issues={len(rep['issues'])} missing={rep['n_results_missing']} "
        f"nonfinite={rep['n_nonfinite_scores']} len_mismatch={rep['n_length_mismatch']} "
        f"max|dF1|={max(rep['max_abs_f1_diff_result_vs_errors'].values()):.2e}")

    d = collect(ds)
    tables["folds_long"].append(d)
    tables["detection"].append(detection_table(d, rng))
    fr, pw = stats_tables(d)
    tables["friedman"].append(fr)
    tables["pairwise"].append(pw)
    log(f"{ds}: grid threshold ...")
    gl, ga = threshold_grid(ds, y, sp)
    tables["threshold_grid_folds"].append(gl)
    tables["threshold_grid"].append(ga)
    sa, sw = subtype_table(ds)
    tables["subtype_long"].append(sa)
    tables["subtype_wide"].append(sw)
    tables["ablation"].append(ablation_table(d))
    tables["fold_variability"].append(fold_variability(d))
    del y, lt, sp, d
    gc.collect()

final = {k: pd.concat(v, ignore_index=True) for k, v in tables.items() if v}
for name, df in final.items():
    df.to_csv(OUT / f"{name}.csv", index=False)

try:
    with pd.ExcelWriter(OUT / "s4_tables.xlsx") as xw:
        for name, df in final.items():
            if name in ("threshold_grid_folds",):
                continue
            df.to_excel(xw, sheet_name=name[:31], index=False)
except Exception as e:
    log(f"Excel dilewati: {e!r}")

pw = final["pairwise"]
summary["key_results"] = {
    "snn_vs_dense": pw[(pw.model_a == "SNN_AE") & (pw.model_b == "Dense_AE")]
    .to_dict("records"),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

all_issues = cfg_issues + sum([summary["integrity"][d]["issues"] for d in ARGS.datasets], [])
log(f"Selesai | issues total={len(all_issues)} | output: {OUT}")
for i in all_issues[:20]:
    log(f"ISSUE: {i}")
print(final["detection"][["dataset", "model", "f1", "auc", "ap", "fpr_test"]].to_string(index=False))
print(pw[(pw.model_a == "SNN_AE") & (pw.model_b == "Dense_AE")]
      [["dataset", "metric", "mean_diff", "cohen_d", "p_holm", "significant_0.05"]].to_string(index=False))
# ======== END: EKSEKUSI ========
