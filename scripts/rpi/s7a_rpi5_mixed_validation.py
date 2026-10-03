# ======== START: HEADER ========
"""
S7a — VALIDASI RASPBERRY PI 5 DENGAN TRAFIK CAMPURAN (REVISI)
Revisi Paper 1 SNN-AE — Scientific Reports

Prinsip: metode pengukuran IDENTIK dengan artikel lama. Seluruh fungsi pengukuran
diimpor langsung dari rpi5_fnirsi_attack_autov2.py (FNIRSI, idle, ResourceMonitor,
calc_board_metrics, SpikingLayer, prepare_input, reconstruction_error, warm-up,
countdown, 50 pengulangan, batch 256, buang pengulangan pertama).

Hanya empat perubahan (sesuai reviewer):
  1. Data: test_raw.parquet dari payload (seed 0/fold 0, normal + serangan, disjoint
     per flow dari training) menggantikan parquet lama berisi serangan saja (R2.11, R2.5).
  2. Artefak: model, scaler, thresholds.json (p99 validation-normal), features.json
     dari payload hasil revisi.
  3. Metrik deteksi trafik campuran: precision, recall, F1, FPR, AUC, AP, per subtipe.
  4. Verifikasi: skor on-device dicocokkan dengan skor referensi server.
Opsional: --with_baselines (IF, OCSVM, LOF; R3.6), --engine tflite.
"""
# ======== END: HEADER ========


# ======== START: IMPORT KODE LAMA DAN PUSTAKA ========
import argparse
import hashlib
import importlib.util
import json
import os
import time
from datetime import datetime
from pathlib import Path

HOME_DIR = Path("/home/a3m-nix/snn-ae-final")
BASE_SCRIPT = HOME_DIR / "rpi5_fnirsi_attack_autov2.py"
spec = importlib.util.spec_from_file_location("base_v2", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)          # env TF thread 4/2 dan SpikingLayer dari kode lama
for fn in ["FNIRSIReader", "measure_idle_power", "ResourceMonitor", "calc_board_metrics",
           "collect_system_info", "get_cpu_temp", "get_throttling_status", "print_system_warning",
           "prepare_input", "reconstruction_error", "sanitize_for_json", "SpikingLayer"]:
    if not hasattr(base, fn):
        raise SystemExit(f"Fungsi {fn} tidak ditemukan di {BASE_SCRIPT}")

import gc
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sklearn
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score
# ======== END: IMPORT KODE LAMA DAN PUSTAKA ========


# ======== START: KONFIGURASI ========
PAYLOAD_DIR = HOME_DIR / "0-revision-payload"
RESULT_DIR = HOME_DIR / "0-revision-results" / "s7a"
AE_MODELS = ["SNN_AE", "Dense_AE", "LSTM_AE", "CNN_AE"]
BASELINES = ["IF", "OCSVM", "LOF"]
OP = "99"


def log(m):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {m}", flush=True)
# ======== END: KONFIGURASI ========


# ======== START: PREFLIGHT (INTEGRITAS & VERSI) ========
def preflight(ds, allow_version_mismatch):
    p = PAYLOAD_DIR / ds
    man = json.loads((p / "manifest.json").read_text())
    bad = [f for f, h in man["sha256"].items()
           if not (p / f).exists() or hashlib.sha256((p / f).read_bytes()).hexdigest() != h]
    if bad:
        raise SystemExit(f"{ds}: file payload rusak/hilang: {bad}")
    srv = man["versions_server"].get("scikit-learn")
    if sklearn.__version__ != srv and not allow_version_mismatch:
        raise SystemExit(f"scikit-learn RPi {sklearn.__version__} != server {srv}. "
                         f"Jalankan: pip install scikit-learn=={srv}")
    log(f"{ds}: payload OK ({len(man['sha256'])} file) | sklearn {sklearn.__version__} "
        f"(server {srv}) | TF {tf.__version__} (server {man['versions_server'].get('tensorflow')})")
    return man
# ======== END: PREFLIGHT (INTEGRITAS & VERSI) ========


# ======== START: DATA DAN ARTEFAK PAYLOAD ========
def load_payload(ds):
    p = PAYLOAD_DIR / ds
    features = json.loads((p / "features.json").read_text())
    thresholds = json.loads((p / "thresholds.json").read_text())
    scaler = joblib.load(p / "scaler.pkl")
    df = pq.read_table(p / "test_raw.parquet").to_pandas()
    # Jalur input identik kode lama: replace inf -> NaN -> 0, float32, lalu scaler.
    Xdf = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    X = Xdf.values.astype(np.float32)
    Xs = scaler.transform(X).astype(np.float32)
    ref = np.load(p / "server_reference_scores.npz")
    if not np.array_equal(ref["sample_ids"], df["sample_id"].to_numpy()):
        raise SystemExit(f"{ds}: urutan sampel tidak sesuai referensi server")
    return {"features": features, "thr": thresholds, "Xs": Xs, "y": df["y"].to_numpy().astype(int),
            "lt": df["label_type"].astype(str).to_numpy(), "ids": df["sample_id"].to_numpy(),
            "ref": ref}


def stratified_subset(y, n_per_class, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), min(n_per_class, int((y == c).sum())),
                                     replace=False) for c in (0, 1)])
    return np.sort(idx)
# ======== END: DATA DAN ARTEFAK PAYLOAD ========


# ======== START: SKOR (KERAS / TFLITE / BASELINE) ========
class Scorer:
    def __init__(self, ds, model_name, engine, batch):
        self.name, self.engine, self.batch = model_name, engine, batch
        mdir = PAYLOAD_DIR / ds / "models"
        if model_name in BASELINES:
            self.kind = "baseline"
            self.path = mdir / f"{model_name}.pkl"
            self.model = joblib.load(self.path)
        elif engine == "tflite":
            self.kind = "tflite"
            self.path = mdir / f"{model_name}.tflite"
            self.model = tf.lite.Interpreter(model_path=str(self.path))
            self.inp = self.model.get_input_details()[0]
            self.out = self.model.get_output_details()[0]
        else:
            self.kind = "keras"
            self.path = mdir / f"{model_name}.keras"
            self.model = tf.keras.models.load_model(self.path, compile=False,
                                                    custom_objects={"SpikingLayer": base.SpikingLayer})

    def predict_raw(self, Xs):
        """Satu inferensi penuh (yang diukur latensi dan dayanya)."""
        if self.kind == "baseline":
            return -self.model.score_samples(Xs)
        Xin = base.prepare_input(Xs, self.name)
        if self.kind == "keras":
            return self.model.predict(Xin, batch_size=self.batch, verbose=0)
        outs = []
        for i in range(0, len(Xin), self.batch):
            xb = Xin[i:i + self.batch].astype(np.float32)
            self.model.resize_tensor_input(self.inp["index"], xb.shape)
            self.model.allocate_tensors()
            self.model.set_tensor(self.inp["index"], xb)
            self.model.invoke()
            outs.append(self.model.get_tensor(self.out["index"]).copy())
        return np.concatenate(outs).reshape((len(Xin),) + outs[0].shape[1:])

    def scores(self, Xs, raw=None):
        raw = self.predict_raw(Xs) if raw is None else raw
        if self.kind == "baseline":
            return np.asarray(raw, dtype=np.float64)
        return base.reconstruction_error(Xs, raw, self.name).astype(np.float64)
# ======== END: SKOR (KERAS / TFLITE / BASELINE) ========


# ======== START: METRIK DETEKSI TRAFIK CAMPURAN ========
def detection_metrics(s, y, lt, thr):
    pred = s > thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    sub = {st: float(pred[(y == 1) & (lt == st)].mean()) for st in sorted(set(lt[y == 1]))}
    return {"threshold": thr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "fpr": fp / (fp + tn) if fp + tn else float("nan"),
            "auc": float(roc_auc_score(y, s)), "ap": float(average_precision_score(y, s)),
            "subtype_recall": sub, "n": int(len(y))}
# ======== END: METRIK DETEKSI TRAFIK CAMPURAN ========


# ======== START: SATU RUN (STRUKTUR SAMA DENGAN run_one LAMA) ========
class FnirsiDropout(RuntimeError):
    pass


def run_one(ds, model_name, P, args, fn, last_attempt):
    fnirsi, idle = fn["r"], fn["idle"]
    log(f"=== {ds} | {model_name} | engine={args.engine} ===")
    sys_before = base.collect_system_info()
    temp_before, thr_before = base.get_cpu_temp(), base.get_throttling_status()
    sc = Scorer(ds, model_name, args.engine, args.predict_batch)
    thr = float(P["thr"][model_name][OP])

    # (A) Deteksi pada seluruh test fold + verifikasi terhadap server (di luar pengukuran daya)
    s_full = sc.scores(P["Xs"])
    ref = P["ref"][model_name].astype(np.float64)
    max_diff = float(np.max(np.abs(s_full - ref)))
    agree = float(np.mean((s_full > thr) == (ref > thr)))
    if not np.isfinite(s_full).all():
        raise SystemExit(f"{model_name}: skor non-finite")
    det_full = detection_metrics(s_full, P["y"], P["lt"], thr)
    np.savez_compressed(RESULT_DIR / f"scores_full_{ds}_{model_name}_{sc.kind}.npz", sample_ids=P["ids"],
                        y=P["y"], rpi_scores=s_full, server_scores=ref, threshold=thr)
    log(f"  full test: F1={det_full['f1']:.4f} P={det_full['precision']:.4f} R={det_full['recall']:.4f} "
        f"FPR={det_full['fpr']:.4f} | vs server: max|Δskor|={max_diff:.2e} keputusan sama={agree:.6f}")
    if agree < args.min_agreement:
        raise SystemExit(f"{model_name}: kesamaan keputusan {agree:.6f} < {args.min_agreement}")

    # (B) Pengukuran latensi & daya pada subset campuran (metode lama)
    idx = P["subset"]
    Xs = P["Xs"][idx]
    warm = Xs[:min(base.WARMUP_SAMPLES, len(Xs))]
    for _ in range(base.WARMUP_RUNS):
        sc.predict_raw(warm)
    for i in range(base.COUNTDOWN_SEC, 0, -1):
        print(f"  Start in {i}...", flush=True); time.sleep(1)
    if fnirsi:
        fnirsi.clear_readings()
    mon = base.ResourceMonitor(interval_sec=args.resource_interval)
    repeat = args.repeat_baseline if model_name in BASELINES else args.repeat
    lat, raw_last = [], None
    mon.start(); t0_all = time.perf_counter()
    try:
        for i in range(repeat):
            t0 = time.perf_counter()
            raw_last = sc.predict_raw(Xs)
            lat.append((time.perf_counter() - t0) * 1000.0)
            print(f"  [{i + 1}/{repeat}] {lat[-1]:.2f} ms", end="\r", flush=True)
    finally:
        elapsed_all = (time.perf_counter() - t0_all) * 1000.0
        mon.stop()
    print()
    used = lat[1:] if len(lat) > 1 else lat
    lat_mean, lat_std = float(np.mean(used)), float(np.std(used))
    board, power_valid, fn_diag = None, None, None
    if fnirsi:
        r = fnirsi.get_readings()
        rate = len(r) / max(elapsed_all / 1000.0, 1e-9)
        alive = fnirsi.is_alive()
        power_valid = bool(alive and fn["rate"] and rate >= 0.5 * fn["rate"] and len(r) > 0)
        fn_diag = {"alive_after": alive, "readings": len(r), "rate_hz": rate, "idle_rate_hz": fn["rate"]}
        if not power_valid and not last_attempt:
            raise FnirsiDropout(f"FNIRSI tidak valid: {fn_diag}")
        if r:
            board = base.calc_board_metrics(float(np.mean(r)), float(np.std(r)), len(r),
                                            idle[0], idle[1], idle[2], lat_mean, lat_std, len(Xs))
            board["fnirsi_diagnostics"] = fn_diag
    s_sub = sc.scores(Xs, raw_last)
    np.savez_compressed(RESULT_DIR / f"scores_subset_{ds}_{model_name}_{sc.kind}.npz",
                        sample_ids=P["ids"][idx], y=P["y"][idx], rpi_scores=s_sub, latency_ms=np.asarray(lat))
    det_sub = detection_metrics(s_sub, P["y"][idx], P["lt"][idx], thr)

    res = {"dataset": ds, "model": model_name, "engine": args.engine if sc.kind != "baseline" else "sklearn",
           "methodology_guard": {"training_performed": False, "scaler_fit_performed": False,
                                 "threshold_selected_from_test_data": False,
                                 "threshold_source": "thresholds.json p99 (validation-normal, server)",
                                 "data": "payload test fold s0_f0 (normal + attack), flow-disjoint from training"},
           "verification_vs_server": {"max_abs_score_diff": max_diff, "decision_agreement": agree},
           "detection_full_test": det_full, "detection_measurement_subset": det_sub,
           "latency_metrics": {"repeat": repeat, "discarded_first_repeat": len(lat) > 1,
                               "elapsed_total_ms_all_repeats": elapsed_all,
                               "latency_all_repeats_ms": lat,
                               "latency_mean_ms_per_full_batch": lat_mean, "latency_std_ms_per_full_batch": lat_std,
                               "latency_ms_per_sample": lat_mean / len(Xs),
                               "latency_us_per_sample": lat_mean / len(Xs) * 1000.0,
                               "throughput_sps": len(Xs) / (lat_mean / 1000.0), "n_samples": int(len(Xs))},
           "board_power_metrics": board, "power_valid": power_valid, "fnirsi_diagnostics": fn_diag,
           "resource_metrics": mon.summary(),
           "params": int(sc.model.count_params()) if sc.kind == "keras" else None,
           "model_file_size_kb": float(sc.path.stat().st_size / 1024.0),
           "system_before": sys_before, "system_after": base.collect_system_info(),
           "temperature": {"before": temp_before, "after": base.get_cpu_temp()},
           "throttling": {"before": thr_before, "after": base.get_throttling_status()}}
    out = RESULT_DIR / f"s7a_{ds}_{model_name}_{res['engine']}.json"
    out.write_text(json.dumps(base.sanitize_for_json(res), indent=2, allow_nan=False))
    if board:
        log(f"  latency={res['latency_metrics']['latency_us_per_sample']:.2f} us/sampel | "
            f"PBUS={board['pbus_inference_w']:.3f} W | ΔP={board['p_delta_w']:.3f} W | "
            f"sps/W aktif={board['active_power_sps_per_w']:.1f}")
    del sc
    tf.keras.backend.clear_session(); gc.collect()
    return res
# ======== END: SATU RUN (STRUKTUR SAMA DENGAN run_one LAMA) ========


# ======== START: RINGKASAN ========
def summary_row(r):
    b = r.get("board_power_metrics") or {}
    d, ds_ = r["detection_full_test"], r["detection_measurement_subset"]
    lm = r["latency_metrics"]; rm = r.get("resource_metrics") or {}
    return {"dataset": r["dataset"], "model": r["model"], "engine": r["engine"],
            "f1_full": d["f1"], "precision_full": d["precision"], "recall_full": d["recall"],
            "fpr_full": d["fpr"], "auc_full": d["auc"], "ap_full": d["ap"],
            "f1_subset": ds_["f1"], "decision_agreement_vs_server": r["verification_vs_server"]["decision_agreement"],
            "max_abs_score_diff": r["verification_vs_server"]["max_abs_score_diff"],
            "latency_ms_per_sample": lm["latency_ms_per_sample"], "throughput_sps": lm["throughput_sps"],
            "pbus_idle_w": b.get("pbus_idle_w"), "pbus_inference_w": b.get("pbus_inference_w"),
            "p_delta_w": b.get("p_delta_w"), "active_power_sps_per_w": b.get("active_power_sps_per_w"),
            "delta_power_sps_per_w": b.get("delta_power_sps_per_w"),
            "board_energy_delta_mj_per_sample": b.get("board_energy_delta_mj_per_sample"),
            "process_cpu_mean_percent": (rm.get("process_cpu_percent") or {}).get("mean"),
            "process_rss_max_mb": (rm.get("process_rss_mb") or {}).get("max"),
            "temp_before": r["temperature"]["before"], "temp_after": r["temperature"]["after"],
            "throttling_after": r["throttling"]["after"], "params": r["params"],
            "power_valid": r.get("power_valid"),
            "model_file_size_kb": r.get("model_file_size_kb")}
# ======== END: RINGKASAN ========


# ======== START: MAIN ========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=["edge_iiotset", "ciciot2023", "all"])
    ap.add_argument("--model", default="all", choices=AE_MODELS + BASELINES + ["all"])
    ap.add_argument("--engine", default="keras", choices=["keras", "tflite"])
    ap.add_argument("--with_baselines", action="store_true")
    ap.add_argument("--repeat", type=int, default=50)
    ap.add_argument("--repeat_baseline", type=int, default=10)
    ap.add_argument("--n_per_class", type=int, default=10_000)
    ap.add_argument("--predict_batch", type=int, default=base.PREDICT_BATCH)
    ap.add_argument("--resource_interval", type=float, default=base.RESOURCE_SAMPLE_INTERVAL)
    ap.add_argument("--idle_sec", type=int, default=base.IDLE_MEASURE_SEC)
    ap.add_argument("--fnirsi_path", default=str(HOME_DIR / "fnirsi_logger.py"))
    ap.add_argument("--no_fnirsi", action="store_true")
    ap.add_argument("--pbus_idle", type=float, default=base.PBUS_IDLE_DEFAULT)
    ap.add_argument("--min_agreement", type=float, default=0.999)
    ap.add_argument("--allow_version_mismatch", action="store_true")
    ap.add_argument("--check_only", action="store_true", help="Hanya preflight + verifikasi skor (tanpa daya)")
    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="Ukur ulang walaupun hasil valid sudah ada")
    args = ap.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = ["edge_iiotset", "ciciot2023"] if args.dataset == "all" else [args.dataset]
    models = AE_MODELS + (BASELINES if args.with_baselines else []) if args.model == "all" else [args.model]
    base.print_system_warning()
    payloads = {}
    for ds in datasets:
        preflight(ds, args.allow_version_mismatch)
        P = load_payload(ds)
        P["subset"] = stratified_subset(P["y"], args.n_per_class)
        np.save(RESULT_DIR / f"measurement_subset_ids_{ds}.npy", P["ids"][P["subset"]])
        payloads[ds] = P

    if args.check_only:
        rows = []
        cdir = RESULT_DIR / "check"; cdir.mkdir(parents=True, exist_ok=True)
        for ds in datasets:
            P = payloads[ds]
            for m in models:
                sc = Scorer(ds, m, args.engine, args.predict_batch)
                s = sc.scores(P["Xs"]); r = P["ref"][m].astype(np.float64)
                t = float(P["thr"][m][OP])
                rel = np.abs(s - r) / np.maximum(np.abs(r), 1e-12)
                det = detection_metrics(s, P["y"], P["lt"], t)
                np.savez_compressed(cdir / f"scores_{ds}_{m}_{sc.kind}.npz", sample_ids=P["ids"], y=P["y"],
                                    rpi_scores=s, server_scores=r, threshold=t)
                row = {"dataset": ds, "model": m, "engine": sc.kind, "n": int(len(s)),
                       "max_abs_diff": float(np.max(np.abs(s - r))), "max_rel_diff": float(np.max(rel)),
                       "decision_agreement": float(np.mean((s > t) == (r > t))),
                       **{k: v for k, v in det.items() if k != "subtype_recall"}}
                rows.append(row)
                (cdir / f"detection_{ds}_{m}_{sc.kind}.json").write_text(
                    json.dumps(base.sanitize_for_json(det), indent=2, allow_nan=False))
                log(f"CHECK {ds} {m}: max|Δ|={row['max_abs_diff']:.2e} max relΔ={row['max_rel_diff']:.2e} "
                    f"keputusan sama={row['decision_agreement']:.6f} | F1={det['f1']:.4f} FPR={det['fpr']:.4f}")
                del sc; tf.keras.backend.clear_session(); gc.collect()
        out = cdir / f"check_summary_{args.engine}_{datetime.now():%Y%m%d_%H%M}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        (cdir / "environment.json").write_text(json.dumps(base.sanitize_for_json(base.collect_system_info()), indent=2))
        log(f"Check tersimpan: {out}")
        return

    fn = {"r": None, "idle": (args.pbus_idle, 0.0, 0), "rate": None}

    def start_fnirsi():
        if fn["r"]:
            try:
                fn["r"].stop()
            except Exception:
                pass
        fn["r"] = None
        rdr = base.FNIRSIReader(logger_path=args.fnirsi_path)
        rdr.start(); time.sleep(2)
        if not rdr.is_alive():
            raise RuntimeError("fnirsi_logger.py tidak berjalan")
        fn["idle"] = base.measure_idle_power(rdr, duration_sec=args.idle_sec)
        fn["rate"] = fn["idle"][2] / float(args.idle_sec)
        fn["r"] = rdr
        log(f"FNIRSI aktif | idle={fn['idle'][0]:.4f} W | laju={fn['rate']:.1f} Hz")

    results = []
    try:
        for ds in datasets:
            for m in models:
                engine = "sklearn" if m in BASELINES else args.engine
                jpath = RESULT_DIR / f"s7a_{ds}_{m}_{engine}.json"
                if jpath.exists() and not args.force:
                    old_r = json.loads(jpath.read_text())
                    if args.no_fnirsi or old_r.get("power_valid"):
                        log(f"SKIP {ds} {m}: hasil valid sudah ada")
                        results.append(old_r); continue
                res = None
                for attempt in range(args.max_retries + 1):
                    last = attempt == args.max_retries
                    try:
                        if not args.no_fnirsi and (fn["r"] is None or not fn["r"].is_alive()):
                            start_fnirsi()
                        res = run_one(ds, m, payloads[ds], args, fn, last_attempt=last)
                        break
                    except (FnirsiDropout, RuntimeError) as e:
                        log(f"[RETRY {attempt + 1}/{args.max_retries}] {ds} {m}: {e}")
                        try:
                            if fn["r"]:
                                fn["r"].stop()
                        except Exception:
                            pass
                        fn["r"] = None
                        time.sleep(5)
                        if last:
                            log(f"[GAGAL] {ds} {m}: daya tidak valid setelah {args.max_retries} percobaan")
                if res is not None:
                    results.append(res)
    finally:
        if fn["r"]:
            fn["r"].stop()
    df = pd.DataFrame([summary_row(r) for r in results])
    out = RESULT_DIR / f"s7a_summary_{args.engine}_{datetime.now():%Y%m%d_%H%M}.csv"
    df.to_csv(out, index=False)
    print(df[["dataset", "model", "power_valid", "f1_full", "fpr_full", "decision_agreement_vs_server", "latency_ms_per_sample",
              "throughput_sps", "p_delta_w", "active_power_sps_per_w", "delta_power_sps_per_w"]].to_string(index=False))
    log(f"Selesai | {out}")


if __name__ == "__main__":
    main()
# ======== END: MAIN ========
