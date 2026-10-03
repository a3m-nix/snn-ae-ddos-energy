# ======== START: HEADER ========
"""
S7b — PIPELINE END-TO-END DI RASPBERRY PI 5 (R2.10, R3.7)
PCAP -> TShark (field identik skrip ekstraksi training) -> IAT -> filter IPv4 ->
33 fitur -> float32 -> scaler -> model -> keputusan.

Mode:
  --parity : uji kesamaan fitur TShark RPi vs fitur training (sample.parquet) pada
             paket yang sama (source_row). Wajib lolos sebelum pengukuran.
  (default): pengukuran. Untuk tiap dataset: (1) TShark saja, (2) pipeline penuh per
             model. Normal (*_e2e) lalu serangan (*_e2e), diulang --repeat kali.
             Daya FNIRSI, idle, ResourceMonitor, sps/W memakai fungsi skrip lama
             (melalui S7a). Resume + retry FNIRSI seperti S7a. Jalankan dengan sudo.
"""
# ======== END: HEADER ========


# ======== START: IMPORT (S7a -> skrip lama) ========
import argparse
import importlib.util
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

HOME = Path("/home/a3m-nix/snn-ae-final")
_spec = importlib.util.spec_from_file_location("s7a", HOME / "s7a_rpi5_mixed_validation.py")
s7a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s7a)
base = s7a.base

import gc
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
# ======== END: IMPORT (S7a -> skrip lama) ========


# ======== START: KONFIGURASI ========
PCAP_DIR = HOME / "pcap-e2e"
PAYLOAD = HOME / "0-revision-payload"
REF_DIR = PAYLOAD / "e2e_reference"
RESULT = HOME / "0-revision-results" / "s7b"
AE = ["SNN_AE", "Dense_AE", "LSTM_AE", "CNN_AE"]
TSHARK_FIELDS = ["frame.len", "frame.time_epoch", "ip.version", "ip.hdr_len", "ip.dsfield.dscp", "ip.len",
                 "ip.id", "ip.frag_offset", "ip.flags.df", "ip.flags.mf", "ip.ttl", "ip.proto", "ip.src",
                 "ip.dst", "tcp.srcport", "tcp.dstport", "tcp.seq", "tcp.ack", "tcp.hdr_len", "tcp.len",
                 "tcp.window_size", "tcp.flags.fin", "tcp.flags.syn", "tcp.flags.reset", "tcp.flags.push",
                 "tcp.flags.ack", "tcp.flags.urg", "tcp.flags.ns", "tcp.flags.cwr", "udp.srcport",
                 "udp.dstport", "udp.length", "icmp.type", "icmp.code", "data.len"]
CHUNK = 10_000
# TShark 3.6.2 (versi server saat data training dibuat) bila tersedia; jika tidak, TShark sistem.
TSHARK_PINNED = Path("/opt/tshark-3.6.2/bin/tshark")
TSHARK = str(TSHARK_PINNED) if TSHARK_PINNED.exists() else "tshark"


def log(m):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [S7b] {m}", flush=True)


def tshark_provenance():
    v = subprocess.run([TSHARK, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    used_alias = {k: a for k, a in FIELD_ALIASES.items() if a in resolved_fields()}
    return {"tshark_binary": TSHARK, "tshark_version": v, "field_aliases_used": used_alias,
            "disabled_protocols_used": active_disabled_protocols()}


def count_packets(pcap):
    out = subprocess.run(["capinfos", "-c", "-M", str(pcap)], capture_output=True, text=True, check=True).stdout
    return int([l for l in out.splitlines() if "Number of packets" in l][0].split(":")[1].strip())


# Padanan nama field antar-versi Wireshark (bit yang sama, nama berbeda).
FIELD_ALIASES = {"tcp.flags.ns": "tcp.flags.ae"}
# Dissector baru di TShark 4.x yang tidak ada saat data training dibuat; dinonaktifkan
# agar payload kembali dicatat sebagai "data" (terbukti lewat parity check).
DISABLE_PROTOCOLS = ["hipercontracer"]
_RESOLVED = None


def resolved_fields():
    """Nama field TShark lokal; kolom output tetap memakai nama training."""
    global _RESOLVED
    if _RESOLVED is None:
        known = set(l.split("\t")[2] for l in subprocess.run([TSHARK, "-G", "fields"], capture_output=True,
                                                               text=True).stdout.splitlines()
                    if l.startswith("F\t") and len(l.split("\t")) > 2)
        _RESOLVED = []
        for f in TSHARK_FIELDS:
            if f in known:
                _RESOLVED.append(f)
            elif FIELD_ALIASES.get(f) in known:
                _RESOLVED.append(FIELD_ALIASES[f])
                log(f"Field {f} tidak ada di TShark lokal -> memakai {FIELD_ALIASES[f]}")
            else:
                raise SystemExit(f"Field {f} tidak dikenal TShark lokal dan tidak ada padanannya")
    return _RESOLVED


_DISABLE = None


def active_disabled_protocols():
    """Hanya nonaktifkan protokol yang memang ada di versi TShark terpasang."""
    global _DISABLE
    if _DISABLE is None and TSHARK == str(TSHARK_PINNED):
        _DISABLE = []          # versi identik server: perintah ekstraksi identik, tanpa penonaktifan
    if _DISABLE is None:
        protos = set(l.split("\t")[2].strip() for l in subprocess.run([TSHARK, "-G", "protocols"],
                     capture_output=True, text=True).stdout.splitlines() if len(l.split("\t")) > 2)
        _DISABLE = [p for p in DISABLE_PROTOCOLS if p in protos]
    return _DISABLE


def rename_columns(df):
    """Kembalikan nama kolom ke nama field training."""
    m = {v: k for k, v in FIELD_ALIASES.items()}
    return df.rename(columns=m)


def tshark_cmd(pcap, count=None):
    """Identik dengan 1.extract_pcap_to_csv.sh (+ -l untuk streaming)."""
    cmd = [TSHARK, "-r", str(pcap), "-n", "-N", "m", "-T", "fields"]
    for proto in active_disabled_protocols():
        cmd += ["--disable-protocol", proto]
    for f in resolved_fields():
        cmd += ["-e", f]
    cmd += ["-E", "header=y", "-E", "separator=,", "-E", "quote=d", "-E", "occurrence=f", "-l"]
    if count:
        cmd += ["-c", str(count)]
    return cmd
# ======== END: KONFIGURASI ========


# ======== START: KONVERSI FITUR (DIVALIDASI OLEH PARITY) ========
def to_num(s):
    """String TShark -> float. Menangani angka biasa, heksadesimal (0x..), dan True/False."""
    s = s.astype("string").str.strip()
    out = pd.to_numeric(s, errors="coerce")
    miss = out.isna() & s.notna() & (s != "")
    if miss.any():
        v = s[miss]
        hx = v.str.match(r"^0x[0-9a-fA-F]+$", na=False)
        if hx.any():
            out.loc[v[hx].index] = v[hx].map(lambda x: int(x, 16))
        bl = v.str.lower().isin(["true", "false"])
        if bl.any():
            out.loc[v[bl].index] = (v[bl].str.lower() == "true").astype(float)
    return out.astype("float64")


def valid_ipv4(df):
    ok = pd.Series(True, index=df.index)
    for c in ["ip.src", "ip.dst"]:
        v = df[c].astype("string").str.strip()
        ok &= ~(v.isna() | v.str.lower().isin(["", "0", "nan", "none", "null"]))
    return ok.to_numpy()


def build_features(chunk, last_t, features):
    """IAT dihitung pada SEMUA paket (sebelum filter IPv4), seperti training."""
    t = to_num(chunk["frame.time_epoch"]).to_numpy()
    iat = np.diff(t, prepend=last_t if last_t is not None else np.nan)
    num = {c: to_num(chunk[c]).to_numpy() for c in features if c != "inter_arrival_time"}
    num["inter_arrival_time"] = iat
    X = np.column_stack([num[c] for c in features])
    keep = valid_ipv4(chunk)
    return X, keep, (t[-1] if len(t) else last_t)
# ======== END: KONVERSI FITUR (DIVALIDASI OLEH PARITY) ========


# ======== START: MODE PARITY ========
def model_parity(ds, kind, Xr, R):
    """Skor fitur RPi vs fitur training dengan model payload; bandingkan keputusan p99."""
    scaler = joblib.load(PAYLOAD / ds / "scaler.pkl")
    thr_all = json.loads((PAYLOAD / ds / "thresholds.json").read_text())
    Xa = scaler.transform(Xr.astype(np.float32)).astype(np.float32)
    Xb = scaler.transform(R.astype(np.float32)).astype(np.float32)
    out = []
    for m in AE + ["IF", "OCSVM", "LOF"]:
        sc = s7a.Scorer(ds, m, "keras", 256)
        a, b = sc.scores(Xa), sc.scores(Xb)
        t = float(thr_all[m]["99"])
        out.append({"dataset": ds, "file": kind, "model": m, "n": len(a),
                    "decision_agreement": float(np.mean((a > t) == (b > t))),
                    "n_decision_diff": int(np.sum((a > t) != (b > t))),
                    "max_abs_score_diff": float(np.max(np.abs(a - b)))})
        del sc
        tf.keras.backend.clear_session(); gc.collect()
    return out


def run_parity(features):
    rows, mrows = [], []
    for ds in ["edge_iiotset", "ciciot2023"]:
        for kind in ["normal", "attack"]:
            ref = pd.read_parquet(REF_DIR / f"ref_{ds}_{kind}.parquet")
            n = int(ref.source_row.max()) + 1
            proc = subprocess.run(tshark_cmd(PCAP_DIR / ds / f"{kind}.pcap", count=n),
                                  capture_output=True, text=True)
            if proc.returncode != 0:
                raise SystemExit(f"TShark gagal: {proc.stderr.strip()[:500]}")
            from io import StringIO
            df = rename_columns(pd.read_csv(StringIO(proc.stdout), dtype=str, keep_default_na=True))
            X, keep, _ = build_features(df, None, features)
            Xr = X[ref.source_row.to_numpy()]
            Xr[~np.isfinite(Xr)] = 0.0
            R = ref[features].to_numpy(dtype=np.float64)
            R[~np.isfinite(R)] = 0.0
            for j, c in enumerate(features):
                tol = 1e-6 if c == "inter_arrival_time" else 1e-9
                bad = ~np.isclose(Xr[:, j], R[:, j], rtol=1e-9, atol=tol)
                rows.append({"dataset": ds, "file": kind, "feature": c, "n": len(R), "mismatch": int(bad.sum()),
                             "example_rpi": Xr[bad, j][:3].tolist(), "example_train": R[bad, j][:3].tolist()})
            mrows.extend(model_parity(ds, kind, Xr, R))
            ipv4_ok = bool(keep[ref.source_row.to_numpy()].all())
            nbad = sum(r["mismatch"] for r in rows if r["dataset"] == ds and r["file"] == kind)
            log(f"PARITY {ds}/{kind}: {len(R)} paket | total mismatch={nbad} | semua lolos filter IPv4={ipv4_ok}")
    out = pd.DataFrame(rows)
    RESULT.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULT / "parity_check.csv", index=False)
    (RESULT / "parity_provenance.json").write_text(json.dumps(tshark_provenance(), indent=2))
    mp = pd.DataFrame(mrows)
    mp.to_csv(RESULT / "parity_models.csv", index=False)
    print(mp.round(6).to_string(index=False))
    bad = out[out.mismatch > 0]
    if len(bad):
        print(bad[["dataset", "file", "feature", "mismatch", "example_rpi", "example_train"]].to_string(index=False))
        log("PARITY FITUR: ada selisih (lihat tabel). Periksa dampak keputusan di tabel model di atas.")
    else:
        log("PARITY LOLOS: fitur RPi identik dengan training.")
# ======== END: MODE PARITY ========


# ======== START: SATU EKSEKUSI PIPELINE ========
def pipeline_once(ds, scorer, scaler, features, thr, tshark_only=False):
    """Satu lintasan normal_e2e lalu attack_e2e. Mengembalikan waktu per tahap dan skor."""
    st = {"t_parse": 0.0, "t_scale": 0.0, "t_infer": 0.0, "n_packets": 0, "n_ipv4": 0}
    scores, labels = [], []
    t_all = time.perf_counter()
    for kind, y in [("normal", 0), ("attack", 1)]:
        pcap = PCAP_DIR / ds / f"{kind}_e2e.pcap"
        if tshark_only:
            r = subprocess.run(tshark_cmd(pcap), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if r.returncode != 0:
                raise SystemExit(f"TShark gagal ({pcap.name}): {r.stderr.strip()[:500]}")
            continue
        proc = subprocess.Popen(tshark_cmd(pcap), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, bufsize=1 << 20)
        last_t = None
        t0 = time.perf_counter()
        for chunk in pd.read_csv(proc.stdout, dtype=str, chunksize=CHUNK):
            chunk = rename_columns(chunk)
            X, keep, last_t = build_features(chunk, last_t, features)
            X = X[keep]
            X[~np.isfinite(X)] = 0.0
            X = X.astype(np.float32)
            t1 = time.perf_counter(); st["t_parse"] += t1 - t0
            Xs = scaler.transform(X).astype(np.float32)
            t2 = time.perf_counter(); st["t_scale"] += t2 - t1
            s = scorer.scores(Xs) if len(Xs) else np.empty(0)
            t3 = time.perf_counter(); st["t_infer"] += t3 - t2
            scores.append(s); labels.append(np.full(len(s), y))
            st["n_packets"] += len(chunk); st["n_ipv4"] += int(keep.sum())
            t0 = time.perf_counter()
        if proc.wait() != 0:
            raise SystemExit(f"TShark gagal ({pcap.name}): {proc.stderr.read().strip()[:500]}")
    st["t_total"] = time.perf_counter() - t_all
    s = np.concatenate(scores) if scores else np.empty(0)
    yv = np.concatenate(labels) if labels else np.empty(0)
    return st, s, yv
# ======== END: SATU EKSEKUSI PIPELINE ========


# ======== START: SATU RUN TERUKUR (DAYA + RETRY) ========
class FnirsiDropout(RuntimeError):
    pass


def measured_run(ds, name, args, fn, last_attempt, features, scaler, thr):
    scorer = None if name == "TSHARK_ONLY" else s7a.Scorer(ds, name, "keras", 256)
    if scorer:   # warm-up
        scorer.scores(np.zeros((512, len(features)), dtype=np.float32))
    for i in range(base.COUNTDOWN_SEC, 0, -1):
        print(f"  Start in {i}...", flush=True); time.sleep(1)
    if fn["r"]:
        fn["r"].clear_readings()
    mon = base.ResourceMonitor(interval_sec=base.RESOURCE_SAMPLE_INTERVAL)
    reps, s_first, y_first = [], None, None
    mon.start(); t0 = time.perf_counter()
    try:
        for k in range(args.repeat):
            st, s, yv = pipeline_once(ds, scorer, scaler, features, thr, tshark_only=scorer is None)
            reps.append(st)
            if k == 0:
                s_first, y_first = s, yv
            log(f"  {name} rep {k + 1}/{args.repeat}: {st['t_total']:.1f} s")
    finally:
        elapsed = time.perf_counter() - t0
        mon.stop()
    n_pk = reps[0]["n_packets"] if scorer else sum(count_packets(PCAP_DIR / ds / f"{k}_e2e.pcap")
                                                   for k in ("normal", "attack"))
    board, valid, diag = None, None, None
    if fn["r"]:
        r = fn["r"].get_readings()
        rate = len(r) / max(elapsed, 1e-9)
        alive = fn["r"].is_alive()
        valid = bool(alive and fn["rate"] and rate >= 0.5 * fn["rate"] and len(r) > 0)
        diag = {"alive_after": alive, "readings": len(r), "rate_hz": rate, "idle_rate_hz": fn["rate"]}
        if not valid and not last_attempt:
            raise FnirsiDropout(f"FNIRSI tidak valid: {diag}")
        if r:
            tot = float(np.mean([x["t_total"] for x in reps])) * 1000.0
            npk = n_pk
            board = base.calc_board_metrics(float(np.mean(r)), float(np.std(r)), len(r), fn["idle"][0],
                                            fn["idle"][1], fn["idle"][2], tot, float(np.std([x["t_total"] for x in reps]) * 1000.0), npk)
    res = {"dataset": ds, "name": name, "repeat": args.repeat, "power_valid": valid, "fnirsi_diagnostics": diag,
           "tshark": tshark_provenance(),
           "board_power_metrics": board, "resource_metrics": mon.summary(), "elapsed_s": elapsed}
    if scorer:
        m = lambda k: float(np.mean([x[k] for x in reps]))
        npk, n4 = reps[0]["n_packets"], reps[0]["n_ipv4"]
        res["stages"] = {"n_packets": npk, "n_ipv4": n4, "total_s": m("t_total"),
                         "parse_features_s": m("t_parse"), "scale_s": m("t_scale"), "infer_s": m("t_infer"),
                         "us_per_packet_total": m("t_total") / npk * 1e6,
                         "us_per_packet_infer": m("t_infer") / npk * 1e6,
                         "packets_per_s": npk / m("t_total")}
        res["detection"] = s7a.detection_metrics(s_first, y_first, np.where(y_first == 1, "syn", "normal"), thr)
        np.savez_compressed(RESULT / f"scores_{ds}_{name}.npz", scores=s_first, y=y_first)
    else:
        ts = float(np.mean([x["t_total"] for x in reps]))
        res["stages"] = {"tshark_only_s": ts, "n_packets": n_pk, "us_per_packet_tshark": ts / n_pk * 1e6,
                         "packets_per_s": n_pk / ts}
    out = RESULT / f"s7b_{ds}_{name}.json"
    out.write_text(json.dumps(base.sanitize_for_json(res), indent=2, allow_nan=False))
    del scorer; tf.keras.backend.clear_session(); gc.collect()
    return res
# ======== END: SATU RUN TERUKUR (DAYA + RETRY) ========


# ======== START: MAIN ========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--dataset", default="all", choices=["edge_iiotset", "ciciot2023", "all"])
    ap.add_argument("--model", default="all", choices=AE + ["TSHARK_ONLY", "all"])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--idle_sec", type=int, default=base.IDLE_MEASURE_SEC)
    ap.add_argument("--fnirsi_path", default=str(HOME / "fnirsi_logger.py"))
    ap.add_argument("--no_fnirsi", action="store_true")
    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    RESULT.mkdir(parents=True, exist_ok=True)
    features = json.loads((PAYLOAD / "edge_iiotset" / "features.json").read_text())
    log(f"TShark: {tshark_provenance()}")
    if args.parity:
        run_parity(features); return

    datasets = ["edge_iiotset", "ciciot2023"] if args.dataset == "all" else [args.dataset]
    names = ["TSHARK_ONLY"] + AE if args.model == "all" else [args.model]
    for ds in datasets:
        s7a.preflight(ds, False)
    base.print_system_warning()
    fn = {"r": None, "idle": (base.PBUS_IDLE_DEFAULT, 0.0, 0), "rate": None}

    def start_fnirsi():
        if fn["r"]:
            try: fn["r"].stop()
            except Exception: pass
        rdr = base.FNIRSIReader(logger_path=args.fnirsi_path); rdr.start(); time.sleep(2)
        if not rdr.is_alive():
            raise RuntimeError("fnirsi_logger.py tidak berjalan")
        fn["idle"] = base.measure_idle_power(rdr, duration_sec=args.idle_sec)
        fn["rate"] = fn["idle"][2] / float(args.idle_sec); fn["r"] = rdr

    results = []
    try:
        for ds in datasets:
            scaler = joblib.load(PAYLOAD / ds / "scaler.pkl")
            thr_all = json.loads((PAYLOAD / ds / "thresholds.json").read_text())
            for name in names:
                jp = RESULT / f"s7b_{ds}_{name}.json"
                if jp.exists() and not args.force:
                    old = json.loads(jp.read_text())
                    if args.no_fnirsi or old.get("power_valid"):
                        log(f"SKIP {ds} {name}"); results.append(old); continue
                thr = float(thr_all[name]["99"]) if name in thr_all else None
                for attempt in range(args.max_retries + 1):
                    last = attempt == args.max_retries
                    try:
                        if not args.no_fnirsi and (fn["r"] is None or not fn["r"].is_alive()):
                            start_fnirsi()
                        log(f"=== {ds} | {name} ===")
                        results.append(measured_run(ds, name, args, fn, last, features, scaler, thr)); break
                    except (FnirsiDropout, RuntimeError) as e:
                        log(f"[RETRY {attempt + 1}/{args.max_retries}] {ds} {name}: {e}")
                        try:
                            if fn["r"]: fn["r"].stop()
                        except Exception: pass
                        fn["r"] = None; time.sleep(5)
    finally:
        if fn["r"]:
            fn["r"].stop()

    rows = []
    for r in results:
        b, sgs, d = r.get("board_power_metrics") or {}, r.get("stages") or {}, r.get("detection") or {}
        rows.append({"dataset": r["dataset"], "name": r["name"], "valid": r.get("power_valid"),
                     "us_pkt_total": sgs.get("us_per_packet_total"), "us_pkt_infer": sgs.get("us_per_packet_infer"),
                     "pkt_per_s": sgs.get("packets_per_s"), "us_pkt_tshark_only": sgs.get("us_per_packet_tshark"),
                     "total_s": sgs.get("total_s"), "parse_s": sgs.get("parse_features_s"),
                     "scale_s": sgs.get("scale_s"), "infer_s": sgs.get("infer_s"),
                     "pbus_w": b.get("pbus_inference_w"), "dP_w": b.get("p_delta_w"),
                     "mJ_pkt_delta": b.get("board_energy_delta_mj_per_sample"),
                     "f1": d.get("f1"), "recall": d.get("recall"), "fpr": d.get("fpr")})
    df = pd.DataFrame(rows)
    out = RESULT / f"s7b_summary_{datetime.now():%Y%m%d_%H%M}.csv"
    df.to_csv(out, index=False)
    print(df.round(4).to_string(index=False))
    log(f"Selesai | {out}")


if __name__ == "__main__":
    main()
# ======== END: MAIN ========
