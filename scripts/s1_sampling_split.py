# ======== START: HEADER ========
"""
S1 — SAMPLING SEIMBANG DAN SPLIT BEKU (GROUP-AWARE)
Revisi Paper 1 SNN-AE — Scientific Reports

Tujuan:
  1) Sampling seimbang mengikuti protokol naskah (cap 50.000 per subtipe
     serangan; jumlah normal = jumlah serangan).
  2) Membentuk seluruh split yang dibekukan untuk S2–S7:
     - holdout tuning 15% vs evaluasi 85%
     - split tuning 60/20/20 (train/calib/dev) di dalam holdout
     - CV 3 seed x 10 fold, dengan validation-normal 20% per fold
  3) Membuktikan tidak ada kebocoran data melalui assert otomatis.

Kebijakan split:
  - Trafik NORMAL dibagi per grup flow dengan group k-fold terstratifikasi
    yang sadar ukuran grup (strata = source_file). Hanya trafik normal yang dipakai untuk fitting, scaler,
    threshold, dan early stopping.
  - Trafik SERANGAN tidak pernah dipakai untuk fitting; dibagi terstratifikasi
    per subtipe pada level paket. Tumpang-tindih grup serangan antar-partisi
    dilaporkan secara deskriptif.

Output: 0-snn-ae-revision/02-splits/{full|validation}/<dataset>/
"""
# ======== END: HEADER ========


# ======== START: IMPORT ========
import hashlib
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

warnings.filterwarnings("ignore", category=UserWarning)
# ======== END: IMPORT ========


# ======== START: KONFIGURASI ========
VALIDATION_MODE = False          # True = probe cepat; False = run penuh
VALIDATION_CAP = 2_000          # cap per subtipe pada mode validasi
VALIDATION_SEEDS = [0]          # seed pada mode validasi
BATCH_SIZE = 500_000

BASE = Path("/home/jupyteruser/nix-ext-backup/0-eksperimen-final")
REV_DIR = BASE / "dataset-revision-33"
WORK = BASE / "0-snn-ae-revision"
CONFIG_PATH = WORK / "00-config" / "config_revision.json"
OUT_ROOT = WORK / "02-splits" / ("validation" if VALIDATION_MODE else "full")

FEATURES = [
    "frame.len", "ip.version", "ip.hdr_len", "ip.dsfield.dscp",
    "ip.len", "ip.id", "ip.frag_offset", "ip.flags.df",
    "ip.flags.mf", "ip.ttl", "ip.proto", "tcp.srcport",
    "tcp.dstport", "tcp.seq", "tcp.ack", "tcp.hdr_len", "tcp.len",
    "tcp.window_size", "tcp.flags.fin", "tcp.flags.syn",
    "tcp.flags.reset", "tcp.flags.push", "tcp.flags.ack",
    "tcp.flags.urg", "tcp.flags.ns", "tcp.flags.cwr",
    "udp.srcport", "udp.dstport", "udp.length", "icmp.type",
    "icmp.code", "data.len", "inter_arrival_time",
]
assert len(FEATURES) == 33

# Bagian konfigurasi yang menjadi tanggung jawab S1.
# Tahap berikutnya menambahkan bagiannya sendiri ke file yang sama.
CONFIG_SECTIONS = {
    "data": {
        "datasets": ["edge_iiotset", "ciciot2023"],
        "revision_parquet": {
            "edge_iiotset": str(REV_DIR / "edge_iiotset_bidir_metadata.parquet"),
            "ciciot2023": str(REV_DIR / "ciciot2023_bidir_metadata.parquet"),
        },
        "manifest": {
            "edge_iiotset": str(REV_DIR / "edge_iiotset_manifest.json"),
            "ciciot2023": str(REV_DIR / "ciciot2023_manifest.json"),
        },
        "fragment_pairs": {
            "ciciot2023": str(REV_DIR / "ciciot2023_fragment_pairs.json"),
        },
        "features": FEATURES,
        "normal_labels": ["normal", "benign"],
        "attack_labels": ["attack"],
        "cap_per_subtype": 50_000,
        "expected_subtypes": {"edge_iiotset": 4, "ciciot2023": 15},
        "scaler_type": {"edge_iiotset": "quantile", "ciciot2023": "minmax"},
        "ip_scope": "IPv4 only (packets without IPv4 src/dst excluded)",
    },
    "split": {
        "random_state": 42,
        "seeds": [0, 1, 2],
        "n_folds": 10,
        "holdout_frac": 0.15,
        "holdout_normal_splits": 20,     # 3 dari 20 fold grup = 15%
        "holdout_normal_take": 3,
        "tuning_normal_splits": 5,       # 3/1/1 fold = 60/20/20
        "tuning_train_folds": 3,
        "val_ratio": 0.20,               # 1 dari 5 fold grup
        "normal_policy": "size-aware stratified group k-fold (largest-first "
                         "greedy), group=flow, strata=source_file",
        "attack_policy": "stratified by subtype at packet level (never used for fitting)",
        "cv_split_seed": "random_state + seed_value",
        "val_split_seed": "random_state + 1000*seed_idx + fold",
    },
}
# ======== END: KONFIGURASI ========


# ======== START: UTILITAS ========
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def section_hash(section):
    payload = json.dumps(section, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def atomic_write_text(path, text):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path, data):
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False,
                                       default=str))


def atomic_write_parquet(path, df):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp,
                   compression="snappy")
    tmp.replace(path)


def atomic_write_npz(path, arrays):
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def array_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).view(np.uint8))
    return h.hexdigest()[:16]


def sync_config(sections):
    """
    Menulis bagian konfigurasi S1 bila belum ada.
    Jika sudah ada tetapi berbeda, proses dihentikan agar run tidak tercampur.
    """
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg = (json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
           if CONFIG_PATH.exists() else {})
    cfg.setdefault("hashes", {})

    for name, sec in sections.items():
        if name in cfg:
            if section_hash(cfg[name]) != section_hash(sec):
                raise RuntimeError(
                    f"Konfigurasi bagian '{name}' berbeda dengan yang tersimpan "
                    f"di {CONFIG_PATH}. Gunakan folder run baru atau "
                    f"kembalikan nilai konfigurasi."
                )
        else:
            cfg[name] = sec
        cfg["hashes"][name] = section_hash(cfg[name])

    cfg.setdefault("created_at", now_str())
    atomic_write_json(CONFIG_PATH, cfg)
    return cfg
# ======== END: UTILITAS ========


# ======== START: PASS 1 — LABEL DAN SAMPLING ========
def load_label_arrays(path, normal_labels, attack_labels):
    """Membaca kolom label sebagai kategori agar hemat memori."""
    # ignore_metadata=True: metadata pandas (dtype "string") pada parquet
    # revisi menimpa strings_to_categorical; tanpa ini kolom tidak menjadi
    # kategori.
    df = pq.read_table(path, columns=["label", "label_type"]).to_pandas(
        strings_to_categorical=True, ignore_metadata=True
    )
    for column in ["label", "label_type"]:
        if not isinstance(df[column].dtype, pd.CategoricalDtype):
            df[column] = df[column].astype("category")
    lab = df["label"]
    cats = pd.Index(lab.cat.categories)
    low = pd.Index(cats.astype(str)).str.strip().str.lower()

    unknown = ~low.isin(normal_labels + attack_labels)
    if unknown.any():
        raise ValueError(f"Label tidak dikenal: {list(cats[unknown])}")

    is_normal = lab.isin(list(cats[low.isin(normal_labels)])).to_numpy()

    lt = df["label_type"]
    if not isinstance(lt.dtype, pd.CategoricalDtype):
        lt = lt.astype("category")
    lt_codes = lt.cat.codes.to_numpy()
    lt_cats = pd.Index(lt.cat.categories.astype(str))

    if (lt_codes[~is_normal] < 0).any():
        raise ValueError("Ada paket serangan tanpa label_type")

    return is_normal, lt_codes, lt_cats


def balanced_sampling(is_normal, lt_codes, lt_cats, cap, rs):
    """
    Replikasi logika load_balanced_data (v8.3):
    subtipe diurutkan berdasarkan nama, masing-masing diambil min(cap, n)
    tanpa pengembalian; normal = jumlah serangan; lalu diacak.
    """
    rng = np.random.RandomState(rs)
    att_pos = np.flatnonzero(~is_normal)
    att_codes = lt_codes[att_pos]

    subtypes = sorted({lt_cats[c] for c in np.unique(att_codes)})
    chosen_att, rows = [], []
    for name in subtypes:
        code = lt_cats.get_loc(name)
        pos = att_pos[att_codes == code]
        take = min(cap, len(pos))
        chosen_att.append(rng.choice(pos, size=take, replace=False))
        rows.append({"subtype": name, "available_samples": int(len(pos)),
                     "sampled_samples": int(take)})

    chosen_att = np.concatenate(chosen_att)
    norm_pos = np.flatnonzero(is_normal)
    n_normal = min(len(chosen_att), len(norm_pos))
    chosen_norm = rng.choice(norm_pos, size=n_normal, replace=False)

    combined = np.concatenate([chosen_norm, chosen_att])
    perm = np.random.RandomState(rs).permutation(len(combined))
    return combined[perm], pd.DataFrame(rows), subtypes
# ======== END: PASS 1 — LABEL DAN SAMPLING ========


# ======== START: PASS 2 — BACA BARIS TERPILIH DAN KUNCI GRUP ========
def read_selected_rows(path, positions, columns):
    """Membaca baris pada posisi global tertentu secara bertahap."""
    order = np.argsort(positions, kind="stable")
    sorted_pos = positions[order]
    pf = pq.ParquetFile(path)

    parts, offset = [], 0
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=columns):
        n = batch.num_rows
        lo = np.searchsorted(sorted_pos, offset, side="left")
        hi = np.searchsorted(sorted_pos, offset + n, side="left")
        if hi > lo:
            local = pa.array(sorted_pos[lo:hi] - offset)
            parts.append(batch.take(local).to_pandas())
        offset += n

    df = pd.concat(parts, ignore_index=True)
    if len(df) != len(positions):
        raise RuntimeError("Jumlah baris terbaca tidak sama dengan jumlah terpilih")

    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    df = df.iloc[inverse].reset_index(drop=True)
    df.insert(0, "row_idx", positions.astype(np.int64))
    return df


def build_group_key(df, dataset, fragment_pairs):
    """
    Edge-IIoTset: FLOW|flow_id.
    CICIoT2023  : PAIR|proto|ipA|ipB untuk pasangan yang memiliki fragmen
                  (aturan audit), selain itu FLOW|flow_id.
    """
    flow = "FLOW|" + df["flow_id"].astype(str)
    if dataset != "ciciot2023":
        return flow

    s = df["ip.src"].astype("string").str.strip()
    d = df["ip.dst"].astype("string").str.strip()
    fwd = s.le(d)
    first = s.where(fwd, d)
    second = d.where(fwd, s)
    proto = pd.to_numeric(df["ip.proto"]).astype("int64").astype(str)
    pair = (proto + "|" + first + "|" + second).astype(str)
    use_pair = pair.isin(fragment_pairs)
    return flow.where(~use_pair, "PAIR|" + pair)
# ======== END: PASS 2 — BACA BARIS TERPILIH DAN KUNCI GRUP ========


# ======== START: PEMBENTUKAN SPLIT ========
def group_folds(ids, strata_all, groups_all, n_splits, seed):
    """
    Group k-fold terstratifikasi yang sadar ukuran grup.
    - Grup diurutkan dari yang terbesar; urutan grup berukuran sama diacak.
    - Setiap grup ditempatkan pada fold dengan beban relatif (per strata)
      terkecil setelah penambahan; seri dipecah secara acak.
    - Label fold diacak per seed.
    Mengembalikan daftar fold berisi ids.
    """
    rng = np.random.RandomState(seed)
    g_codes, g_inv = np.unique(groups_all[ids], return_inverse=True)
    s_codes, s_inv = np.unique(strata_all[ids], return_inverse=True)
    n_g, n_s = len(g_codes), len(s_codes)
    if n_g < n_splits:
        raise RuntimeError(f"Jumlah grup ({n_g}) < jumlah fold ({n_splits})")

    counts = np.zeros((n_g, n_s))
    np.add.at(counts, (g_inv, s_inv), 1.0)
    target = counts.sum(axis=0) / n_splits
    sizes = counts.sum(axis=1)
    order = np.lexsort((rng.random_sample(n_g), -sizes))

    load = np.zeros((n_splits, n_s))
    assign = np.empty(n_g, dtype=np.int64)
    for gi in order:
        c = counts[gi]
        # Biaya inkremental: hanya strata milik grup ini yang dihitung,
        # sehingga strata langka tidak "mengunci" sebuah fold.
        delta = (((2.0 * load + c) * c) / target ** 2).sum(axis=1)
        best = np.flatnonzero(np.isclose(delta, delta.min()))
        if len(best) > 1:
            total = load[best].sum(axis=1)
            best = best[np.isclose(total, total.min())]
        k = best[rng.randint(len(best))]
        assign[gi] = k
        load[k] += counts[gi]

    assign = rng.permutation(n_splits)[assign]
    fold_of_sample = assign[g_inv]
    return [ids[fold_of_sample == k] for k in range(n_splits)]


def build_splits(y, strata_norm, strata_att, groups, seeds, sp):
    rs = sp["random_state"]
    ids = np.arange(len(y))
    norm_ids = ids[y == 0]
    att_ids = ids[y == 1]
    arrays = {}

    # ---- Holdout 15% vs evaluasi 85% ----
    folds20 = group_folds(norm_ids, strata_norm, groups,
                          sp["holdout_normal_splits"], rs)
    k = sp["holdout_normal_take"]
    hold_norm = np.sort(np.concatenate(folds20[:k]))
    eval_norm = np.sort(np.concatenate(folds20[k:]))

    sss = StratifiedShuffleSplit(n_splits=1, train_size=sp["holdout_frac"],
                                 random_state=rs)
    h_loc, e_loc = next(sss.split(np.zeros(len(att_ids)), strata_att[att_ids]))
    hold_att = np.sort(att_ids[h_loc])
    eval_att = np.sort(att_ids[e_loc])

    arrays["holdout"] = np.sort(np.concatenate([hold_norm, hold_att]))
    arrays["eval"] = np.sort(np.concatenate([eval_norm, eval_att]))

    # ---- Split tuning 60/20/20 di dalam holdout ----
    tfolds = group_folds(hold_norm, strata_norm, groups,
                         sp["tuning_normal_splits"], rs)
    ntr = sp["tuning_train_folds"]
    tune_train = np.sort(np.concatenate(tfolds[:ntr]))
    tune_calib = np.sort(tfolds[ntr])
    dev_norm = np.sort(np.concatenate(tfolds[ntr + 1:]))
    rng = np.random.RandomState(rs)
    n_dev_att = min(len(dev_norm), len(hold_att))
    dev_att = np.sort(rng.choice(hold_att, size=n_dev_att, replace=False))
    arrays["tune_train"] = tune_train
    arrays["tune_calib"] = tune_calib
    arrays["tune_dev"] = np.sort(np.concatenate([dev_norm, dev_att]))

    # ---- CV per seed ----
    n_val_splits = int(round(1.0 / sp["val_ratio"]))
    for seed_idx, seed_value in enumerate(seeds):
        split_seed = rs + int(seed_value)
        nfolds = group_folds(eval_norm, strata_norm, groups,
                             sp["n_folds"], split_seed)
        skf = StratifiedKFold(n_splits=sp["n_folds"], shuffle=True,
                              random_state=split_seed)
        afolds = [eval_att[te] for _, te in skf.split(
            np.zeros(len(eval_att)), strata_att[eval_att])]

        for fold in range(sp["n_folds"]):
            test = np.sort(np.concatenate([nfolds[fold], afolds[fold]]))
            tv_norm = np.sort(np.concatenate(
                [nfolds[j] for j in range(sp["n_folds"]) if j != fold]))
            vfolds = group_folds(tv_norm, strata_norm, groups, n_val_splits,
                                 rs + 1000 * seed_idx + fold)
            val = np.sort(vfolds[0])
            train = np.sort(np.concatenate(vfolds[1:]))
            key = f"s{seed_idx}_f{fold}"
            arrays[f"{key}_train"] = train
            arrays[f"{key}_val"] = val
            arrays[f"{key}_test"] = test

    return arrays
# ======== END: PEMBENTUKAN SPLIT ========


# ======== START: PEMERIKSAAN KEBOCORAN ========
def check_leakage(arrays, y, groups, seeds, n_folds):
    """Assert keras; pelanggaran menghentikan proses."""
    def ng(ids):  # grup normal
        return np.unique(groups[ids[y[ids] == 0]])

    def ag(ids):  # grup serangan
        return np.unique(groups[ids[y[ids] == 1]])

    def disjoint(a, b, name):
        inter = np.intersect1d(a, b)
        if len(inter):
            raise RuntimeError(f"KEBOCORAN: {name} ({len(inter)} irisan)")

    report = {}
    hold, ev = arrays["holdout"], arrays["eval"]
    disjoint(hold, ev, "sampel holdout vs eval")
    disjoint(ng(hold), ng(ev), "grup normal holdout vs eval")

    tt, tc, td = arrays["tune_train"], arrays["tune_calib"], arrays["tune_dev"]
    for a, b, n in [(tt, tc, "tune_train vs tune_calib"),
                    (tt, td, "tune_train vs tune_dev"),
                    (tc, td, "tune_calib vs tune_dev")]:
        disjoint(a, b, f"sampel {n}")
        disjoint(ng(a), ng(b), f"grup normal {n}")
    if (y[tt] != 0).any() or (y[tc] != 0).any():
        raise RuntimeError("KEBOCORAN: tune_train/tune_calib memuat serangan")
    for a in (tt, tc, td):
        if not np.isin(a, hold).all():
            raise RuntimeError("KEBOCORAN: split tuning keluar dari holdout")

    ev_att = ev[y[ev] == 1]
    ho_att = hold[y[hold] == 1]
    report["attack_group_overlap_holdout_eval_share"] = float(
        np.isin(groups[ev_att], ag(ho_att)).mean()) if len(ev_att) else 0.0

    mixed_shares = []
    for s in range(len(seeds)):
        coverage = np.zeros(len(y), dtype=np.int32)
        for f in range(n_folds):
            k = f"s{s}_f{f}"
            tr, va, te = arrays[f"{k}_train"], arrays[f"{k}_val"], arrays[f"{k}_test"]
            if (y[tr] != 0).any() or (y[va] != 0).any():
                raise RuntimeError(f"KEBOCORAN: {k} train/val memuat serangan")
            for a, b, n in [(tr, va, "train vs val"), (tr, te, "train vs test"),
                            (va, te, "val vs test")]:
                disjoint(a, b, f"sampel {k} {n}")
                disjoint(ng(a), ng(b), f"grup normal {k} {n}")
            for part in (tr, va, te):
                if np.isin(part, hold).any():
                    raise RuntimeError(f"KEBOCORAN: {k} memuat sampel holdout")
            coverage[te] += 1
            te_att = te[y[te] == 1]
            if len(te_att):
                mixed_shares.append(float(
                    np.isin(groups[te_att], ng(np.concatenate([tr, va]))).mean()))
        if not (coverage[ev] == 1).all() or coverage[hold].any():
            raise RuntimeError(f"Seed {s}: sampel eval tidak diuji tepat satu kali")

    report["test_attack_in_train_normal_group_share_mean"] = (
        float(np.mean(mixed_shares)) if mixed_shares else 0.0)
    report["all_assertions_passed"] = True
    return report
# ======== END: PEMERIKSAAN KEBOCORAN ========


# ======== START: LAPORAN SPLIT ========
def partition_row(name, ids, y, groups, label_type):
    att = ids[y[ids] == 1]
    return {
        "partition": name,
        "n": int(len(ids)),
        "n_normal": int((y[ids] == 0).sum()),
        "n_attack": int(len(att)),
        "n_normal_groups": int(len(np.unique(groups[ids[y[ids] == 0]]))),
        "attack_subtypes": json.dumps(
            pd.Series(label_type[att]).value_counts().sort_index().to_dict()),
    }


def split_report(arrays, y, groups, label_type):
    order = ["holdout", "eval", "tune_train", "tune_calib", "tune_dev"]
    keys = order + sorted(k for k in arrays if k.startswith("s"))
    return pd.DataFrame([partition_row(k, arrays[k], y, groups, label_type)
                         for k in keys])
# ======== END: LAPORAN SPLIT ========


# ======== START: EKSEKUSI PER DATASET ========
cfg = sync_config(CONFIG_SECTIONS)
dcfg, sp = cfg["data"], cfg["split"]
cap = VALIDATION_CAP if VALIDATION_MODE else dcfg["cap_per_subtype"]
seeds = VALIDATION_SEEDS if VALIDATION_MODE else sp["seeds"]
run_hash = {"data": cfg["hashes"]["data"], "split": cfg["hashes"]["split"],
            "validation_mode": VALIDATION_MODE, "cap": cap, "seeds": seeds}

print(f"S1 mulai {now_str()} | mode={'VALIDASI' if VALIDATION_MODE else 'PENUH'}"
      f" | cap={cap:,} | seeds={seeds}")

for ds in dcfg["datasets"]:
    out = OUT_ROOT / ds
    out.mkdir(parents=True, exist_ok=True)
    done_path = out / "done.json"

    # ---- Checkpoint ----
    if done_path.exists():
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if done.get("run_hash") == run_hash:
            print(f"\nSKIP {ds}: split sudah dibuat dengan konfigurasi sama")
            continue
        raise RuntimeError(f"{ds}: done.json ada dengan konfigurasi berbeda. "
                           f"Hapus folder {out} secara sadar sebelum run ulang.")

    t0 = time.time()
    print(f"\n=== {ds} ===", flush=True)
    path = Path(dcfg["revision_parquet"][ds])
    manifest = json.loads(Path(dcfg["manifest"][ds]).read_text(encoding="utf-8"))
    if manifest["features"] != FEATURES:
        raise RuntimeError(f"{ds}: daftar fitur manifest berbeda dengan konfigurasi")

    # ---- Pass 1 ----
    is_normal, lt_codes, lt_cats = load_label_arrays(
        path, dcfg["normal_labels"], dcfg["attack_labels"])
    positions, sub_df, subtypes = balanced_sampling(
        is_normal, lt_codes, lt_cats, cap, sp["random_state"])
    if len(subtypes) != dcfg["expected_subtypes"][ds]:
        raise RuntimeError(f"{ds}: jumlah subtipe {len(subtypes)} != "
                           f"{dcfg['expected_subtypes'][ds]}")
    sub_df.insert(0, "dataset", ds)
    atomic_write_json(out / "subtype_sampling.json", sub_df.to_dict("records"))
    del lt_codes
    print(f"Pass 1: normal={int(is_normal.sum()):,} | "
          f"attack={int((~is_normal).sum()):,} | terpilih={len(positions):,} | "
          f"{time.time() - t0:.0f} s", flush=True)

    # ---- Pass 2 ----
    cols = FEATURES + ["label", "label_type", "flow_id", "ip.src", "ip.dst",
                       "source_file", "source_row"]
    df = read_selected_rows(path, positions, cols)
    y = (~is_normal[positions]).astype(np.int8)
    low = df["label"].astype(str).str.strip().str.lower()
    if not np.array_equal(y, (~low.isin(dcfg["normal_labels"])).to_numpy().astype(np.int8)):
        raise RuntimeError(f"{ds}: label pass 2 tidak cocok dengan pass 1")
    del is_normal

    fragment_pairs = []
    if ds in dcfg["fragment_pairs"]:
        fragment_pairs = json.loads(
            Path(dcfg["fragment_pairs"][ds]).read_text(encoding="utf-8"))
    group_key = build_group_key(df, ds, fragment_pairs)
    if group_key.isna().any():
        raise RuntimeError(f"{ds}: ada grup kosong")
    groups, _ = pd.factorize(group_key)
    groups = groups.astype(np.int64)

    label_type = np.where(y == 1, df["label_type"].astype(str), "normal")
    strata_norm, _ = pd.factorize(df["source_file"].astype(str))
    strata_att, _ = pd.factorize(pd.Series(label_type))
    print(f"Pass 2: sampel={len(df):,} | grup={len(np.unique(groups)):,} | "
          f"{time.time() - t0:.0f} s", flush=True)

    # ---- Split ----
    arrays = build_splits(y, strata_norm, strata_att, groups, seeds, sp)
    leak = check_leakage(arrays, y, groups, seeds, sp["n_folds"])
    print(f"Split + assert kebocoran lolos | {time.time() - t0:.0f} s", flush=True)

    # ---- Simpan sampel ----
    sample = pd.DataFrame({
        "sample_id": np.arange(len(df), dtype=np.int64),
        "row_idx": df["row_idx"].to_numpy(),
        "y": y,
        "label_type": label_type,
        "group_key": group_key.to_numpy(),
        "group_code": groups,
        "source_file": df["source_file"].astype(str).to_numpy(),
        "source_row": df["source_row"].to_numpy(dtype=np.int64),
    })
    for c in FEATURES:
        sample[c] = pd.to_numeric(df[c], errors="raise").astype("float64")
    atomic_write_parquet(out / "sample.parquet", sample)
    atomic_write_npz(out / "splits.npz", arrays)

    rep = split_report(arrays, y, groups, label_type)
    rep.insert(0, "dataset", ds)
    rep.to_csv(out / "split_report.csv", index=False)

    fold_keys = [k for k in arrays if k.endswith("_test")]
    ev_norm = arrays["eval"][y[arrays["eval"]] == 0]
    ev_norm_sizes = np.bincount(groups[ev_norm])
    ho = arrays["holdout"]
    summary = {
        "dataset": ds,
        "total_pcap_samples": int(manifest["raw_rows"]),
        "excluded_non_ipv4": int(manifest["excluded_missing_ip"]),
        "ipv4_packets": int(manifest["retained_rows"]),
        "selected_ddos_subtypes": len(subtypes),
        "cap_per_subtype": cap,
        "balanced_total_samples": int(len(y)),
        "balanced_normal_samples": int((y == 0).sum()),
        "balanced_attack_samples": int((y == 1).sum()),
        "tuning_holdout_samples": int(len(arrays["holdout"])),
        "final_evaluation_samples": int(len(arrays["eval"])),
        "test_samples_per_fold_mean": float(np.mean([len(arrays[k]) for k in fold_keys])),
        "test_samples_per_fold_min": int(min(len(arrays[k]) for k in fold_keys)),
        "test_samples_per_fold_max": int(max(len(arrays[k]) for k in fold_keys)),
        "holdout_normal_share": float((y[ho] == 0).sum() / (y == 0).sum()),
        "holdout_attack_share": float((y[ho] == 1).sum() / (y == 1).sum()),
        "eval_normal_largest_group": int(ev_norm_sizes.max()),
        "eval_normal_fold_target": float(len(ev_norm) / sp["n_folds"]),
        "final_features": len(FEATURES),
        "feature_columns": FEATURES,
        "scaler": dcfg["scaler_type"][ds],
        "cv_protocol": f"{len(seeds)} seeds x {sp['n_folds']}-fold; normal "
                       f"group-aware; folds regenerated per seed",
        "leakage_report": leak,
    }
    atomic_write_json(out / "dataset_summary.json", summary)

    done = {
        "dataset": ds,
        "run_hash": run_hash,
        "sample_hash": array_hash(sample["row_idx"].to_numpy(), y),
        "splits_hash": array_hash(*[arrays[k] for k in sorted(arrays)]),
        "finished_at": now_str(),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    atomic_write_json(done_path, done)

    print(json.dumps({k: summary[k] for k in [
        "balanced_total_samples", "tuning_holdout_samples",
        "holdout_normal_share", "holdout_attack_share",
        "eval_normal_largest_group", "eval_normal_fold_target",
        "final_evaluation_samples", "test_samples_per_fold_mean",
        "test_samples_per_fold_min", "test_samples_per_fold_max"]}, indent=2))
    print("Laporan kebocoran:", json.dumps(leak, indent=2))
    del df, sample, arrays

print(f"\nS1 selesai {now_str()} | output: {OUT_ROOT}")
# ======== END: EKSEKUSI PER DATASET ========
