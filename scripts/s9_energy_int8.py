# ======== START: HEADER ========
"""
s9_energy_int8.py — Antisipasi putaran kedua (R3): SNN-AE vs Dense-AE terkuantisasi INT8.

Analitik, TANPA melatih ulang. Memakai k_enc, k_dec, s_enc, s_dec per fold dari S6
(07-energy/spike_sparsity_folds.csv) dan rumus yang sama dengan s6_energy.py.
Hanya konstanta energi dan lebar data (bit) yang diganti.

Skenario:
  A      : SNN-AE FP32 vs Dense-AE FP32   (harus identik dengan S6 -> validasi)
  B_lo/hi: SNN-AE FP32 vs Dense-AE INT8   (kritik langsung: ANN terkuantisasi)
  C_lo/hi: SNN-AE INT8 vs Dense-AE INT8   (pembanding adil)
  lo = INT8 MAC 0.23 pJ, AC 0.03 pJ (8-bit mult + 8-bit add; menguntungkan ANN)
  hi = INT8 MAC 0.30 pJ, AC 0.10 pJ (8-bit mult + akumulator 32-bit; lebih realistis)
Memori (T3): bobot dan aktivasi INT8 dikemas 4 per word 32-bit; spike tetap 1-bit;
state membran tetap 32-bit (konservatif untuk SNN).

Jalankan:  /home/jupyteruser/jupyter_env/bin/python s9_energy_int8.py
Keluaran:  06-posthoc/int8/
"""
# ======== END: HEADER ========


# ======== START: IMPORT DAN KONFIGURASI ========
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon

BASE = Path("/home/jupyteruser/nix-ext-backup/0-eksperimen-final/0-snn-ae-revision")
OUT = BASE / "06-posthoc" / "int8"; OUT.mkdir(parents=True, exist_ok=True)
CFG = json.loads((BASE / "00-config" / "config_revision.json").read_text())

L, DIN, DOUT = 32, 33, 33                         # ukuran laten, input, output (sama dengan S6)
EM_GRID = [2.5, 5.0, 10.0, 20.0]                  # pJ per akses word 32-bit (sama dengan S6)
TIERS = ["T1", "T2", "T3_opt", "T3_pess"]

# Konstanta per presisi (Horowitz 2014, 45 nm). UBAH di sini jika ingin skenario lain.
PREC = {
    "FP32":    {"MAC": CFG["energy"]["E_MAC"], "AC": CFG["energy"]["E_AC"], "LIF": CFG["energy"]["E_LIF"], "pack": 1},
    "INT8_lo": {"MAC": 0.2 + 0.03, "AC": 0.03, "LIF": CFG["energy"]["E_LIF"], "pack": 4},
    "INT8_hi": {"MAC": 0.2 + 0.1,  "AC": 0.10, "LIF": CFG["energy"]["E_LIF"], "pack": 4},
}
# Catatan: E_LIF dibiarkan sama di semua skenario (konservatif untuk SNN).

SCEN = {                                          # skenario: (presisi SNN-AE, presisi Dense-AE)
    "A":    ("FP32", "FP32"),
    "B_lo": ("FP32", "INT8_lo"), "B_hi": ("FP32", "INT8_hi"),
    "C_lo": ("INT8_lo", "INT8_lo"), "C_hi": ("INT8_hi", "INT8_hi"),
}
DS_ORDER = ["Edge-IIoTset", "CICIoT2023"]
# ======== END: IMPORT DAN KONFIGURASI ========


# ======== START: MODEL ENERGI (rumus = s6_energy.py, konstanta dapat diganti) ========
def e_snn(T, k_enc, k_dec, em, p):
    c = PREC[p]; wi, wint, wo = DIN * L, L * L, L * DOUT
    t1 = wi * c["MAC"] + k_enc * wint * c["AC"] + k_dec * wo * c["AC"] + 2 * L * T * c["LIF"]
    reads = (wi + k_enc * wint + k_dec * wo) / c["pack"]          # word 32-bit
    act = (DIN + DOUT) / c["pack"] + 2 * 2 * L * T / 32           # spike 1-bit
    state = 2 * 2 * L * T                                          # membran 32-bit
    t3 = t1 + em * (reads + act)
    return {"T1": t1, "T2": t1, "T3_opt": t3, "T3_pess": t3 + em * state}

def e_dense(s_enc, s_dec, em, p):
    c = PREC[p]; wi, wint, wo = DIN * L, L * L, L * DOUT
    t1 = (wi + wint + wo) * c["MAC"]
    t2 = (wi + (1 - s_enc) * wint + (1 - s_dec) * wo) * c["MAC"]
    reads = (wi + (1 - s_enc) * wint + (1 - s_dec) * wo) / c["pack"]
    act = (DIN + DOUT + 2 * 2 * L) / c["pack"]
    t3 = t2 + em * (reads + act)
    return {"T1": t1, "T2": t2, "T3_opt": t3, "T3_pess": t3}

def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(run, 1.0)
    return adj
# ======== END: MODEL ENERGI ========


# ======== START: DATA ========
SS = pd.read_csv(BASE / "07-energy" / "spike_sparsity_folds.csv")
key = ["dataset", "seed", "fold"]
S = SS[SS.job == "SNN_AE"][key + ["T", "k_enc", "k_dec"]].sort_values(key).reset_index(drop=True)
D = SS[SS.job == "Dense_AE"][key + ["s_enc", "s_dec"]].sort_values(key).reset_index(drop=True)
assert len(S) == len(D) == 60, (len(S), len(D))
assert (S[key].values == D[key].values).all(), "urutan fold SNN-AE dan Dense-AE tidak sama"
print("data: 60 fold (2 dataset x 30)")
# ======== END: DATA ========


# ======== START: VALIDASI SKENARIO A TERHADAP S6 ========
EF = pd.read_csv(BASE / "07-energy" / "energy_folds.csv")
worst = 0.0
for job in ["SNN_AE", "Dense_AE"]:
    ref = EF[EF.job == job].set_index(key + ["Em"])
    for i in range(60):
        for em in EM_GRID:
            e = e_snn(S["T"][i], S.k_enc[i], S.k_dec[i], em, "FP32") if job == "SNN_AE" \
                else e_dense(D.s_enc[i], D.s_dec[i], em, "FP32")
            r = ref.loc[(S.dataset[i], S.seed[i], S.fold[i], em)]
            for t in TIERS:
                worst = max(worst, abs(e[t] - r[t]) / r[t])
assert worst < 1e-9, f"skenario A tidak sama dengan S6 (selisih relatif {worst:.2e})"
print(f"validasi skenario A = S6: OK (selisih relatif maks {worst:.1e})")
# ======== END: VALIDASI SKENARIO A ========


# ======== START: HITUNG SEMUA SKENARIO ========
rows = []
for sc, (ps, pd_) in SCEN.items():
    for i in range(60):
        for em in EM_GRID:
            es = e_snn(S["T"][i], S.k_enc[i], S.k_dec[i], em, ps)
            ed = e_dense(D.s_enc[i], D.s_dec[i], em, pd_)
            for t in TIERS:
                rows.append({"scenario": sc, "dataset": S.dataset[i], "seed": S.seed[i], "fold": S.fold[i],
                             "Em": em, "tier": t, "snn": es[t], "dense": ed[t]})
F = pd.DataFrame(rows)
F.to_csv(OUT / "int8_energy_folds.csv", index=False)
# ======== END: HITUNG SEMUA SKENARIO ========


# ======== START: UJI DAN BREAK-EVEN ========
tests = []
for (sc, ds), g in F.groupby(["scenario", "dataset"], sort=False):
    res = []
    for (em, t), x in g.groupby(["Em", "tier"], sort=False):
        a, b = x.snn.to_numpy(), x.dense.to_numpy()
        try:
            _, p = wilcoxon(a, b) if not np.allclose(a - b, 0) else (np.nan, 1.0)
        except ValueError:
            p = 1.0
        res.append({"scenario": sc, "dataset": ds, "Em": em, "tier": t, "snn_mean": a.mean(),
                    "dense_mean": b.mean(), "saving_pct": 100 * (1 - a.mean() / b.mean()),
                    "folds_snn_lower": int((a < b).sum()), "p": p})
    for r, h in zip(res, holm([r["p"] for r in res])):
        r["p_holm"] = h
    tests.extend(res)
TST = pd.DataFrame(tests); TST.to_csv(OUT / "int8_tests.csv", index=False)

be = []
for sc, (ps, pd_) in SCEN.items():
    for ds in DS_ORDER:
        s = S[S.dataset == ds]; d = D[D.dataset == ds]; T = int(s["T"].iloc[0])
        k_eff = ((s.k_enc * L * L + s.k_dec * L * DOUT) / (L * L + L * DOUT)).mean()
        for em in EM_GRID:
            for t in TIERS:
                dense_e = np.mean([e_dense(se, sd, em, pd_)[t] for se, sd in zip(d.s_enc, d.s_dec)])
                e0, e1 = e_snn(T, 0, 0, em, ps)[t], e_snn(T, 1, 1, em, ps)[t]
                ks = (dense_e - e0) / (e1 - e0)
                be.append({"scenario": sc, "dataset": ds, "Em": em, "tier": t, "k_star": ks,
                           "k_measured_eff": k_eff, "snn_below_breakeven": bool(k_eff < ks)})
BE = pd.DataFrame(be); BE.to_csv(OUT / "int8_break_even.csv", index=False)
# ======== END: UJI DAN BREAK-EVEN ========


# ======== START: RINGKASAN (Em = 5 pJ) ========
pd.set_option("display.width", 200)
m = TST[TST.Em == 5.0].merge(BE[BE.Em == 5.0], on=["scenario", "dataset", "Em", "tier"])
m = m[["scenario", "dataset", "tier", "snn_mean", "dense_mean", "saving_pct", "folds_snn_lower",
       "p_holm", "k_star", "k_measured_eff"]].round({"snn_mean": 0, "dense_mean": 0, "saving_pct": 1,
                                                       "k_star": 3, "k_measured_eff": 3})
m.to_csv(OUT / "int8_summary_Em5.csv", index=False)
print("\nRingkasan Em = 5 pJ (saving_pct > 0 berarti SNN-AE lebih hemat):")
print(m.to_string(index=False))
print("\nKeluaran:", sorted(p.name for p in OUT.iterdir()))
# ======== END: RINGKASAN ========
