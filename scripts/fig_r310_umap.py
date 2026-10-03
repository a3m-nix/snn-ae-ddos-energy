# ======== START: HEADER ========
"""
fig_r310_umap.py — Gambar final R3.10 dari umap_s0_f0.npz (tanpa run ulang).
  Fig utama  : 2x2 laten SNN (Edge, CIC) — (a,b) normal vs serangan;
               (c,d) subtipe; CIC menyorot HTTP_Flood, SlowLoris, UDP_Flood.
  Fig suplemen: per dataset 2x3 (laten SNN | laten Dense | input terskala).
Keluaran: 06-posthoc/latent/figs/*.pdf + *.png (300 dpi), lebar 180 mm.
"""
# ======== END: HEADER ========


# ======== START: KONFIGURASI ========
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/jupyteruser/nix-ext-backup/0-eksperimen-final/0-snn-ae-revision/06-posthoc/latent")
OUT = ROOT / "figs"; OUT.mkdir(exist_ok=True)
MM = 1 / 25.4
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 7, "axes.titlesize": 8, "legend.fontsize": 6})
DSN = {"edge_iiotset": "Edge-IIoTset", "ciciot2023": "CICIoT2023"}
C_NORM, C_ATT, C_PALE = "#9E9E9E", "#E69F00", "#F3E3C3"   # Okabe-Ito
EDGE_SUB = {"ddos-http-flood-attacks": ("HTTP flood", "#0072B2"),
            "ddos-icmp-flood-attacks": ("ICMP flood", "#CC79A7"),
            "ddos-tcp-syn-flood-attacks": ("TCP SYN flood", "#E69F00"),
            "ddos-udp-flood-attacks": ("UDP flood", "#009E73")}
CIC_HL = {"DDoS-HTTP_Flood-": ("HTTP_Flood", "#0072B2"),
          "DDoS-SlowLoris": ("SlowLoris", "#D55E00"),
          "DDoS-UDP_Flood": ("UDP_Flood", "#009E73")}
S = 2.5   # ukuran titik
# ======== END: KONFIGURASI ========


# ======== START: UTILITAS ========
def load(ds):
    return np.load(ROOT / ds / "umap_s0_f0.npz")

def clean(ax, title=None, tag=None):
    ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title)
    if tag: ax.text(-0.04, 1.04, tag, transform=ax.transAxes,
                    fontweight="bold", fontsize=9, va="bottom", ha="right")

def sc(ax, E, m, c, lab=None, z=1, a=0.6):
    ax.scatter(E[m, 0], E[m, 1], s=S, c=c, alpha=a, lw=0, label=lab,
               zorder=z, rasterized=True)

def panel_binary(ax, E, y):
    sc(ax, E, y == 0, C_NORM, "Normal", 1)
    sc(ax, E, y == 1, C_ATT, "Attack", 2)
    ax.legend(markerscale=3, loc="best", frameon=False)

def panel_sub(ax, E, y, lt, ds):
    sc(ax, E, y == 0, C_NORM, "Normal", 1)
    if ds == "edge_iiotset":
        for k, (lab, c) in EDGE_SUB.items():
            sc(ax, E, lt == k, c, lab, 2)
    else:
        sc(ax, E, (y == 1) & ~np.isin(lt, list(CIC_HL)), C_PALE,
           "Other attacks", 2, 0.8)
        for k, (lab, c) in CIC_HL.items():
            sc(ax, E, lt == k, c, lab, 3, 0.9)
    ax.legend(markerscale=3, loc="best", frameon=False)

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
# ======== END: UTILITAS ========


# ======== START: GAMBAR UTAMA ========
fig, ax = plt.subplots(2, 2, figsize=(180 * MM, 150 * MM))
for j, ds in enumerate(DSN):
    d = load(ds); E, y, lt = d["latent"], d["y"], d["label_type"].astype(str)
    panel_binary(ax[0, j], E, y); clean(ax[0, j], DSN[ds], "ab"[j])
    panel_sub(ax[1, j], E, y, lt, ds); clean(ax[1, j], None, "cd"[j])
fig.tight_layout()
save(fig, "fig_r310_umap_main")
# ======== END: GAMBAR UTAMA ========


# ======== START: GAMBAR SUPLEMEN ========
SPACES = [("latent", "SNN-AE latent"), ("dense", "Dense-AE latent"),
          ("input", "Scaled input (33 features)")]
for ds in DSN:
    d = load(ds); y, lt = d["y"], d["label_type"].astype(str)
    subs = sorted(set(lt[y == 1]))
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(2, 3, figsize=(180 * MM, 120 * MM))
    for j, (key, title) in enumerate(SPACES):
        E = d[key]
        panel_binary(ax[0, j], E, y); clean(ax[0, j], title, "abc"[j])
        sc(ax[1, j], E, y == 0, C_NORM, "Normal", 1)
        for i, s in enumerate(subs):
            sc(ax[1, j], E, lt == s, cmap(i % 20), s, 2)
        clean(ax[1, j], None, "def"[j])
    h, l = ax[1, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, markerscale=3,
               frameon=False, bbox_to_anchor=(0.5, -0.02 - 0.012 * len(subs) / 4))
    fig.tight_layout()
    save(fig, f"fig_r310_umap_supp_{ds}")
print("selesai ->", OUT)
# ======== END: GAMBAR SUPLEMEN ========
