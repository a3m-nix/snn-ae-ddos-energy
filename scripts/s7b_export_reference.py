# ======== START: HEADER ========
"""
S7b-REF — Ekspor fitur referensi untuk uji kesamaan (parity) TShark server vs RPi.
Dijalankan di SERVER. Mengambil paket tersampel S1 dari file sumber yang sama dengan
PCAP end-to-end, lengkap dengan source_row (= urutan paket di PCAP) dan 33 fitur mentah.
Output: 06-posthoc/e2e_reference/ref_<dataset>_<normal|attack>.parquet
"""
# ======== END: HEADER ========

# ======== START: KONFIGURASI ========
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

WORK = Path("/home/jupyteruser/nix-ext-backup/0-eksperimen-final/0-snn-ae-revision")
FEATURES = json.loads((WORK / "00-config" / "config_revision.json").read_text())["data"]["features"]
OUT = WORK / "06-posthoc" / "e2e_reference"
OUT.mkdir(parents=True, exist_ok=True)
MAP = {"edge_iiotset": {"normal": "Temperature_and_Humidity.parquet",
                        "attack": "ddos-tcp-syn-flood-attacks.parquet"},
       "ciciot2023": {"normal": "BenignTraffic.parquet", "attack": "DDoS-SYN_Flood.parquet"}}
MAX_ROW = 300_000      # cukup ekstrak 300 ribu paket pertama di RPi
N_ROWS = 3_000
# ======== END: KONFIGURASI ========

# ======== START: EKSPOR ========
for ds, files in MAP.items():
    t = pq.read_table(WORK / "02-splits" / "full" / ds / "sample.parquet",
                      columns=["source_file", "source_row", "y"] + FEATURES).to_pandas()
    for kind, fname in files.items():
        sub = t[(t.source_file == fname) & (t.source_row < MAX_ROW)].sort_values("source_row").head(N_ROWS)
        if sub.empty:
            raise SystemExit(f"{ds}/{kind}: tidak ada baris tersampel dari {fname}")
        exp_y = 0 if kind == "normal" else 1
        if (sub.y != exp_y).any():
            raise SystemExit(f"{ds}/{kind}: label tidak sesuai")
        out = OUT / f"ref_{ds}_{kind}.parquet"
        pq.write_table(pa.Table.from_pandas(sub.reset_index(drop=True), preserve_index=False), out)
        print(f"{ds}/{kind}: {len(sub)} baris | source_row {sub.source_row.min()}–{sub.source_row.max()} -> {out.name}")
(OUT / "features.json").write_text(json.dumps(FEATURES))
print(f"Selesai: {OUT}")
# ======== END: EKSPOR ========
