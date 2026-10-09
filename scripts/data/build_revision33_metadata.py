#!/usr/bin/env python
# coding: utf-8

# In[2]:


from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# KONFIGURASI
# ============================================================

SOURCES = {
    "edge_iiotset": Path(
        "/home/jupyteruser/nix-ext/dataset/edge-iiotset_parquet"
    ),
    "ciciot2023": Path(
        "/home/jupyteruser/nix-ext/dataset/ciciot2023_parquet"
    ),
}

OUT_DIR = Path(
    "/home/jupyteruser/nix-ext-backup/0-eksperimen-final/"
    "dataset-revision-33"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 250_000

# Hanya kolom ini yang boleh masuk ke model.
FEATURES = [
    "frame.len",
    "ip.version",
    "ip.hdr_len",
    "ip.dsfield.dscp",
    "ip.len",
    "ip.id",
    "ip.frag_offset",
    "ip.flags.df",
    "ip.flags.mf",
    "ip.ttl",
    "ip.proto",
    "tcp.srcport",
    "tcp.dstport",
    "tcp.seq",
    "tcp.ack",
    "tcp.hdr_len",
    "tcp.len",
    "tcp.window_size",
    "tcp.flags.fin",
    "tcp.flags.syn",
    "tcp.flags.reset",
    "tcp.flags.push",
    "tcp.flags.ack",
    "tcp.flags.urg",
    "tcp.flags.ns",
    "tcp.flags.cwr",
    "udp.srcport",
    "udp.dstport",
    "udp.length",
    "icmp.type",
    "icmp.code",
    "data.len",
    "inter_arrival_time",
]

METADATA = [
    "ip.src",
    "ip.dst",
    "frame.time_epoch",
    "flow_id",
    "packet_index",
    "label",
    "label_type",
]

assert len(FEATURES) == 33


# ============================================================
# FUNGSI PEMERIKSAAN DAN FLOW ID
# ============================================================

def valid_ip_mask(df):
    """Periksa ketersediaan IP; tidak mengisi IP kosong dengan '0'."""
    valid = pd.Series(True, index=df.index)

    for column in ["ip.src", "ip.dst"]:
        values = df[column].astype("string").str.strip()

        invalid = (
            values.isna()
            | values.str.lower().isin(
                ["", "0", "nan", "none", "null"]
            )
        )
        valid &= ~invalid

    return valid


def checked_integer(series, name):
    values = pd.to_numeric(series, errors="raise")

    if values.isna().any():
        raise ValueError(
            f"{name}: ada nilai kosong pada paket terkait"
        )

    array = values.to_numpy(dtype="float64")

    if not np.isfinite(array).all():
        raise ValueError(f"{name}: ada nilai tidak finite")

    if (array != np.floor(array)).any():
        raise ValueError(f"{name}: ada nilai bukan bilangan bulat")

    return values.astype("int64")


def make_bidir_id(df):
    """
    Kelompok bidirectional berdasarkan:
    IP endpoint, port TCP/UDP, dan protokol.

    Tidak menggunakan label, subtype, timestamp, atau nama file.
    ID yang sama lintas file tetap menjadi kelompok yang sama.
    """
    if not valid_ip_mask(df).all():
        raise ValueError("Masih ada IP kosong setelah penyaringan")

    src_ip = df["ip.src"].astype("string").str.strip().astype(str)
    dst_ip = df["ip.dst"].astype("string").str.strip().astype(str)

    proto = checked_integer(df["ip.proto"], "ip.proto")

    if ((proto < 0) | (proto > 255)).any():
        raise ValueError("ip.proto di luar rentang 0–255")

    src_port = pd.Series(0, index=df.index, dtype="int64")
    dst_port = pd.Series(0, index=df.index, dtype="int64")

    for protocol, prefix in [(6, "tcp"), (17, "udp")]:
        mask = proto.eq(protocol)

        for side, target in [
            ("src", src_port),
            ("dst", dst_port),
        ]:
            column = f"{prefix}.{side}port"
            ports = checked_integer(df.loc[mask, column], column)

            if ((ports < 0) | (ports > 65535)).any():
                raise ValueError(f"{column}: port di luar rentang")

            target.loc[mask] = ports

    endpoint_a = pd.Series(
        [
            json.dumps([ip, int(port)], separators=(",", ":"))
            for ip, port in zip(src_ip, src_port)
        ],
        index=df.index,
    )

    endpoint_b = pd.Series(
        [
            json.dumps([ip, int(port)], separators=(",", ":"))
            for ip, port in zip(dst_ip, dst_port)
        ],
        index=df.index,
    )

    forward = endpoint_a.le(endpoint_b)
    first = endpoint_a.where(forward, endpoint_b)
    second = endpoint_b.where(forward, endpoint_a)

    keys = proto.astype(str) + "|" + first + "|" + second

    # Hash setiap tuple unik sekali per batch.
    lookup = {
        key: hashlib.sha256(key.encode("utf-8")).hexdigest()
        for key in keys.unique()
    }

    return keys.map(lookup).to_numpy()


# ============================================================
# PEMBUATAN PARQUET
# ============================================================

for dataset, source_dir in SOURCES.items():
    print(f"\n=== {dataset} ===", flush=True)

    destination = OUT_DIR / f"{dataset}_bidir_metadata.parquet"
    temporary = OUT_DIR / f"{dataset}_bidir_metadata.partial.parquet"
    report_path = OUT_DIR / f"{dataset}_filter_report.csv"
    manifest_path = OUT_DIR / f"{dataset}_manifest.json"

    # Lewati output yang sudah selesai dan memiliki manifest.
    if destination.exists():
        if not manifest_path.exists():
            raise FileExistsError(
                f"{destination} sudah ada, tetapi manifest belum ada. "
                "Periksa output tersebut sebelum menjalankan ulang."
            )

        existing = json.loads(manifest_path.read_text())
        actual_rows = pq.ParquetFile(destination).metadata.num_rows

        if (
            existing.get("features") != FEATURES
            or existing.get("retained_rows") != actual_rows
        ):
            raise ValueError(
                f"Output dan manifest tidak cocok: {destination}"
            )

        print(
            f"SKIP: output sudah selesai, {actual_rows:,} baris",
            flush=True,
        )
        continue

    files = sorted(source_dir.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"Tidak ada file parquet di {source_dir}"
        )

    # Periksa semua schema sebelum mulai menulis.
    for source in files:
        columns = pq.ParquetFile(source).schema_arrow.names
        missing = sorted(
            set(FEATURES + METADATA) - set(columns)
        )

        if missing:
            raise ValueError(
                f"{source.name}: kolom hilang {missing}"
            )

    # Hanya membersihkan output sementara dari proses gagal.
    if temporary.exists():
        print(
            f"Menghapus output sementara: {temporary.name}",
            flush=True,
        )
        temporary.unlink()

    writer = None
    reports = []
    total_raw = 0
    total_retained = 0
    total_excluded = 0

    try:
        for source in files:
            parquet = pq.ParquetFile(source)

            file_raw = 0
            file_retained = 0
            file_excluded = 0

            for batch in parquet.iter_batches(
                batch_size=BATCH_SIZE,
                columns=FEATURES + METADATA,
            ):
                df = batch.to_pandas().reset_index(drop=True)
                raw_rows = len(df)

                # Posisi asli dalam parquet sumber.
                df["source_file"] = source.name
                df["source_row"] = np.arange(
                    file_raw,
                    file_raw + raw_rows,
                    dtype=np.int64,
                )
                file_raw += raw_rows

                valid = valid_ip_mask(df)
                excluded = int((~valid).sum())
                file_excluded += excluded

                if excluded:
                    print(
                        f"FILTER | {source.name} | "
                        f"IP tidak tersedia={excluded:,}",
                        flush=True,
                    )

                df = (
                    df.loc[valid]
                    .copy()
                    .reset_index(drop=True)
                )

                if df.empty:
                    continue

                df["original_flow_id"] = (
                    df["flow_id"].astype("string")
                )
                df["flow_id"] = make_bidir_id(df)

                # Tetapkan tipe kolom konsisten lintas file/batch.
                for column in FEATURES + ["frame.time_epoch"]:
                    df[column] = pd.to_numeric(
                        df[column], errors="raise"
                    ).astype("float64")

                for column in [
                    "ip.src", "ip.dst", "flow_id",
                    "original_flow_id", "label",
                    "label_type", "source_file",
                ]:
                    df[column] = df[column].astype("string")

                df["packet_index"] = pd.to_numeric(
                    df["packet_index"], errors="raise"
                ).astype("Int64")

                table = pa.Table.from_pandas(
                    df, preserve_index=False
                )

                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary,
                        table.schema,
                        compression="snappy",
                    )
                else:
                    table = table.cast(writer.schema)

                writer.write_table(table)
                file_retained += len(df)

            assert file_raw == file_retained + file_excluded

            reports.append({
                "dataset": dataset,
                "source_file": source.name,
                "raw_rows": file_raw,
                "excluded_missing_ip": file_excluded,
                "retained_rows": file_retained,
            })

            total_raw += file_raw
            total_retained += file_retained
            total_excluded += file_excluded

            # Simpan laporan setelah setiap file selesai.
            pd.DataFrame(reports).to_csv(
                report_path, index=False
            )

            print(
                f"DONE | {source.name} | "
                f"awal={file_raw:,} | "
                f"dikeluarkan={file_excluded:,} | "
                f"tersimpan={file_retained:,}",
                flush=True,
            )

        if writer is None:
            raise ValueError(
                f"{dataset}: tidak ada baris yang lolos penyaringan"
            )

        writer.close()
        writer = None

        actual_rows = pq.ParquetFile(temporary).metadata.num_rows
        assert actual_rows == total_retained
        assert total_raw == total_retained + total_excluded

        temporary.rename(destination)

        manifest = {
            "dataset": dataset,
            "source_dir": str(source_dir),
            "output": str(destination),
            "features": FEATURES,
            "input_dim": len(FEATURES),
            "raw_rows": total_raw,
            "excluded_missing_ip": total_excluded,
            "retained_rows": total_retained,
            "group_definition": (
                "Global bidirectional endpoint tuple: "
                "IP addresses, TCP/UDP ports, IP protocol; "
                "zero ports for other protocols"
            ),
            "group_hash": "SHA-256",
            "source_file_is_verified_capture_id": False,
        }

        manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        print(f"\nSAVED: {destination}", flush=True)
        print(f"BARIS AWAL: {total_raw:,}")
        print(f"DIKELUARKAN: {total_excluded:,}")
        print(f"BARIS FINAL: {total_retained:,}")
        print(f"FITUR MODEL: {len(FEATURES)}")
        print(f"LAPORAN: {report_path}")

    finally:
        if writer is not None:
            writer.close()

print("\nSELESAI")


