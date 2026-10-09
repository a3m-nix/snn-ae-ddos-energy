#!/bin/bash

INPUT_DIR="/home/jupyteruser/nix-ext-backup/0-eksperimen-final/dataset-raw/pcap"
OUTPUT_DIR="/home/jupyteruser/nix-ext-backup/0-eksperimen-final/dataset-raw/csv"

mkdir -p $OUTPUT_DIR

extract_pcap() {
    pcap_file=$1
    filename=$(basename "$pcap_file" .pcap)
    output_csv="$OUTPUT_DIR/${filename}.csv"
    
    echo "[$(date +%H:%M:%S)] Processing: $filename"
    
    tshark -r "$pcap_file" \
      -n -N m \
      -T fields \
      -e frame.len -e frame.time_epoch \
      -e ip.version -e ip.hdr_len -e ip.dsfield.dscp -e ip.len -e ip.id \
      -e ip.frag_offset -e ip.flags.df -e ip.flags.mf -e ip.ttl -e ip.proto \
      -e ip.src -e ip.dst \
      -e tcp.srcport -e tcp.dstport -e tcp.seq -e tcp.ack -e tcp.hdr_len \
      -e tcp.len -e tcp.window_size \
      -e tcp.flags.fin -e tcp.flags.syn -e tcp.flags.reset -e tcp.flags.push \
      -e tcp.flags.ack -e tcp.flags.urg -e tcp.flags.ns -e tcp.flags.cwr \
      -e udp.srcport -e udp.dstport -e udp.length \
      -e icmp.type -e icmp.code \
      -e data.len \
      -E header=y -E separator=, -E quote=d -E occurrence=f \
      > "$output_csv"
    
    packets=$(wc -l < "$output_csv")
    size=$(du -h "$output_csv" | cut -f1)
    echo "  ✓ Done: $packets packets, $size"
}

export -f extract_pcap
export OUTPUT_DIR

#total_attacks=$(find $INPUT_DIR -maxdepth 1 -name "*.pcap" | wc -l)
total_normal=$(find $INPUT_DIR/ddos -name "*.pcap" | wc -l)
total_normal=$(find $INPUT_DIR/normal -name "*.pcap" | wc -l)
total=$((total_attacks + total_normal))

echo "========================================="
echo "Edge-IIoTset TShark Extraction"
echo "Total files: $total ($total_attacks attacks + $total_normal normal)"
echo "Mode: Sequential"
echo "========================================="
echo ""

# Extract attack files
echo "Extracting attack files..."
for pcap_file in $INPUT_DIR/ddos/*.pcap; do
    if [ -f "$pcap_file" ]; then
        extract_pcap "$pcap_file"
    fi
done

# Extract normal files
echo ""
echo "Extracting normal traffic files..."
for pcap_file in $INPUT_DIR/normal/*.pcap; do
    extract_pcap "$pcap_file"
done

echo ""
echo "========================================="
echo "EXTRACTION COMPLETED at $(date)"
echo "========================================="
ls -lh $OUTPUT_DIR/*.csv
