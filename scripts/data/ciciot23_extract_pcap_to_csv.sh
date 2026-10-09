#!/bin/bash

INPUT_DIR="./ciciot2023pcap"
OUTPUT_DIR="./ciciot2023_extracted"

mkdir -p $OUTPUT_DIR

total=$(find $INPUT_DIR -name "*.pcap" | wc -l)
count=0

echo "========================================="
echo "Total files: $total"
echo "Mode: Sequential (1 by 1)"
echo "========================================="

for pcap_file in $INPUT_DIR/*.pcap; do
    filename=$(basename "$pcap_file" .pcap)
    output_csv="$OUTPUT_DIR/${filename}.csv"
    
    # Skip if already exists
    if [ -f "$output_csv" ]; then
        echo "[SKIP] $filename already extracted"
        continue
    fi
    
    count=$((count+1))
    echo ""
    echo "[$count/$total] [$(date +%H:%M:%S)] Processing: $filename"
    echo "Progress updates every 30 seconds..."
    
    # Run TShark dengan progress monitoring
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
      > "$output_csv" &
    
    tshark_pid=$!
    
    # Monitor progress setiap 30 detik
    while kill -0 $tshark_pid 2>/dev/null; do
        sleep 30
        if [ -f "$output_csv" ]; then
            current_lines=$(wc -l < "$output_csv" 2>/dev/null || echo 0)
            current_size=$(du -h "$output_csv" 2>/dev/null | cut -f1 || echo "0")
            echo "  → $(date +%H:%M:%S) - $current_lines packets extracted, $current_size"
        fi
    done
    
    wait $tshark_pid
    
    packets=$(wc -l < "$output_csv")
    size=$(du -h "$output_csv" | cut -f1)
    echo "✓ Done: $packets packets, $size"
done

echo ""
echo "========================================="
echo "ALL COMPLETED at $(date)"
echo "========================================="
