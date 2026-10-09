#!/bin/bash

INPUT_DIR="/home/jupyteruser/nix-ext/dataset/edge-iiotset_extracted"
OUTPUT_DIR="/home/jupyteruser/nix-ext/dataset/edge-iiotset_sampled"
SAMPLE_SIZE=1500000

mkdir -p $OUTPUT_DIR

echo "Sampling 1.5M packets (DDoS + Normal only)..."
echo "============================================="

# List of files to process
files=(
    "Distance.csv"
    "Flame_Sensor.csv"
    "Heart_Rate.csv"
    "IR_Receiver.csv"
    "Modbus.csv"
    "Soil_Moisture.csv"
    "Sound_Sensor.csv"
    "Temperature_and_Humidity.csv"
    "Water_Level.csv"
    "phValue.csv"
    "ddos-http-flood-attacks.csv"
    "ddos-icmp-flood-attacks.csv"
    "ddos-tcp-syn-flood-attacks.csv"
    "ddos-udp-flood-attacks.csv"
)

for filename in "${files[@]}"; do
    input_file="$INPUT_DIR/$filename"
    output_file="$OUTPUT_DIR/$filename"
    
    if [ -f "$input_file" ]; then
        echo "Processing: $filename"
        head -n $((SAMPLE_SIZE + 1)) "$input_file" > "$output_file"
        lines=$(wc -l < "$output_file")
        echo "  ✓ Saved: $lines lines"
    else
        echo "  ✗ File not found: $filename"
    fi
done

echo ""
echo "============================================="
echo "Sampling completed!"
echo "============================================="
ls -lh $OUTPUT_DIR
