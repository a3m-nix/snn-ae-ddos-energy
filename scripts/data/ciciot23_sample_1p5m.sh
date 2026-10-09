#!/bin/bash

INPUT_DIR="./ciciot2023_extracted"
OUTPUT_DIR="./ciciot2023_sampled"
SAMPLE_SIZE=1500000

mkdir -p $OUTPUT_DIR

echo "Sampling 1.5M packets per class..."

for csv_file in $INPUT_DIR/*.csv; do
    filename=$(basename "$csv_file")
    output_file="$OUTPUT_DIR/$filename"
    
    echo "Processing: $filename"
    
    # Ambil header + 1.5M baris pertama
    head -n $((SAMPLE_SIZE + 1)) "$csv_file" > "$output_file"
    
    lines=$(wc -l < "$output_file")
    echo "  ✓ Saved: $lines lines"
done

echo ""
echo "Done! Sampled files in: $OUTPUT_DIR"
ls -lh $OUTPUT_DIR
