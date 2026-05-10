#!/bin/bash

# Path to executable and checkpoint
EXE="./build/default/Release/brain_mri_pgm.exe"
CKPT="checkpoints/brain_mri.ckpt"

# Output file
OUTPUT="results.txt"

# Clear old output
> "$OUTPUT"

# Run inference on all .pgm files in yes and no folders
for folder in "data/brain_mri_pgm/yes" "data/brain_mri_pgm/no"
do
  for image in "$folder"/*.pgm
  do
    echo "Running on: \"$image\"" | tee -a "$OUTPUT"

    "$EXE" \
      --load "$CKPT" \
      --infer-only \
      --size 64 \
      --image "$image" | tee -a "$OUTPUT"

    echo "-----------------------------------" | tee -a "$OUTPUT"
  done
done	
