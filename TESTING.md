# Model Inference Testing Guide

This document provides step-by-step instructions for testing the brain MRI model in inference-only mode using a pre-trained checkpoint.

## Prerequisites

1. **Build the project** with the brain MRI PGM target:
   ```bash
   cmake --preset default -DNN_BUILD_BRAIN_MRI=OFF -DNN_BUILD_BRAIN_MRI_PGM=ON
   cmake --build --preset default --config Release --target brain_mri_pgm
   ```

2. **Checkpoint available:** Ensure `checkpoints/brain_mri.ckpt` exists (pre-trained model).

3. **Test images available:** Images in `data/brain_mri_pgm/` directory (preprocessed as `.pgm` files).

## Quick Start: Inference with a Single Image

Run inference on a tumor image:

```bash
./build/default/Release/brain_mri_pgm.exe \
  --load checkpoints/brain_mri.ckpt \
  --infer-only \
  --size 64 \
  --image data/brain_mri_pgm/yes/000002_Y100.pgm
```

Run inference on a non-tumor image:

```bash
./build/default/Release/brain_mri_pgm.exe \
  --load checkpoints/brain_mri.ckpt \
  --infer-only \
  --size 64 \
  --image data/brain_mri_pgm/no/000001_N100.pgm
```

## Expected Output

Upon successful inference, you will see:

```
Mode: infer-only
Backend (conv/dense): GPU/GPU (auto)
Loaded checkpoint: checkpoints/brain_mri_hybrid.ckpt
Inference result for data/brain_mri_pgm/yes/000002_Y100.pgm: 1 (tumor detected)
```

Or for negative cases:

```
Inference result for data/brain_mri_pgm/no/000001_N100.pgm: 0 (no tumor)
```

## Inference Options

### GPU vs CPU

**GPU inference (default):**
```bash
./build/default/Release/brain_mri_pgm.exe \
  --load checkpoints/brain_mri_hybrid.ckpt \
  --infer-only \
  --size 64 \
  --image data/brain_mri_pgm/yes/000002_Y100.pgm
```

**CPU-only inference** (for debugging or GPU unavailability):
```bash
./build/default/Release/brain_mri_pgm.exe \
  --load checkpoints/brain_mri_hybrid.ckpt \
  --infer-only \
  --size 64 \
  --image data/brain_mri_pgm/yes/000002_Y100.pgm \
  --cpu
```

### Custom Image Size

The model expects images of a specific size. Verify the checkpoint was trained with `--size 64`:

```bash
./build/default/Release/brain_mri_pgm.exe \
  --load checkpoints/brain_mri_hybrid.ckpt \
  --infer-only \
  --size 64 \
  --image data/brain_mri_pgm/yes/000002_Y100.pgm
```

## Batch Inference Testing

To test the model on all images in the test set, use the test manifest:

```bash
./build/default/Release/brain_mri_pgm.exe \
  --load checkpoints/brain_mri_hybrid.ckpt \
  --test-manifest data/brain_mri_pgm/test_manifest.txt \
  --infer-only \
  --size 64
```

This runs inference on all images listed in the test manifest and outputs accuracy metrics.

## Path Handling on Windows

File paths automatically normalize both forward slashes and backslashes:

**When using Git Bash or Unix-like shells:**
- **Always use forward slashes** (`/`) — this is the recommended approach
  ```bash
  ./build/default/Release/brain_mri_pgm.exe --load checkpoints/brain_mri_hybrid.ckpt --infer-only --size 64 --image data/brain_mri_pgm/yes/000002_Y100.pgm
  ```

- If you want to use backslashes, **quote the path**:
  ```bash
  ./build/default/Release/brain_mri_pgm.exe --load 'checkpoints\brain_mri_hybrid.ckpt' --infer-only --size 64 --image 'data\brain_mri_pgm\yes\000002_Y100.pgm'
  ```

**When using cmd.exe or PowerShell:**
- Both forward slashes and backslashes work directly:
  ```batch
  build\default\Release\brain_mri_pgm.exe --load checkpoints\brain_mri_hybrid.ckpt --infer-only --size 64 --image data\brain_mri_pgm\yes\000002_Y100.pgm
  ```

**Note on Git Bash:** Without quotes, backslashes are treated as escape characters by the shell and are stripped before reaching the application. The application's path normalization will then receive a malformed path.

## Troubleshooting

### Error: "Failed to read image: ..."

**Cause:** Image file not found or path is incorrect.

**Solution:**
- Verify image file exists: `ls data/brain_mri_pgm/yes/` (on Git Bash) or `dir data\brain_mri_pgm\yes\` (on cmd.exe)
- Check file extension is `.pgm`
- Ensure path uses forward slashes or let the application normalize them

### Error: "Could not open checkpoint for reading: ..."

**Cause:** Checkpoint file not found.

**Solution:**
- Verify checkpoint exists: `ls checkpoints/brain_mri_hybrid.ckpt`
- Check checkpoint was created during training with `--save` flag

### Error: "CUDA not available" or "GPU inference failed"

**Cause:** GPU/CUDA not available on your machine.

**Solution:**
- Use `--cpu` flag to fall back to CPU inference:
  ```bash
  ./build/default/Release/brain_mri_pgm.exe \
    --load checkpoints/brain_mri_hybrid.ckpt \
    --infer-only \
    --size 64 \
    --image data/brain_mri_pgm/yes/000002_Y100.pgm \
    --cpu
  ```

### Backend says "CPU/CPU" instead of "GPU/GPU"

**Cause:** CUDA not available or disabled at build time.

**Solution:**
- Rebuild with CUDA enabled:
  ```bash
  cmake --preset default -DNN_FORCE_CPU=OFF
  cmake --build --preset default --config Release --target brain_mri_pgm
  ```

## Performance Metrics

When running batch inference with a test manifest, the application reports:

- **Accuracy:** Percentage of correct predictions
- **Processing time:** How long inference took (useful for performance benchmarking)

Example output:
```
Test Accuracy: 92.5%
```

## Distribution: Inference Package

### ZIP Package Contents

```
project/
├── build/default/Release/
│   └── brain_mri_pgm.exe
├── checkpoints/
│   └── brain_mri_hybrid.ckpt
├── data/brain_mri_pgm/
│   ├── yes/
│   │   └── *.pgm files
│   └── no/
│       └── *.pgm files
└── README_INFERENCE.txt
```

### System Requirements

**CPU Mode (No External Dependencies):**
- Windows 10/11
- Use `--cpu` flag for all commands

**GPU Mode (Requires CUDA):**
- Windows 10/11  
- NVIDIA GPU (compute capability 7.5+)
- CUDA 13.2 Toolkit: https://developer.nvidia.com/cuda-toolkit-archive

### Inference Execution

**Extract and run:**

```bash
build\default\Release\brain_mri_pgm.exe --load checkpoints/brain_mri_hybrid.ckpt --infer-only --size 64 --image data/brain_mri_pgm/yes/000002_Y100.pgm --cpu
```

**Output:**
```
Mode: infer-only
Backend (conv/dense): CPU/CPU
Loaded checkpoint: checkpoints/brain_mri_hybrid.ckpt
Image probability(tumor=yes): 0.XXXXXX
Prediction: yes/no
```

### Included Instructions

Add `README_INFERENCE.txt` with:

```
BRAIN MRI MODEL - INFERENCE PACKAGE
====================================

SYSTEM REQUIREMENTS:
- Windows 10/11
- CPU mode: No external dependencies
- GPU mode: CUDA 13.2 required

QUICK START:
1. Extract ZIP
2. Open Command Prompt
3. Run:
   build\default\Release\brain_mri_pgm.exe --load checkpoints/brain_mri_hybrid.ckpt --infer-only --size 64 --image data/brain_mri_pgm/yes/000002_Y100.pgm --cpu

TEST DATA:
- Positive samples: data/brain_mri_pgm/yes/
- Negative samples: data/brain_mri_pgm/no/

COMMAND REFERENCE:

Single image (CPU):
  build\default\Release\brain_mri_pgm.exe --load checkpoints/brain_mri_hybrid.ckpt --infer-only --size 64 --image data/brain_mri_pgm/yes/000002_Y100.pgm --cpu

Single image (GPU):
  build\default\Release\brain_mri_pgm.exe --load checkpoints/brain_mri_hybrid.ckpt --infer-only --size 64 --image data/brain_mri_pgm/yes/000002_Y100.pgm

Batch inference:
  build\default\Release\brain_mri_pgm.exe --load checkpoints/brain_mri_hybrid.ckpt --test-manifest data/brain_mri_pgm/test_manifest.txt --infer-only --size 64 --cpu
```

## Additional Resources

- Training workflow: [README.md](README.md)
- Performance benchmarks: `./build/default/Release/benchmark_conv2d.exe`
- Custom image preprocessing: `python tools/preprocess_brain_mri.py`
