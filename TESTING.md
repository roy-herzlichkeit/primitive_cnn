# Model Inference Testing Guide

This document provides step-by-step instructions for testing the brain MRI model in inference-only mode using a pre-trained checkpoint.

## Prerequisites

1. **Build the project** with the brain MRI PGM target:
   ```bash
   cmake --preset default -DNN_BUILD_BRAIN_MRI=OFF -DNN_BUILD_BRAIN_MRI_PGM=ON
   cmake --build --preset default --config Release --target brain_mri_pgm
   ```

2. **Checkpoint available:** Ensure `checkpoints/brain_mri_hybrid.ckpt` exists (pre-trained model).

3. **Test images available:** Images in `data/brain_mri_pgm/` directory (preprocessed as `.pgm` files).

## Quick Start: Inference with a Single Image

Run inference on a tumor image:

```bash
./build/default/Release/brain_mri_pgm.exe \
  --load checkpoints/brain_mri_hybrid.ckpt \
  --infer-only \
  --size 64 \
  --image data/brain_mri_pgm/yes/000002_Y100.pgm
```

Run inference on a non-tumor image:

```bash
./build/default/Release/brain_mri_pgm.exe \
  --load checkpoints/brain_mri_hybrid.ckpt \
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

## Distribution: Sharing the Model as ZIP

If you want to send this project to a friend for inference-only testing, follow this guide.

### ZIP Structure

Include the following in your ZIP file:

```
project/
├── build/default/Release/
│   └── brain_mri_pgm.exe
├── checkpoints/
│   └── brain_mri_hybrid.ckpt  (add your pre-trained checkpoint here)
├── data/brain_mri_pgm/
│   ├── yes/
│   │   └── *.pgm image files
│   └── no/
│       └── *.pgm image files
└── INFERENCE_QUICK_START.txt  (optional: see section below)
```

### System Requirements for Your Friend

**Option 1: GPU Inference (requires CUDA)**
- Windows 10/11
- NVIDIA GPU with compute capability 7.5 or higher
- CUDA 13.2 Toolkit installed
  - Download: https://developer.nvidia.com/cuda-toolkit-archive

**Option 2: CPU Inference (recommended—no external dependencies)**
- Windows 10/11
- Just add `--cpu` flag to all commands
- No CUDA, NVIDIA drivers, or GPU required

### Quick Start for Your Friend

**Extract ZIP and run this command:**

```bash
build\default\Release\brain_mri_pgm.exe --load checkpoints/brain_mri_hybrid.ckpt --infer-only --size 64 --image data/brain_mri_pgm/yes/000002_Y100.pgm --cpu
```

**Expected output:**
```
Mode: infer-only
Backend (conv/dense): CPU/CPU
Loaded checkpoint: checkpoints/brain_mri_hybrid.ckpt
Image probability(tumor=yes): 0.XXXXXX
Prediction: yes/no
```

### Create an Instructions File (Optional)

Add `INFERENCE_QUICK_START.txt` to your ZIP with this content:

```
========================================
BRAIN MRI MODEL - INFERENCE ONLY
========================================

REQUIREMENTS:
- Windows 10/11
- No external dependencies if using --cpu flag

QUICK START (CPU, no CUDA needed):
1. Extract ZIP
2. Open Command Prompt in this folder
3. Run:
   build\default\Release\brain_mri_pgm.exe --load checkpoints/brain_mri_hybrid.ckpt --infer-only --size 64 --image data/brain_mri_pgm/yes/000002_Y100.pgm --cpu

TEST IMAGES:
- Tumor samples: data/brain_mri_pgm/yes/
- Normal samples: data/brain_mri_pgm/no/

AVAILABLE COMMANDS:

Single image inference:
  build\default\Release\brain_mri_pgm.exe --load checkpoints/brain_mri_hybrid.ckpt --infer-only --size 64 --image data/brain_mri_pgm/yes/000002_Y100.pgm --cpu

GPU inference (requires CUDA 13.2):
  build\default\Release\brain_mri_pgm.exe --load checkpoints/brain_mri_hybrid.ckpt --infer-only --size 64 --image data/brain_mri_pgm/yes/000002_Y100.pgm

Batch inference on all test images:
  build\default\Release\brain_mri_pgm.exe --load checkpoints/brain_mri_hybrid.ckpt --test-manifest data/brain_mri_pgm/test_manifest.txt --infer-only --size 64 --cpu
```

## Next Steps

- For training a new model, see [README.md](README.md)
- For benchmark comparisons, run `./build/default/Release/benchmark_conv2d.exe`
- To preprocess custom images, use `python tools/preprocess_brain_mri.py`
