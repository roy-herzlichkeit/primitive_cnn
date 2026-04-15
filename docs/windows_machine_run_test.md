# Windows Machine Run/Test Guide (This Workspace)

This guide is tailored for this machine and workspace:
- OS: Windows
- Workspace: `E:/Projects/_neural_network_`
- Generator: Visual Studio 18 2026 (x64)
- CUDA toolkit: `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2`
- GPU target: RTX 4060 (SM 89)

## 1) Known-good default flow (GPU-first)
From `E:/Projects/_neural_network_`:

```bash
cmake --preset default
cmake --build --preset default
ctest --preset default
```

Expected:
- CUDA is enabled during configure.
- All tests pass, including `test_conv_gpu`.

## 2) CPU-only flow (explicit opt-out)

```bash
cmake --preset cpu-no-opencv
cmake --build --preset cpu-no-opencv
ctest --test-dir build/cpu-no-opencv -C Debug --output-on-failure
```

Notes:
- This forces CPU mode (`NN_FORCE_CPU=ON`, `NN_ENABLE_CUDA=OFF`).
- OpenCV-dependent `brain_mri` target is skipped in this preset.

## 3) Manual CUDA configure (if not using presets)
Use this exact command shape on this machine:

```bash
TMP="$PWD/.tmp" TEMP="$PWD/.tmp" cmake -S . -B build/cuda -G "Visual Studio 18 2026" -A x64 -T "cuda=$CUDA_PATH" -DNN_ENABLE_CUDA=ON -DNN_BUILD_BRAIN_MRI=OFF -DNN_BUILD_BENCHMARKS=ON -DNN_ENABLE_TESTS=ON
cmake --build build/cuda -j
ctest --test-dir build/cuda -C Debug --output-on-failure
```

Why this matters:
- `-A x64 -T "cuda=..."` is required for reliable NVCC detection with VS generators.
- `TMP/TEMP` avoids MSBuild temp-permission failures seen on this machine.

## 4) Verify runtime backend selection

### Convolution model, auto backend
```bash
./build/default/Debug/mnist_conv.exe
```
Expected first line includes:
- `Backend (conv/dense): GPU/GPU (auto)`

### Convolution model, forced CPU
```bash
./build/default/Debug/mnist_conv.exe --cpu
```
Expected first line includes:
- `Backend (conv/dense): CPU/CPU (forced CPU)`

## 5) Verify GPU speedup

```bash
./build/default/Debug/benchmark_dot.exe 512 512 512 10
./build/default/Debug/benchmark_conv2d.exe 512 512 5 10
```

Expected:
- GPU times lower than CPU on larger sizes.
- Very small numeric difference (around `1e-14` to `1e-8` scale).

## 6) Brain MRI target note for this machine
Current configure warns:
- OpenCV not found, so `brain_mri` target is skipped.

If you want Brain MRI executable on this machine, install OpenCV and reconfigure with `NN_BUILD_BRAIN_MRI=ON`.

## 7) Brain MRI with CUDA + Python OpenCV (hybrid path)
Use this flow when you want OpenCV preprocessing and CUDA training/inference without linking OpenCV into C++.

### Step A: Python preprocessing (OpenCV)
```bash
python -m pip install opencv-python
python tools/preprocess_brain_mri.py --input data/brain_mri --output data/brain_mri_pgm --size 64 --train-ratio 0.8
python tools/preprocess_brain_mri.py --input data/brain_mri --output data/brain_mri_pgm --size 64 --image data/brain_mri/yes/Y1.jpg --image-out data/brain_mri_pgm/single_image.pgm
```

### Step B: Configure/build C++ with CUDA enabled and OpenCV C++ target disabled
```bash
cmake --preset default -DNN_BUILD_BRAIN_MRI=OFF -DNN_BUILD_BRAIN_MRI_PGM=ON -DNN_FORCE_CPU=OFF
cmake --build --preset default --config Release --target brain_mri_pgm
```

### Step C: Train on GPU
```bash
./build/default/Release/brain_mri_pgm.exe --train-manifest data/brain_mri_pgm/train_manifest.txt --test-manifest data/brain_mri_pgm/test_manifest.txt --size 64 --epochs 10 --lr 0.01 --save checkpoints/brain_mri_hybrid.ckpt
```

### Step D: Inference on GPU
```bash
./build/default/Release/brain_mri_pgm.exe --load checkpoints/brain_mri_hybrid.ckpt --infer-only --size 64 --image data/brain_mri_pgm/single_image.pgm
```

Expected backend line when CUDA is active:
- `Backend (conv/dense): GPU/GPU (auto)`

Troubleshooting:
- Add `--cpu` to force CPU fallback if CUDA runtime or drivers are unavailable.
- Ensure preprocess `--size` matches C++ `--size`.
