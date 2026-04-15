# Neural Network Assignment (C++ / CMake)

This project provides a from-scratch neural-network implementation with optional CUDA acceleration and reproducible CMake workflows.

## Refactored Architecture

The codebase is now organized with clear interface/implementation boundaries:

```text
apps/                 # Executable entry points
	xor.cxx
	mnist.cxx
	mnist_conv.cxx
	brain_mri.cxx

include/nn/           # Public API (headers)
	backend/
	layers/
	loss/
	math/

src/                  # Implementations
	backend/
	layers/

tests/                # Unit/integration tests (CTest)
benchmarks/           # Performance micro-benchmarks
docs/                 # Project documentation
```

### Design Principles Applied

- Interfaces are in `include/nn`, implementation files are in `src`.
- Layer modules (`Dense`, `Convolutional`, `Reshape`, activations) each have one clear responsibility.
- Shared math, signal, loss, and backend concerns are modularized into separate subsystems.
- All applications/tests/benchmarks consume a single core target: `nn_core`.
- Legacy headers under `utils/` remain as compatibility wrappers to the new API.

## Build and Test

### GitHub CI (portable defaults)

GitHub Actions builds and tests in CPU-only mode (no CUDA, no OpenCV) to keep checks reproducible across hosted runners.

Run the same validation locally before pushing:

```bash
cmake -S . -B build/ci -DNN_ENABLE_TESTS=ON -DNN_ENABLE_CUDA=OFF -DNN_FORCE_CPU=ON -DNN_BUILD_BRAIN_MRI=OFF -DNN_BUILD_BRAIN_MRI_PGM=ON -DNN_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/ci --config Release --parallel
ctest --test-dir build/ci --build-config Release --output-on-failure
```

### Recommended (GPU-first)

```bash
cmake --preset default
cmake --build --preset default
ctest --preset default
```

### CPU-only mode

```bash
cmake --preset cpu-no-opencv
cmake --build --preset cpu-no-opencv
ctest --test-dir build/cpu-no-opencv -C Debug --output-on-failure
```

### CUDA preset workflow

```bash
cmake --preset cuda-ready
cmake --build --preset cuda-ready
ctest --preset cuda-ready
```

## Run Examples

```bash
./build/default/Debug/xor.exe
./build/default/Debug/mnist.exe
./build/default/Debug/mnist_conv.exe
./build/default/Debug/mnist_conv.exe --cpu
./build/default/Debug/benchmark_dot.exe 256 256 256 10
./build/default/Debug/benchmark_conv2d.exe 128 128 3 10
```

If OpenCV is installed and the `brain_mri` target is enabled:

```bash
./build/default/Debug/brain_mri.exe --data data/brain_mri --size 64 --epochs 10 --lr 0.01
./build/default/Debug/brain_mri.exe --size 64 --load checkpoints/brain_mri.ckpt --infer-only --image data/brain_mri/yes/Y1.jpg
```

## Brain MRI Without OpenCV in C++ (Python + C++ Hybrid)

If your C++ OpenCV build is blocked, use this flow:

1. Use Python + OpenCV only for image preprocessing into `.pgm` files + train/test manifests.
2. Train and infer in pure C++ with `brain_mri_pgm` (no OpenCV link dependency).

### 1) Preprocess dataset in Python

Install Python dependency (one-time):

```bash
python -m pip install opencv-python
```

Run preprocessing:

```bash
python tools/preprocess_brain_mri.py --input data/brain_mri --output data/brain_mri_pgm --size 64 --train-ratio 0.8
```

Optional single-image preprocessing for inference:

```bash
python tools/preprocess_brain_mri.py --input data/brain_mri --output data/brain_mri_pgm --size 64 --image data/brain_mri/yes/Y1.jpg --image-out data/brain_mri_pgm/single_image.pgm
```

### 2) Build C++ app without OpenCV target

```bash
cmake --preset default -DNN_BUILD_BRAIN_MRI=OFF -DNN_BUILD_BRAIN_MRI_PGM=ON
cmake --build --preset default --config Release --target brain_mri_pgm
```

### CUDA + Python OpenCV hybrid flow (recommended)

Use Python OpenCV only for preprocessing, then run C++ training/inference on CUDA:

```bash
python -m pip install opencv-python
python tools/preprocess_brain_mri.py --input data/brain_mri --output data/brain_mri_pgm --size 64 --train-ratio 0.8

cmake --preset default -DNN_BUILD_BRAIN_MRI=OFF -DNN_BUILD_BRAIN_MRI_PGM=ON -DNN_FORCE_CPU=OFF
cmake --build --preset default --config Release --target brain_mri_pgm

./build/default/Release/brain_mri_pgm.exe --train-manifest data/brain_mri_pgm/train_manifest.txt --test-manifest data/brain_mri_pgm/test_manifest.txt --size 64 --epochs 10 --lr 0.01 --save checkpoints/brain_mri_hybrid.ckpt
./build/default/Release/brain_mri_pgm.exe --load checkpoints/brain_mri_hybrid.ckpt --infer-only --size 64 --image data/brain_mri_pgm/single_image.pgm
```

Expected startup line when CUDA is active:

```text
Backend (conv/dense): GPU/GPU (auto)
```

If you want CPU fallback for troubleshooting, add `--cpu` to `brain_mri_pgm.exe` commands.

### 3) Train/evaluate in C++

```bash
./build/default/Release/brain_mri_pgm.exe --train-manifest data/brain_mri_pgm/train_manifest.txt --test-manifest data/brain_mri_pgm/test_manifest.txt --size 64 --epochs 10 --lr 0.01 --save checkpoints/brain_mri.ckpt
```

### 4) Infer in C++

```bash
./build/default/Release/brain_mri_pgm.exe --load checkpoints/brain_mri.ckpt --infer-only --size 64 --image data/brain_mri_pgm/single_image.pgm
```

## Datasets

### MNIST (IDX)

- `data/mnist/train-images-idx3-ubyte` or `data/mnist/train-images.idx3-ubyte`
- `data/mnist/train-labels-idx1-ubyte` or `data/mnist/train-labels.idx1-ubyte`
- `data/mnist/t10k-images-idx3-ubyte` or `data/mnist/t10k-images.idx3-ubyte`
- `data/mnist/t10k-labels-idx1-ubyte` or `data/mnist/t10k-labels.idx1-ubyte`

### Brain MRI

- `data/brain_mri/yes/*.jpg|png|jpeg`
- `data/brain_mri/no/*.jpg|png|jpeg`

## Documentation

- Machine-specific run/test guide: `docs/windows_machine_run_test.md`

## Project Governance

- Contributing guide: `CONTRIBUTING.md`
- Security policy: `SECURITY.md`