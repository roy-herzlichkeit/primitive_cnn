# Architecture Notes

## Overview

The project is split into clear layers of responsibility:

- Public interfaces: `include/nn/*`
- Implementations: `src/*`
- Executable entry points: `apps/*`
- Validation: `tests/*`, `benchmarks/*`

## Core Modules

### Math (`include/nn/math`)

- `matrix.h`: matrix primitives (`dot`, `transpose`, elementwise ops, random init)
- `signal2d.h`: CPU reference correlation/convolution helpers

### Loss (`include/nn/loss`)

- `losses.h`: MSE and BCE plus derivatives

### Layers (`include/nn/layers`, `src/layers`)

- `Dense`: linear transform and SGD update
- `Convolutional`: valid correlation forward and full-convolution gradient propagation
- `Reshape`: deterministic tensor/matrix shape conversion
- `Activation`, `Tanh`, `Sigmoid`, `Softmax`: nonlinearities and backprop support

### Backend (`include/nn/backend`, `src/backend`)

- `cuda_dot`: optional CUDA matrix multiplication
- `cuda_conv2d`: optional CUDA correlation/convolution kernels

## Build Model

- `nn_core` static library contains the reusable neural-network implementation.
- Optional `nn_cuda_backend` is linked when CUDA is enabled and available.
- Apps/tests/benchmarks depend on `nn_core` to avoid code duplication.

## Naming and Style

- Types use `PascalCase` (`Dense`, `Convolutional`, `Reshape`).
- Functions and variables use `snake_case`.
- Private members use trailing underscore (`weights_`, `input_cache_`).
- Code avoids hidden side effects and keeps functions focused on one task.

## Compatibility

Legacy headers under `utils/` are wrappers that include the new API so older include paths continue to work while the codebase transitions.
