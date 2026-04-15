# Contributing

Thanks for contributing.

## Development workflow

1. Create a feature branch.
2. Keep changes focused and small.
3. Build and run tests locally before opening a PR.
4. Open a pull request with a clear description and test notes.

## Local validation

Use the same CPU-only configuration that CI uses:

```bash
cmake -S . -B build/ci -DNN_ENABLE_TESTS=ON -DNN_ENABLE_CUDA=OFF -DNN_FORCE_CPU=ON -DNN_BUILD_BRAIN_MRI=OFF -DNN_BUILD_BRAIN_MRI_PGM=ON -DNN_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/ci --config Release --parallel
ctest --test-dir build/ci --build-config Release --output-on-failure
```

## Coding style

- Follow existing C++ style in the touched files.
- Avoid unrelated refactors in the same PR.
- Add tests for behavior changes when possible.

## Commit messages

Use concise imperative messages, for example:

- `Fix CUDA fallback in convolution layer`
- `Add test for MNIST IDX loader`
