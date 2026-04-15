#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "nn/backend/cuda_dot.h"
#include "nn/layers/dense.h"
#include "nn/math/matrix.h"

using nn::Dense;

static double max_abs_diff(const np::Matrix& a, const np::Matrix& b) {
    if (a.size() != b.size() || (a.empty() ? false : a[0].size() != b[0].size())) {
        throw std::runtime_error("shape mismatch in max_abs_diff");
    }

    double diff = 0.0;
    for (std::size_t r = 0; r < a.size(); ++r)
        for (std::size_t c = 0; c < a[r].size(); ++c)
            diff = std::max(diff, std::fabs(a[r][c] - b[r][c]));
    return diff;
}

int main() {
    try {
        if (!nn_cuda::is_available()) {
            std::cout << "test_dense_gpu skipped: CUDA backend not available\n";
            return 0;
        }

        Dense dense_cpu(4, 3);
        Dense dense_gpu(4, 3);

        dense_gpu.set_weights(dense_cpu.get_weights());
        dense_gpu.set_bias(dense_cpu.get_bias());

        dense_cpu.set_use_gpu(false);
        dense_gpu.set_use_gpu(true);

        // Dense currently assumes column-vector style batches (n x 1) because bias add has no broadcasting.
        const np::Matrix input = np::randn(4, 1);
        const np::Matrix grad_out = np::randn(3, 1);

        const np::Matrix out_cpu = dense_cpu.forward(input);
        const np::Matrix out_gpu = dense_gpu.forward(input);
        const double out_diff = max_abs_diff(out_cpu, out_gpu);

        const np::Matrix in_grad_cpu = dense_cpu.backward(grad_out, 0.01);
        const np::Matrix in_grad_gpu = dense_gpu.backward(grad_out, 0.01);
        const double grad_diff = max_abs_diff(in_grad_cpu, in_grad_gpu);

        const double w_diff = max_abs_diff(dense_cpu.get_weights(), dense_gpu.get_weights());
        const double b_diff = max_abs_diff(dense_cpu.get_bias(), dense_gpu.get_bias());

        if (out_diff > 1e-8 || grad_diff > 1e-8 || w_diff > 1e-8 || b_diff > 1e-8) {
            throw std::runtime_error("Dense CPU/GPU mismatch above tolerance");
        }

        std::cout << "test_dense_gpu passed\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "test_dense_gpu failed: " << ex.what() << "\n";
        return 1;
    }
}
