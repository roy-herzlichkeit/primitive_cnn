#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "nn/backend/cuda_conv2d.h"
#include "nn/backend/cuda_dot.h"
#include "nn/layers/convolutional.h"
#include "nn/math/matrix.h"
#include "nn/math/signal2d.h"

using Matrix = np::Matrix;
using Tensor3D = std::vector<Matrix>;
using Tensor4D = std::vector<Tensor3D>;
using nn::Convolutional;

static double max_abs_diff_matrix(const Matrix& a, const Matrix& b) {
    if (a.size() != b.size() || (a.empty() ? false : a[0].size() != b[0].size()))
        throw std::runtime_error("shape mismatch in max_abs_diff_matrix");

    double diff = 0.0;
    for (std::size_t r = 0; r < a.size(); ++r)
        for (std::size_t c = 0; c < a[r].size(); ++c)
            diff = std::max(diff, std::fabs(a[r][c] - b[r][c]));
    return diff;
}

static double max_abs_diff_tensor3(const Tensor3D& a, const Tensor3D& b) {
    if (a.size() != b.size())
        throw std::runtime_error("shape mismatch in max_abs_diff_tensor3");

    double diff = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        diff = std::max(diff, max_abs_diff_matrix(a[i], b[i]));
    return diff;
}

static double max_abs_diff_tensor4(const Tensor4D& a, const Tensor4D& b) {
    if (a.size() != b.size())
        throw std::runtime_error("shape mismatch in max_abs_diff_tensor4");

    double diff = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        diff = std::max(diff, max_abs_diff_tensor3(a[i], b[i]));
    return diff;
}

int main() {
    try {
        if (!nn_cuda::is_available()) {
            std::cout << "test_conv_gpu skipped: CUDA backend not available\n";
            return 0;
        }

        {
            const Matrix input = np::randn(16, 16);
            const Matrix kernel = np::randn(5, 5);

            const Matrix cpu_corr = scipy_signal::correlate2d(input, kernel, "valid");
            const Matrix gpu_corr = nn_cuda::correlate2d_valid(input, kernel);
            const Matrix cpu_full = scipy_signal::convolve2d(input, kernel, "full");
            const Matrix gpu_full = nn_cuda::convolve2d_full(input, kernel);

            if (max_abs_diff_matrix(cpu_corr, gpu_corr) > 1e-8 || max_abs_diff_matrix(cpu_full, gpu_full) > 1e-8)
                throw std::runtime_error("CUDA conv2d primitive mismatch above tolerance");
        }

        Convolutional conv_cpu({1, 8, 8}, 3, 2);
        Convolutional conv_gpu({1, 8, 8}, 3, 2);

        conv_gpu.set_kernels(conv_cpu.get_kernels());
        conv_gpu.set_biases(conv_cpu.get_biases());

        conv_cpu.set_use_gpu(false);
        conv_gpu.set_use_gpu(true);

        const Tensor3D input(1, np::randn(8, 8));
        const Tensor3D grad_out(2, np::randn(6, 6));

        const Tensor3D out_cpu = conv_cpu.forward(input);
        const Tensor3D out_gpu = conv_gpu.forward(input);

        const Tensor3D in_grad_cpu = conv_cpu.backward(grad_out, 0.01);
        const Tensor3D in_grad_gpu = conv_gpu.backward(grad_out, 0.01);

        const double out_diff = max_abs_diff_tensor3(out_cpu, out_gpu);
        const double in_grad_diff = max_abs_diff_tensor3(in_grad_cpu, in_grad_gpu);
        const double kernel_diff = max_abs_diff_tensor4(conv_cpu.get_kernels(), conv_gpu.get_kernels());
        const double bias_diff = max_abs_diff_tensor3(conv_cpu.get_biases(), conv_gpu.get_biases());

        if (out_diff > 1e-8 || in_grad_diff > 1e-8 || kernel_diff > 1e-8 || bias_diff > 1e-8)
            throw std::runtime_error("Convolutional CPU/GPU mismatch above tolerance");

        std::cout << "test_conv_gpu passed\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "test_conv_gpu failed: " << ex.what() << "\n";
        return 1;
    }
}
