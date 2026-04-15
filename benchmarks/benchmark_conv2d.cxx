#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "nn/backend/cuda_conv2d.h"
#include "nn/backend/cuda_dot.h"
#include "nn/math/matrix.h"
#include "nn/math/signal2d.h"

static double max_abs_diff(const np::Matrix& a, const np::Matrix& b) {
    if (a.size() != b.size() || (a.empty() ? false : a[0].size() != b[0].size()))
        throw std::runtime_error("shape mismatch in max_abs_diff");

    double diff = 0.0;
    for (std::size_t r = 0; r < a.size(); ++r)
        for (std::size_t c = 0; c < a[r].size(); ++c)
            diff = std::max(diff, std::fabs(a[r][c] - b[r][c]));
    return diff;
}

int main(int argc, char** argv) {
    const int h = (argc > 1) ? std::atoi(argv[1]) : 256;
    const int w = (argc > 2) ? std::atoi(argv[2]) : 256;
    const int k = (argc > 3) ? std::atoi(argv[3]) : 5;
    const int iters = (argc > 4) ? std::atoi(argv[4]) : 20;

    np::Matrix input = np::randn(h, w);
    np::Matrix kernel = np::randn(k, k);

    np::Matrix cpu_corr = scipy_signal::correlate2d(input, kernel, "valid");
    auto c0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i)
        cpu_corr = scipy_signal::correlate2d(input, kernel, "valid");
    auto c1 = std::chrono::high_resolution_clock::now();
    const double cpu_corr_ms = std::chrono::duration<double, std::milli>(c1 - c0).count() / static_cast<double>(iters);

    np::Matrix cpu_full = scipy_signal::convolve2d(input, kernel, "full");
    auto c2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i)
        cpu_full = scipy_signal::convolve2d(input, kernel, "full");
    auto c3 = std::chrono::high_resolution_clock::now();
    const double cpu_full_ms = std::chrono::duration<double, std::milli>(c3 - c2).count() / static_cast<double>(iters);

    std::cout << "CPU correlate2d(valid) avg: " << cpu_corr_ms << " ms\n";
    std::cout << "CPU convolve2d(full) avg: " << cpu_full_ms << " ms\n";

    if (!nn_cuda::is_available()) {
        std::cout << "GPU conv unavailable (CUDA backend not enabled or no CUDA device).\n";
        return 0;
    }

    np::Matrix gpu_corr = nn_cuda::correlate2d_valid(input, kernel);
    auto g0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i)
        gpu_corr = nn_cuda::correlate2d_valid(input, kernel);
    auto g1 = std::chrono::high_resolution_clock::now();
    const double gpu_corr_ms = std::chrono::duration<double, std::milli>(g1 - g0).count() / static_cast<double>(iters);

    np::Matrix gpu_full = nn_cuda::convolve2d_full(input, kernel);
    auto g2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i)
        gpu_full = nn_cuda::convolve2d_full(input, kernel);
    auto g3 = std::chrono::high_resolution_clock::now();
    const double gpu_full_ms = std::chrono::duration<double, std::milli>(g3 - g2).count() / static_cast<double>(iters);

    std::cout << "GPU correlate2d(valid) avg: " << gpu_corr_ms << " ms\n";
    std::cout << "GPU convolve2d(full) avg: " << gpu_full_ms << " ms\n";
    std::cout << "Correlate speedup (CPU/GPU): " << (cpu_corr_ms / gpu_corr_ms) << "x\n";
    std::cout << "Full conv speedup (CPU/GPU): " << (cpu_full_ms / gpu_full_ms) << "x\n";

    std::cout << "Correlate max abs diff: " << max_abs_diff(cpu_corr, gpu_corr) << "\n";
    std::cout << "Full conv max abs diff: " << max_abs_diff(cpu_full, gpu_full) << "\n";

    return 0;
}
