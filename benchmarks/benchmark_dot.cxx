#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "nn/backend/cuda_dot.h"
#include "nn/math/matrix.h"

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
    const int m = (argc > 1) ? std::atoi(argv[1]) : 512;
    const int k = (argc > 2) ? std::atoi(argv[2]) : 512;
    const int n = (argc > 3) ? std::atoi(argv[3]) : 512;
    const int iters = (argc > 4) ? std::atoi(argv[4]) : 20;

    np::Matrix a = np::randn(m, k);
    np::Matrix b = np::randn(k, n);

    // Warm-up CPU path.
    np::Matrix cpu_out = np::dot(a, b);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i)
        cpu_out = np::dot(a, b);
    auto t1 = std::chrono::high_resolution_clock::now();

    const double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / static_cast<double>(iters);
    std::cout << "CPU dot avg: " << cpu_ms << " ms\n";

    if (!nn_cuda::is_available()) {
        std::cout << "GPU dot unavailable (CUDA backend not enabled or no CUDA device).\n";
        return 0;
    }

    np::Matrix gpu_out;
    gpu_out = nn_cuda::dot(a, b); // warm-up GPU path

    auto g0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i)
        gpu_out = nn_cuda::dot(a, b);
    auto g1 = std::chrono::high_resolution_clock::now();

    const double gpu_ms = std::chrono::duration<double, std::milli>(g1 - g0).count() / static_cast<double>(iters);
    const double diff = max_abs_diff(cpu_out, gpu_out);

    std::cout << "GPU dot avg: " << gpu_ms << " ms\n";
    std::cout << "Speedup (CPU/GPU): " << (cpu_ms / gpu_ms) << "x\n";
    std::cout << "Max abs diff: " << diff << "\n";

    return 0;
}
