#include <algorithm>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#define NN_USE_CUDA_DOT 1
#include "nn/backend/cuda_dot.h"

namespace nn_cuda {
namespace {
inline void cuda_check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

struct DeviceBuffer {
    double* ptr;
    std::size_t bytes;

    DeviceBuffer() : ptr(nullptr), bytes(0) {}

    explicit DeviceBuffer(std::size_t nbytes) : ptr(nullptr), bytes(nbytes) {
        cuda_check(cudaMalloc(&ptr, bytes), "cudaMalloc(DeviceBuffer)");
    }

    ~DeviceBuffer() {
        if (ptr != nullptr)
            cudaFree(ptr);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

inline void validate_rectangular(const np::Matrix& matrix, const char* function_name) {
    if (matrix.empty() || matrix[0].empty())
        throw std::invalid_argument(std::string(function_name) + " expects non-empty matrix");

    const std::size_t cols = matrix[0].size();
    for (std::size_t r = 0; r < matrix.size(); ++r) {
        if (matrix[r].size() != cols)
            throw std::invalid_argument(std::string(function_name) + " expects rectangular matrix");
    }
}

inline std::vector<double> flatten_row_major(const np::Matrix& matrix) {
    std::vector<double> out;
    out.reserve(matrix.size() * matrix[0].size());
    for (std::size_t r = 0; r < matrix.size(); ++r)
        for (std::size_t c = 0; c < matrix[r].size(); ++c)
            out.push_back(matrix[r][c]);
    return out;
}

inline np::Matrix unflatten_row_major(const std::vector<double>& flat, std::size_t rows, std::size_t cols) {
    np::Matrix matrix(rows, np::Vector(cols, 0.0));
    std::size_t index = 0;
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < cols; ++c)
            matrix[r][c] = flat[index++];
    return matrix;
}

template <int TILE>
__global__ void matmul_tiled_kernel(const double* a, const double* b, double* c, int m, int k, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ double a_tile[TILE][TILE];
    __shared__ double b_tile[TILE][TILE];

    double sum = 0.0;
    const int tiles = (k + TILE - 1) / TILE;

    for (int t = 0; t < tiles; ++t) {
        const int a_col = t * TILE + threadIdx.x;
        const int b_row = t * TILE + threadIdx.y;

        a_tile[threadIdx.y][threadIdx.x] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0;
        b_tile[threadIdx.y][threadIdx.x] = (b_row < k && col < n) ? b[b_row * n + col] : 0.0;

        __syncthreads();

        for (int i = 0; i < TILE; ++i)
            sum += a_tile[threadIdx.y][i] * b_tile[i][threadIdx.x];

        __syncthreads();
    }

    if (row < m && col < n)
        c[row * n + col] = sum;
}
} // namespace

bool is_available() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess)
        return false;
    return count > 0;
}

np::Matrix dot(const np::Matrix& a, const np::Matrix& b) {
    if (!is_available())
        throw std::runtime_error("nn_cuda::dot called without an available CUDA device");

    validate_rectangular(a, "nn_cuda::dot");
    validate_rectangular(b, "nn_cuda::dot");

    const int m = static_cast<int>(a.size());
    const int k = static_cast<int>(a[0].size());
    const int b_rows = static_cast<int>(b.size());
    const int n = static_cast<int>(b[0].size());

    if (k != b_rows)
        throw std::invalid_argument("nn_cuda::dot dimension mismatch");

    std::vector<double> h_a = flatten_row_major(a);
    std::vector<double> h_b = flatten_row_major(b);
    std::vector<double> h_c(static_cast<std::size_t>(m) * static_cast<std::size_t>(n), 0.0);

    const std::size_t bytes_a = sizeof(double) * h_a.size();
    const std::size_t bytes_b = sizeof(double) * h_b.size();
    const std::size_t bytes_c = sizeof(double) * h_c.size();

    DeviceBuffer d_a(bytes_a);
    DeviceBuffer d_b(bytes_b);
    DeviceBuffer d_c(bytes_c);

    cuda_check(cudaMemcpy(d_a.ptr, h_a.data(), bytes_a, cudaMemcpyHostToDevice), "cudaMemcpy H2D a");
    cuda_check(cudaMemcpy(d_b.ptr, h_b.data(), bytes_b, cudaMemcpyHostToDevice), "cudaMemcpy H2D b");

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    matmul_tiled_kernel<16><<<grid, block>>>(d_a.ptr, d_b.ptr, d_c.ptr, m, k, n);
    cuda_check(cudaGetLastError(), "matmul_tiled_kernel launch");
    cuda_check(cudaDeviceSynchronize(), "matmul_tiled_kernel sync");

    cuda_check(cudaMemcpy(h_c.data(), d_c.ptr, bytes_c, cudaMemcpyDeviceToHost), "cudaMemcpy D2H c");

    return unflatten_row_major(h_c, static_cast<std::size_t>(m), static_cast<std::size_t>(n));
}

} // namespace nn_cuda
