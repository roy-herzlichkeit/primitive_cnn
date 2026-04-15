#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#define NN_USE_CUDA_CONV2D 1
#include "nn/backend/cuda_conv2d.h"

namespace nn_cuda {
namespace {
inline void cuda_check(cudaError_t status, const char* what) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
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

__global__ void correlate2d_valid_kernel(
    const double* input,
    const double* kernel,
    double* output,
    int in_h,
    int in_w,
    int k_h,
    int k_w,
    int out_w) {
    const int out_r = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_c = blockIdx.x * blockDim.x + threadIdx.x;

    const int out_h = in_h - k_h + 1;
    if (out_r >= out_h || out_c >= out_w)
        return;

    double sum = 0.0;
    for (int kr = 0; kr < k_h; ++kr) {
        for (int kc = 0; kc < k_w; ++kc) {
            const int in_r = out_r + kr;
            const int in_c = out_c + kc;
            sum += input[in_r * in_w + in_c] * kernel[kr * k_w + kc];
        }
    }

    output[out_r * out_w + out_c] = sum;
}

__global__ void convolve2d_full_kernel(
    const double* input,
    const double* kernel,
    double* output,
    int in_h,
    int in_w,
    int k_h,
    int k_w,
    int out_h,
    int out_w) {
    const int out_r = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_c = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_r >= out_h || out_c >= out_w)
        return;

    double sum = 0.0;
    for (int kr = 0; kr < k_h; ++kr) {
        for (int kc = 0; kc < k_w; ++kc) {
            const int in_r = out_r - kr;
            const int in_c = out_c - kc;
            if (in_r < 0 || in_c < 0 || in_r >= in_h || in_c >= in_w)
                continue;

            const int flipped_kr = k_h - 1 - kr;
            const int flipped_kc = k_w - 1 - kc;
            sum += input[in_r * in_w + in_c] * kernel[flipped_kr * k_w + flipped_kc];
        }
    }

    output[out_r * out_w + out_c] = sum;
}

inline np::Matrix launch_correlate2d_valid(const np::Matrix& input, const np::Matrix& kernel) {
    const int in_h = static_cast<int>(input.size());
    const int in_w = static_cast<int>(input[0].size());
    const int k_h = static_cast<int>(kernel.size());
    const int k_w = static_cast<int>(kernel[0].size());

    if (k_h > in_h || k_w > in_w)
        throw std::invalid_argument("nn_cuda::correlate2d_valid kernel cannot be larger than input");

    const int out_h = in_h - k_h + 1;
    const int out_w = in_w - k_w + 1;

    std::vector<double> h_input = flatten_row_major(input);
    std::vector<double> h_kernel = flatten_row_major(kernel);
    std::vector<double> h_output(static_cast<std::size_t>(out_h) * static_cast<std::size_t>(out_w), 0.0);

    const std::size_t bytes_input = sizeof(double) * h_input.size();
    const std::size_t bytes_kernel = sizeof(double) * h_kernel.size();
    const std::size_t bytes_output = sizeof(double) * h_output.size();

    DeviceBuffer d_input(bytes_input);
    DeviceBuffer d_kernel(bytes_kernel);
    DeviceBuffer d_output(bytes_output);

    cuda_check(cudaMemcpy(d_input.ptr, h_input.data(), bytes_input, cudaMemcpyHostToDevice), "cudaMemcpy H2D conv input");
    cuda_check(cudaMemcpy(d_kernel.ptr, h_kernel.data(), bytes_kernel, cudaMemcpyHostToDevice), "cudaMemcpy H2D conv kernel");

    dim3 block(16, 16);
    dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);
    correlate2d_valid_kernel<<<grid, block>>>(d_input.ptr, d_kernel.ptr, d_output.ptr, in_h, in_w, k_h, k_w, out_w);
    cuda_check(cudaGetLastError(), "correlate2d_valid_kernel launch");
    cuda_check(cudaDeviceSynchronize(), "correlate2d_valid_kernel sync");

    cuda_check(cudaMemcpy(h_output.data(), d_output.ptr, bytes_output, cudaMemcpyDeviceToHost), "cudaMemcpy D2H conv output");

    return unflatten_row_major(h_output, static_cast<std::size_t>(out_h), static_cast<std::size_t>(out_w));
}

inline np::Matrix launch_convolve2d_full(const np::Matrix& input, const np::Matrix& kernel) {
    const int in_h = static_cast<int>(input.size());
    const int in_w = static_cast<int>(input[0].size());
    const int k_h = static_cast<int>(kernel.size());
    const int k_w = static_cast<int>(kernel[0].size());

    const int out_h = in_h + k_h - 1;
    const int out_w = in_w + k_w - 1;

    std::vector<double> h_input = flatten_row_major(input);
    std::vector<double> h_kernel = flatten_row_major(kernel);
    std::vector<double> h_output(static_cast<std::size_t>(out_h) * static_cast<std::size_t>(out_w), 0.0);

    const std::size_t bytes_input = sizeof(double) * h_input.size();
    const std::size_t bytes_kernel = sizeof(double) * h_kernel.size();
    const std::size_t bytes_output = sizeof(double) * h_output.size();

    DeviceBuffer d_input(bytes_input);
    DeviceBuffer d_kernel(bytes_kernel);
    DeviceBuffer d_output(bytes_output);

    cuda_check(cudaMemcpy(d_input.ptr, h_input.data(), bytes_input, cudaMemcpyHostToDevice), "cudaMemcpy H2D full-conv input");
    cuda_check(cudaMemcpy(d_kernel.ptr, h_kernel.data(), bytes_kernel, cudaMemcpyHostToDevice), "cudaMemcpy H2D full-conv kernel");

    dim3 block(16, 16);
    dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);
    convolve2d_full_kernel<<<grid, block>>>(d_input.ptr, d_kernel.ptr, d_output.ptr, in_h, in_w, k_h, k_w, out_h, out_w);
    cuda_check(cudaGetLastError(), "convolve2d_full_kernel launch");
    cuda_check(cudaDeviceSynchronize(), "convolve2d_full_kernel sync");

    cuda_check(cudaMemcpy(h_output.data(), d_output.ptr, bytes_output, cudaMemcpyDeviceToHost), "cudaMemcpy D2H full-conv output");

    return unflatten_row_major(h_output, static_cast<std::size_t>(out_h), static_cast<std::size_t>(out_w));
}
} // namespace

np::Matrix correlate2d_valid(const np::Matrix& input, const np::Matrix& kernel) {
    validate_rectangular(input, "nn_cuda::correlate2d_valid");
    validate_rectangular(kernel, "nn_cuda::correlate2d_valid");
    return launch_correlate2d_valid(input, kernel);
}

np::Matrix convolve2d_full(const np::Matrix& input, const np::Matrix& kernel) {
    validate_rectangular(input, "nn_cuda::convolve2d_full");
    validate_rectangular(kernel, "nn_cuda::convolve2d_full");
    return launch_convolve2d_full(input, kernel);
}

} // namespace nn_cuda
