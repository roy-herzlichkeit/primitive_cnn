#include "nn/layers/convolutional.h"

#include <stdexcept>

#include "nn/backend/cuda_conv2d.h"
#include "nn/backend/cuda_dot.h"
#include "nn/math/signal2d.h"

namespace nn {
Matrix Convolutional::zeros_matrix(int rows, int cols) {
    return Matrix(static_cast<std::size_t>(rows), np::Vector(static_cast<std::size_t>(cols), 0.0));
}

Tensor3D Convolutional::zeros_3d(int channels, int rows, int cols) {
    return Tensor3D(static_cast<std::size_t>(channels), zeros_matrix(rows, cols));
}

Tensor4D Convolutional::zeros_4d(int out_channels, int in_channels, int rows, int cols) {
    return Tensor4D(static_cast<std::size_t>(out_channels), zeros_3d(in_channels, rows, cols));
}

Tensor3D Convolutional::randn_3d(int channels, int rows, int cols) {
    Tensor3D tensor;
    tensor.reserve(static_cast<std::size_t>(channels));
    for (int channel = 0; channel < channels; ++channel)
        tensor.push_back(np::randn(rows, cols));
    return tensor;
}

Tensor4D Convolutional::randn_4d(int out_channels, int in_channels, int rows, int cols) {
    Tensor4D tensor;
    tensor.reserve(static_cast<std::size_t>(out_channels));
    for (int out_channel = 0; out_channel < out_channels; ++out_channel)
        tensor.push_back(randn_3d(in_channels, rows, cols));
    return tensor;
}

void Convolutional::add_inplace(Matrix& target, const Matrix& delta) {
    if (target.size() != delta.size() || (target.empty() ? false : target[0].size() != delta[0].size()))
        throw std::invalid_argument("Convolutional::add_inplace shape mismatch");

    for (std::size_t r = 0; r < target.size(); ++r)
        for (std::size_t c = 0; c < target[r].size(); ++c)
            target[r][c] += delta[r][c];
}

Convolutional::Convolutional(std::array<int, 3> input_shape, int kernel_size, int depth)
    : depth_(depth),
      input_shape_(input_shape),
      input_depth_(input_shape[0]),
      output_shape_({depth, input_shape[1] - kernel_size + 1, input_shape[2] - kernel_size + 1}),
      kernels_shape_({depth, input_shape[0], kernel_size, kernel_size}),
      kernels_(randn_4d(depth, input_shape[0], kernel_size, kernel_size)),
      biases_(randn_3d(depth, output_shape_[1], output_shape_[2])),
      gpu_enabled_(nn_cuda::is_available()) {
    if (input_shape_[0] <= 0 || input_shape_[1] <= 0 || input_shape_[2] <= 0)
        throw std::invalid_argument("Convolutional expects positive input dimensions");
    if (kernel_size <= 0)
        throw std::invalid_argument("Convolutional expects kernel_size > 0");
    if (depth_ <= 0)
        throw std::invalid_argument("Convolutional expects depth > 0");
    if (output_shape_[1] <= 0 || output_shape_[2] <= 0)
        throw std::invalid_argument("Convolutional kernel_size is too large for input shape");
}

Matrix Convolutional::correlate2d_valid(const Matrix& input, const Matrix& kernel) {
#if defined(NN_USE_CUDA_CONV2D)
    if (gpu_enabled_) {
        try {
            return nn_cuda::correlate2d_valid(input, kernel);
        } catch (...) {
            gpu_enabled_ = false;
        }
    }
#endif
    return signal::correlate2d_valid(input, kernel);
}

Matrix Convolutional::convolve2d_full(const Matrix& input, const Matrix& kernel) {
#if defined(NN_USE_CUDA_CONV2D)
    if (gpu_enabled_) {
        try {
            return nn_cuda::convolve2d_full(input, kernel);
        } catch (...) {
            gpu_enabled_ = false;
        }
    }
#endif
    return signal::convolve2d_full(input, kernel);
}

Tensor3D Convolutional::forward(const Tensor3D& input) {
    if (static_cast<int>(input.size()) != input_depth_)
        throw std::invalid_argument("Convolutional::forward input depth mismatch");

    input_cache_ = input;
    output_cache_ = biases_;

    for (int out_channel = 0; out_channel < depth_; ++out_channel) {
        for (int in_channel = 0; in_channel < input_depth_; ++in_channel) {
            add_inplace(
                output_cache_[static_cast<std::size_t>(out_channel)],
                correlate2d_valid(
                    input_cache_[static_cast<std::size_t>(in_channel)],
                    kernels_[static_cast<std::size_t>(out_channel)][static_cast<std::size_t>(in_channel)]));
        }
    }

    return output_cache_;
}

Tensor3D Convolutional::backward(const Tensor3D& output_gradient, double learning_rate) {
    if (static_cast<int>(output_gradient.size()) != depth_)
        throw std::invalid_argument("Convolutional::backward output gradient depth mismatch");

    Tensor4D kernels_gradient = zeros_4d(kernels_shape_[0], kernels_shape_[1], kernels_shape_[2], kernels_shape_[3]);
    Tensor3D input_gradient = zeros_3d(input_shape_[0], input_shape_[1], input_shape_[2]);

    for (int out_channel = 0; out_channel < depth_; ++out_channel) {
        for (int in_channel = 0; in_channel < input_depth_; ++in_channel) {
            kernels_gradient[static_cast<std::size_t>(out_channel)][static_cast<std::size_t>(in_channel)] = correlate2d_valid(
                input_cache_[static_cast<std::size_t>(in_channel)],
                output_gradient[static_cast<std::size_t>(out_channel)]);

            add_inplace(
                input_gradient[static_cast<std::size_t>(in_channel)],
                convolve2d_full(
                    output_gradient[static_cast<std::size_t>(out_channel)],
                    kernels_[static_cast<std::size_t>(out_channel)][static_cast<std::size_t>(in_channel)]));
        }
    }

    for (int out_channel = 0; out_channel < depth_; ++out_channel) {
        for (int in_channel = 0; in_channel < input_depth_; ++in_channel) {
            const Matrix scaled = np::multiply_scalar(
                kernels_gradient[static_cast<std::size_t>(out_channel)][static_cast<std::size_t>(in_channel)],
                learning_rate);
            kernels_[static_cast<std::size_t>(out_channel)][static_cast<std::size_t>(in_channel)] = np::subtract(
                kernels_[static_cast<std::size_t>(out_channel)][static_cast<std::size_t>(in_channel)],
                scaled);
        }

        const Matrix scaled_bias = np::multiply_scalar(output_gradient[static_cast<std::size_t>(out_channel)], learning_rate);
        biases_[static_cast<std::size_t>(out_channel)] = np::subtract(biases_[static_cast<std::size_t>(out_channel)], scaled_bias);
    }

    return input_gradient;
}

void Convolutional::set_use_gpu(bool enabled) {
    gpu_enabled_ = enabled && nn_cuda::is_available();
}

bool Convolutional::is_using_gpu() const {
    return gpu_enabled_;
}

const Tensor4D& Convolutional::get_kernels() const {
    return kernels_;
}

const Tensor3D& Convolutional::get_biases() const {
    return biases_;
}

void Convolutional::set_kernels(const Tensor4D& kernels) {
    if (kernels.size() != kernels_.size())
        throw std::invalid_argument("Convolutional::set_kernels depth mismatch");

    for (std::size_t out_channel = 0; out_channel < kernels.size(); ++out_channel) {
        if (kernels[out_channel].size() != kernels_[out_channel].size())
            throw std::invalid_argument("Convolutional::set_kernels input-depth mismatch");

        for (std::size_t in_channel = 0; in_channel < kernels[out_channel].size(); ++in_channel) {
            if (kernels[out_channel][in_channel].size() != kernels_[out_channel][in_channel].size() ||
                (kernels[out_channel][in_channel].empty()
                     ? false
                     : kernels[out_channel][in_channel][0].size() != kernels_[out_channel][in_channel][0].size())) {
                throw std::invalid_argument("Convolutional::set_kernels shape mismatch");
            }
        }
    }

    kernels_ = kernels;
}

void Convolutional::set_biases(const Tensor3D& biases) {
    if (biases.size() != biases_.size())
        throw std::invalid_argument("Convolutional::set_biases depth mismatch");

    for (std::size_t out_channel = 0; out_channel < biases.size(); ++out_channel) {
        if (biases[out_channel].size() != biases_[out_channel].size() ||
            (biases[out_channel].empty() ? false : biases[out_channel][0].size() != biases_[out_channel][0].size())) {
            throw std::invalid_argument("Convolutional::set_biases shape mismatch");
        }
    }

    biases_ = biases;
}
} // namespace nn
