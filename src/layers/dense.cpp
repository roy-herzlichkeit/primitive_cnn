#include "nn/layers/dense.h"

#include <stdexcept>

#include "nn/backend/cuda_dot.h"

namespace nn {
Dense::Dense(int input_size, int output_size)
    : weights_(np::randn(output_size, input_size)),
      bias_(np::randn(output_size, 1)),
      gpu_enabled_(nn_cuda::is_available()) {}

Matrix Dense::matmul(const Matrix& left, const Matrix& right) {
#if defined(NN_USE_CUDA_DOT)
    if (gpu_enabled_) {
        try {
            return nn_cuda::dot(left, right);
        } catch (...) {
            gpu_enabled_ = false;
        }
    }
#endif
    return np::dot(left, right);
}

Matrix Dense::forward(const Matrix& input) {
    input_cache_ = input;
    return np::add(matmul(weights_, input_cache_), bias_);
}

Matrix Dense::backward(const Matrix& output_gradient, double learning_rate) {
    const Matrix weights_gradient = matmul(output_gradient, np::transpose(input_cache_));
    const Matrix input_gradient = matmul(np::transpose(weights_), output_gradient);

    weights_ = np::subtract(weights_, np::multiply_scalar(weights_gradient, learning_rate));
    bias_ = np::subtract(bias_, np::multiply_scalar(output_gradient, learning_rate));

    return input_gradient;
}

void Dense::set_use_gpu(bool enabled) {
    gpu_enabled_ = enabled && nn_cuda::is_available();
}

bool Dense::is_using_gpu() const {
    return gpu_enabled_;
}

const Matrix& Dense::get_weights() const {
    return weights_;
}

const Matrix& Dense::get_bias() const {
    return bias_;
}

void Dense::set_weights(const Matrix& weights) {
    if (weights.size() != weights_.size() || (!weights.empty() && weights[0].size() != weights_[0].size()))
        throw std::invalid_argument("Dense::set_weights shape mismatch");
    weights_ = weights;
}

void Dense::set_bias(const Matrix& bias) {
    if (bias.size() != bias_.size() || (!bias.empty() && bias[0].size() != bias_[0].size()))
        throw std::invalid_argument("Dense::set_bias shape mismatch");
    bias_ = bias;
}
} // namespace nn
