#ifndef NN_LAYERS_ACTIVATION_H
#define NN_LAYERS_ACTIVATION_H

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>

#include "nn/math/matrix.h"

namespace nn {
class Activation {
public:
    using ActivationFn = std::function<Matrix(const Matrix&)>;

    Activation(ActivationFn activation, ActivationFn activation_prime)
        : activation_(activation), activation_prime_(activation_prime) {}

    Matrix forward(const Matrix& input) {
        input_cache_ = input;
        return activation_(input_cache_);
    }

    Matrix backward(const Matrix& output_gradient, double /*learning_rate*/) {
        return np::multiply(output_gradient, activation_prime_(input_cache_));
    }

private:
    ActivationFn activation_;
    ActivationFn activation_prime_;
    Matrix input_cache_;
};

class Tanh : public Activation {
public:
    Tanh()
        : Activation(
              [](const Matrix& x) {
                  Matrix out = x;
                  for (np::Index r = 0; r < out.size(); ++r)
                      for (np::Index c = 0; c < out[r].size(); ++c)
                          out[r][c] = std::tanh(out[r][c]);
                  return out;
              },
              [](const Matrix& x) {
                  Matrix out = x;
                  for (np::Index r = 0; r < out.size(); ++r) {
                      for (np::Index c = 0; c < out[r].size(); ++c) {
                          const double t = std::tanh(out[r][c]);
                          out[r][c] = 1.0 - t * t;
                      }
                  }
                  return out;
              }) {}
};

class Sigmoid : public Activation {
public:
    Sigmoid()
        : Activation(
              [](const Matrix& x) {
                  Matrix out = x;
                  for (np::Index r = 0; r < out.size(); ++r)
                      for (np::Index c = 0; c < out[r].size(); ++c)
                          out[r][c] = 1.0 / (1.0 + std::exp(-out[r][c]));
                  return out;
              },
              [](const Matrix& x) {
                  Matrix sigmoid_values = x;
                  for (np::Index r = 0; r < sigmoid_values.size(); ++r)
                      for (np::Index c = 0; c < sigmoid_values[r].size(); ++c)
                          sigmoid_values[r][c] = 1.0 / (1.0 + std::exp(-sigmoid_values[r][c]));

                  Matrix out = sigmoid_values;
                  for (np::Index r = 0; r < out.size(); ++r)
                      for (np::Index c = 0; c < out[r].size(); ++c)
                          out[r][c] = sigmoid_values[r][c] * (1.0 - sigmoid_values[r][c]);
                  return out;
              }) {}
};

class Softmax {
public:
    Matrix forward(const Matrix& input) {
        if (input.empty() || input[0].empty())
            throw std::invalid_argument("Softmax expects a non-empty matrix");
        if (input[0].size() != 1)
            throw std::invalid_argument("Softmax expects a column vector (n x 1)");

        double max_value = input[0][0];
        for (np::Index r = 0; r < input.size(); ++r)
            max_value = std::max(max_value, input[r][0]);

        Matrix exponents = input;
        double sum_exp = 0.0;
        for (np::Index r = 0; r < exponents.size(); ++r) {
            exponents[r][0] = std::exp(exponents[r][0] - max_value);
            sum_exp += exponents[r][0];
        }

        output_cache_ = exponents;
        for (np::Index r = 0; r < output_cache_.size(); ++r)
            output_cache_[r][0] /= sum_exp;

        return output_cache_;
    }

    Matrix backward(const Matrix& output_gradient, double /*learning_rate*/) {
        if (output_cache_.empty() || output_cache_[0].size() != 1)
            throw std::runtime_error("Softmax backward requires a previous valid forward pass");

        const np::Index n = output_cache_.size();
        Matrix jacobian(n, np::Vector(n, 0.0));
        for (np::Index i = 0; i < n; ++i) {
            for (np::Index j = 0; j < n; ++j) {
                const double delta = (i == j) ? 1.0 : 0.0;
                jacobian[i][j] = output_cache_[i][0] * (delta - output_cache_[j][0]);
            }
        }

        return np::dot(jacobian, output_gradient);
    }

private:
    Matrix output_cache_;
};
} // namespace nn

#endif // NN_LAYERS_ACTIVATION_H
