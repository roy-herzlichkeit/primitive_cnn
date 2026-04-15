#ifndef NN_LOSS_LOSSES_H
#define NN_LOSS_LOSSES_H

#include <cmath>
#include <stdexcept>
#include <string>

#include "nn/math/matrix.h"

namespace nn {
namespace loss {
inline double clamp_value(double value, double min_value, double max_value) {
    if (value < min_value)
        return min_value;
    if (value > max_value)
        return max_value;
    return value;
}

inline void validate_same_shape(const Matrix& left, const Matrix& right, const char* function_name) {
    if (left.size() != right.size())
        throw std::invalid_argument(std::string(function_name) + " expects matching matrix row counts");

    for (np::Index r = 0; r < left.size(); ++r) {
        if (left[r].size() != right[r].size())
            throw std::invalid_argument(std::string(function_name) + " expects matching matrix shapes");
    }
}

inline np::Index element_count(const Matrix& matrix) {
    np::Index total = 0;
    for (np::Index r = 0; r < matrix.size(); ++r)
        total += matrix[r].size();
    return total;
}

inline double mse(const Matrix& y_true, const Matrix& y_pred) {
    validate_same_shape(y_true, y_pred, "mse");
    const np::Index n = element_count(y_true);
    if (n == 0)
        throw std::invalid_argument("mse expects non-empty matrices");

    double sum = 0.0;
    for (np::Index r = 0; r < y_true.size(); ++r)
        for (np::Index c = 0; c < y_true[r].size(); ++c) {
            const double delta = y_true[r][c] - y_pred[r][c];
            sum += delta * delta;
        }

    return sum / static_cast<double>(n);
}

inline Matrix mse_prime(const Matrix& y_true, const Matrix& y_pred) {
    validate_same_shape(y_true, y_pred, "mse_prime");
    const np::Index n = element_count(y_true);
    if (n == 0)
        throw std::invalid_argument("mse_prime expects non-empty matrices");

    Matrix gradient = y_pred;
    const double scale = 2.0 / static_cast<double>(n);
    for (np::Index r = 0; r < gradient.size(); ++r)
        for (np::Index c = 0; c < gradient[r].size(); ++c)
            gradient[r][c] = scale * (y_pred[r][c] - y_true[r][c]);

    return gradient;
}

inline double binary_cross_entropy(const Matrix& y_true, const Matrix& y_pred) {
    validate_same_shape(y_true, y_pred, "binary_cross_entropy");
    const np::Index n = element_count(y_true);
    if (n == 0)
        throw std::invalid_argument("binary_cross_entropy expects non-empty matrices");

    constexpr double kEpsilon = 1e-12;
    double sum = 0.0;
    for (np::Index r = 0; r < y_true.size(); ++r) {
        for (np::Index c = 0; c < y_true[r].size(); ++c) {
            const double y = y_true[r][c];
            const double p = clamp_value(y_pred[r][c], kEpsilon, 1.0 - kEpsilon);
            sum += -y * std::log(p) - (1.0 - y) * std::log(1.0 - p);
        }
    }

    return sum / static_cast<double>(n);
}

inline Matrix binary_cross_entropy_prime(const Matrix& y_true, const Matrix& y_pred) {
    validate_same_shape(y_true, y_pred, "binary_cross_entropy_prime");
    const np::Index n = element_count(y_true);
    if (n == 0)
        throw std::invalid_argument("binary_cross_entropy_prime expects non-empty matrices");

    constexpr double kEpsilon = 1e-12;
    Matrix gradient = y_pred;
    const double scale = 1.0 / static_cast<double>(n);

    for (np::Index r = 0; r < gradient.size(); ++r) {
        for (np::Index c = 0; c < gradient[r].size(); ++c) {
            const double y = y_true[r][c];
            const double p = clamp_value(y_pred[r][c], kEpsilon, 1.0 - kEpsilon);
            gradient[r][c] = scale * (((1.0 - y) / (1.0 - p)) - (y / p));
        }
    }

    return gradient;
}
} // namespace loss
} // namespace nn

namespace loss {
using nn::loss::binary_cross_entropy;
using nn::loss::binary_cross_entropy_prime;
using nn::loss::mse;
using nn::loss::mse_prime;
} // namespace loss

#endif // NN_LOSS_LOSSES_H
