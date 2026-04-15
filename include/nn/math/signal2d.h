#ifndef NN_MATH_SIGNAL2D_H
#define NN_MATH_SIGNAL2D_H

#include <stdexcept>
#include <string>

#include "nn/math/matrix.h"

namespace nn {
namespace signal {
inline Matrix correlate2d_valid(const Matrix& input, const Matrix& kernel) {
    np::validate_rectangular(input, "correlate2d_valid");
    np::validate_rectangular(kernel, "correlate2d_valid");

    if (input.empty() || input[0].empty() || kernel.empty() || kernel[0].empty())
        throw std::invalid_argument("correlate2d_valid expects non-empty matrices");

    const np::Index in_h = input.size();
    const np::Index in_w = input[0].size();
    const np::Index k_h = kernel.size();
    const np::Index k_w = kernel[0].size();

    if (k_h > in_h || k_w > in_w)
        throw std::invalid_argument("correlate2d_valid kernel cannot be larger than input");

    const np::Index out_h = in_h - k_h + 1;
    const np::Index out_w = in_w - k_w + 1;
    Matrix out(out_h, np::Vector(out_w, 0.0));

    for (np::Index r = 0; r < out_h; ++r) {
        for (np::Index c = 0; c < out_w; ++c) {
            double sum = 0.0;
            for (np::Index kr = 0; kr < k_h; ++kr) {
                for (np::Index kc = 0; kc < k_w; ++kc) {
                    sum += input[r + kr][c + kc] * kernel[kr][kc];
                }
            }
            out[r][c] = sum;
        }
    }

    return out;
}

inline Matrix convolve2d_full(const Matrix& input, const Matrix& kernel) {
    np::validate_rectangular(input, "convolve2d_full");
    np::validate_rectangular(kernel, "convolve2d_full");

    if (input.empty() || input[0].empty() || kernel.empty() || kernel[0].empty())
        throw std::invalid_argument("convolve2d_full expects non-empty matrices");

    const np::Index in_h = input.size();
    const np::Index in_w = input[0].size();
    const np::Index k_h = kernel.size();
    const np::Index k_w = kernel[0].size();

    const np::Index out_h = in_h + k_h - 1;
    const np::Index out_w = in_w + k_w - 1;
    Matrix out(out_h, np::Vector(out_w, 0.0));

    for (np::Index r = 0; r < out_h; ++r) {
        for (np::Index c = 0; c < out_w; ++c) {
            double sum = 0.0;
            for (np::Index kr = 0; kr < k_h; ++kr) {
                for (np::Index kc = 0; kc < k_w; ++kc) {
                    const int in_r = static_cast<int>(r) - static_cast<int>(kr);
                    const int in_c = static_cast<int>(c) - static_cast<int>(kc);
                    if (in_r < 0 || in_c < 0 || in_r >= static_cast<int>(in_h) || in_c >= static_cast<int>(in_w))
                        continue;

                    const np::Index flipped_kr = k_h - 1 - kr;
                    const np::Index flipped_kc = k_w - 1 - kc;
                    sum += input[static_cast<np::Index>(in_r)][static_cast<np::Index>(in_c)] * kernel[flipped_kr][flipped_kc];
                }
            }
            out[r][c] = sum;
        }
    }

    return out;
}
} // namespace signal
} // namespace nn

namespace scipy_signal {
using Matrix = nn::Matrix;
using ini = np::Index;

inline Matrix correlate2d(const Matrix& input, const Matrix& kernel, const std::string& mode = "valid") {
    if (mode != "valid")
        throw std::invalid_argument("correlate2d currently supports only mode='valid'");
    return nn::signal::correlate2d_valid(input, kernel);
}

inline Matrix convolve2d(const Matrix& input, const Matrix& kernel, const std::string& mode = "full") {
    if (mode != "full")
        throw std::invalid_argument("convolve2d currently supports only mode='full'");
    return nn::signal::convolve2d_full(input, kernel);
}
} // namespace scipy_signal

#endif // NN_MATH_SIGNAL2D_H
