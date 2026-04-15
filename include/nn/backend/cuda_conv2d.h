#ifndef NN_BACKEND_CUDA_CONV2D_H
#define NN_BACKEND_CUDA_CONV2D_H

#include "nn/math/matrix.h"
#include "nn/math/signal2d.h"

namespace nn_cuda {
#if defined(NN_USE_CUDA_CONV2D)
np::Matrix correlate2d_valid(const np::Matrix& input, const np::Matrix& kernel);
np::Matrix convolve2d_full(const np::Matrix& input, const np::Matrix& kernel);
#else
inline np::Matrix correlate2d_valid(const np::Matrix& input, const np::Matrix& kernel) {
    return nn::signal::correlate2d_valid(input, kernel);
}

inline np::Matrix convolve2d_full(const np::Matrix& input, const np::Matrix& kernel) {
    return nn::signal::convolve2d_full(input, kernel);
}
#endif
} // namespace nn_cuda

#endif // NN_BACKEND_CUDA_CONV2D_H
