#ifndef NN_BACKEND_CUDA_DOT_H
#define NN_BACKEND_CUDA_DOT_H

#include "nn/math/matrix.h"

namespace nn_cuda {
#if defined(NN_USE_CUDA_DOT)
bool is_available();
np::Matrix dot(const np::Matrix& a, const np::Matrix& b);
#else
inline bool is_available() {
    return false;
}

inline np::Matrix dot(const np::Matrix& a, const np::Matrix& b) {
    return np::dot(a, b);
}
#endif
} // namespace nn_cuda

#endif // NN_BACKEND_CUDA_DOT_H
