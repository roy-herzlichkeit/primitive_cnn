#ifndef NN_LAYERS_RESHAPE_H
#define NN_LAYERS_RESHAPE_H

#include <array>

#include "nn/math/matrix.h"

namespace nn {
class Reshape {
public:
    Reshape(std::array<int, 3> input_shape, std::array<int, 2> output_shape);

    Matrix forward(const Tensor3D& input);
    Tensor3D backward(const Matrix& output_gradient, double learning_rate);

private:
    static std::size_t element_count(const std::array<int, 3>& shape);
    static std::size_t element_count(const std::array<int, 2>& shape);

    std::array<int, 3> input_shape_;
    std::array<int, 2> output_shape_;
};
} // namespace nn

#endif // NN_LAYERS_RESHAPE_H
