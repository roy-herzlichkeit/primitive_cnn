#include "nn/layers/reshape.h"

#include <stdexcept>
#include <vector>

namespace nn {
std::size_t Reshape::element_count(const std::array<int, 3>& shape) {
    return static_cast<std::size_t>(shape[0]) * static_cast<std::size_t>(shape[1]) * static_cast<std::size_t>(shape[2]);
}

std::size_t Reshape::element_count(const std::array<int, 2>& shape) {
    return static_cast<std::size_t>(shape[0]) * static_cast<std::size_t>(shape[1]);
}

Reshape::Reshape(std::array<int, 3> input_shape, std::array<int, 2> output_shape)
    : input_shape_(input_shape), output_shape_(output_shape) {
    if (element_count(input_shape_) != element_count(output_shape_))
        throw std::invalid_argument("Reshape expects equal input/output element count");
}

Matrix Reshape::forward(const Tensor3D& input) {
    if (static_cast<int>(input.size()) != input_shape_[0])
        throw std::invalid_argument("Reshape::forward channel count mismatch");

    std::vector<double> flat;
    flat.reserve(element_count(input_shape_));

    for (int channel = 0; channel < input_shape_[0]; ++channel) {
        if (static_cast<int>(input[static_cast<std::size_t>(channel)].size()) != input_shape_[1])
            throw std::invalid_argument("Reshape::forward row count mismatch");

        for (int row = 0; row < input_shape_[1]; ++row) {
            if (static_cast<int>(input[static_cast<std::size_t>(channel)][static_cast<std::size_t>(row)].size()) != input_shape_[2])
                throw std::invalid_argument("Reshape::forward column count mismatch");

            for (int col = 0; col < input_shape_[2]; ++col) {
                flat.push_back(input[static_cast<std::size_t>(channel)][static_cast<std::size_t>(row)][static_cast<std::size_t>(col)]);
            }
        }
    }

    Matrix output(static_cast<std::size_t>(output_shape_[0]), np::Vector(static_cast<std::size_t>(output_shape_[1]), 0.0));
    std::size_t index = 0;
    for (int row = 0; row < output_shape_[0]; ++row)
        for (int col = 0; col < output_shape_[1]; ++col)
            output[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] = flat[index++];

    return output;
}

Tensor3D Reshape::backward(const Matrix& output_gradient, double /*learning_rate*/) {
    if (static_cast<int>(output_gradient.size()) != output_shape_[0])
        throw std::invalid_argument("Reshape::backward row count mismatch");

    std::vector<double> flat;
    flat.reserve(element_count(output_shape_));

    for (int row = 0; row < output_shape_[0]; ++row) {
        if (static_cast<int>(output_gradient[static_cast<std::size_t>(row)].size()) != output_shape_[1])
            throw std::invalid_argument("Reshape::backward column count mismatch");

        for (int col = 0; col < output_shape_[1]; ++col)
            flat.push_back(output_gradient[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)]);
    }

    Tensor3D input_gradient(
        static_cast<std::size_t>(input_shape_[0]),
        Matrix(static_cast<std::size_t>(input_shape_[1]), np::Vector(static_cast<std::size_t>(input_shape_[2]), 0.0)));

    std::size_t index = 0;
    for (int channel = 0; channel < input_shape_[0]; ++channel)
        for (int row = 0; row < input_shape_[1]; ++row)
            for (int col = 0; col < input_shape_[2]; ++col)
                input_gradient[static_cast<std::size_t>(channel)][static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] = flat[index++];

    return input_gradient;
}
} // namespace nn
