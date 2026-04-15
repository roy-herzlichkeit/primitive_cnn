#ifndef NN_LAYERS_CONVOLUTIONAL_H
#define NN_LAYERS_CONVOLUTIONAL_H

#include <array>

#include "nn/math/matrix.h"

namespace nn {
class Convolutional {
public:
    Convolutional(std::array<int, 3> input_shape, int kernel_size, int depth);

    Tensor3D forward(const Tensor3D& input);
    Tensor3D backward(const Tensor3D& output_gradient, double learning_rate);

    void set_use_gpu(bool enabled);
    bool is_using_gpu() const;

    const Tensor4D& get_kernels() const;
    const Tensor3D& get_biases() const;
    void set_kernels(const Tensor4D& kernels);
    void set_biases(const Tensor3D& biases);

private:
    static Matrix zeros_matrix(int rows, int cols);
    static Tensor3D zeros_3d(int channels, int rows, int cols);
    static Tensor4D zeros_4d(int out_channels, int in_channels, int rows, int cols);
    static Tensor3D randn_3d(int channels, int rows, int cols);
    static Tensor4D randn_4d(int out_channels, int in_channels, int rows, int cols);
    static void add_inplace(Matrix& target, const Matrix& delta);

    Matrix correlate2d_valid(const Matrix& input, const Matrix& kernel);
    Matrix convolve2d_full(const Matrix& input, const Matrix& kernel);

    int depth_;
    std::array<int, 3> input_shape_;
    int input_depth_;
    std::array<int, 3> output_shape_;
    std::array<int, 4> kernels_shape_;

    Tensor4D kernels_;
    Tensor3D biases_;
    Tensor3D input_cache_;
    Tensor3D output_cache_;
    bool gpu_enabled_;
};
} // namespace nn

#endif // NN_LAYERS_CONVOLUTIONAL_H
