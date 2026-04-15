#ifndef NN_LAYERS_DENSE_H
#define NN_LAYERS_DENSE_H

#include "nn/math/matrix.h"

namespace nn {
class Dense {
public:
    Dense(int input_size, int output_size);

    Matrix forward(const Matrix& input);
    Matrix backward(const Matrix& output_gradient, double learning_rate);

    void set_use_gpu(bool enabled);
    bool is_using_gpu() const;

    const Matrix& get_weights() const;
    const Matrix& get_bias() const;
    void set_weights(const Matrix& weights);
    void set_bias(const Matrix& bias);

private:
    Matrix matmul(const Matrix& left, const Matrix& right);

    Matrix weights_;
    Matrix bias_;
    Matrix input_cache_;
    bool gpu_enabled_;
};
} // namespace nn

#endif // NN_LAYERS_DENSE_H
