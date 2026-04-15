#ifndef NN_MATH_MATRIX_H
#define NN_MATH_MATRIX_H

#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace np {
using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;
using Index = std::size_t;

// Backward-compatible aliases used in older files.
using vd = Vector;
using ini = Index;

inline void validate_rectangular(const Matrix& matrix, const char* function_name) {
    if (matrix.empty())
        return;

    const Index cols = matrix[0].size();
    for (Index r = 0; r < matrix.size(); ++r) {
        if (matrix[r].size() != cols) {
            throw std::invalid_argument(std::string(function_name) + " expects a rectangular matrix");
        }
    }
}

inline Matrix randn(int rows, int cols) {
    if (rows <= 0 || cols <= 0)
        throw std::invalid_argument("randn expects positive dimensions");

    static thread_local std::mt19937 generator(std::random_device{}());
    static thread_local std::normal_distribution<double> distribution(0.0, 1.0);

    Matrix out(static_cast<Index>(rows), Vector(static_cast<Index>(cols), 0.0));
    for (Index r = 0; r < out.size(); ++r)
        for (Index c = 0; c < out[r].size(); ++c)
            out[r][c] = distribution(generator);

    return out;
}

inline Matrix transpose(const Matrix& matrix) {
    if (matrix.empty())
        return Matrix();

    validate_rectangular(matrix, "transpose");

    const Index rows = matrix.size();
    const Index cols = matrix[0].size();

    Matrix out(cols, Vector(rows, 0.0));
    for (Index r = 0; r < rows; ++r)
        for (Index c = 0; c < cols; ++c)
            out[c][r] = matrix[r][c];

    return out;
}

inline Matrix dot(const Matrix& left, const Matrix& right) {
    if (left.empty() || right.empty() || left[0].empty() || right[0].empty())
        throw std::invalid_argument("dot expects non-empty matrices");

    validate_rectangular(left, "dot");
    validate_rectangular(right, "dot");

    const Index left_rows = left.size();
    const Index left_cols = left[0].size();
    const Index right_rows = right.size();
    const Index right_cols = right[0].size();

    if (left_cols != right_rows)
        throw std::invalid_argument("dot dimension mismatch");

    Matrix out(left_rows, Vector(right_cols, 0.0));
    for (Index i = 0; i < left_rows; ++i) {
        for (Index k = 0; k < left_cols; ++k) {
            const double left_value = left[i][k];
            for (Index j = 0; j < right_cols; ++j)
                out[i][j] += left_value * right[k][j];
        }
    }

    return out;
}

inline Matrix add(const Matrix& left, const Matrix& right) {
    if (left.size() != right.size() || (!left.empty() && left[0].size() != right[0].size()))
        throw std::invalid_argument("add dimension mismatch");

    Matrix out = left;
    for (Index r = 0; r < out.size(); ++r) {
        if (out[r].size() != right[r].size())
            throw std::invalid_argument("add expects matching matrix shapes");

        for (Index c = 0; c < out[r].size(); ++c)
            out[r][c] += right[r][c];
    }

    return out;
}

inline Matrix subtract(const Matrix& left, const Matrix& right) {
    if (left.size() != right.size() || (!left.empty() && left[0].size() != right[0].size()))
        throw std::invalid_argument("subtract dimension mismatch");

    Matrix out = left;
    for (Index r = 0; r < out.size(); ++r) {
        if (out[r].size() != right[r].size())
            throw std::invalid_argument("subtract expects matching matrix shapes");

        for (Index c = 0; c < out[r].size(); ++c)
            out[r][c] -= right[r][c];
    }

    return out;
}

inline Matrix multiply_scalar(const Matrix& matrix, double scalar) {
    Matrix out = matrix;
    for (Index r = 0; r < out.size(); ++r)
        for (Index c = 0; c < out[r].size(); ++c)
            out[r][c] *= scalar;

    return out;
}

inline Matrix multiply(const Matrix& left, const Matrix& right) {
    if (left.size() != right.size() || (!left.empty() && left[0].size() != right[0].size()))
        throw std::invalid_argument("multiply dimension mismatch");

    Matrix out = left;
    for (Index r = 0; r < out.size(); ++r) {
        if (out[r].size() != right[r].size())
            throw std::invalid_argument("multiply expects matching matrix shapes");

        for (Index c = 0; c < out[r].size(); ++c)
            out[r][c] *= right[r][c];
    }

    return out;
}
} // namespace np

namespace nn {
using Matrix = np::Matrix;
using Tensor3D = std::vector<Matrix>;
using Tensor4D = std::vector<Tensor3D>;
} // namespace nn

#endif // NN_MATH_MATRIX_H
