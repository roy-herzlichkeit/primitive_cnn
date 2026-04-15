#include <cmath>
#include <iostream>
#include <stdexcept>

#include "nn/backend/cuda_dot.h"
#include "nn/loss/losses.h"
#include "nn/math/matrix.h"
#include "nn/math/signal2d.h"

static bool almost_equal(double a, double b, double eps = 1e-9) {
    return std::fabs(a - b) < eps;
}

int main() {
    try {
        {
            np::Matrix a = {{1.0, 2.0}, {3.0, 4.0}};
            np::Matrix b = {{2.0}, {1.0}};
            np::Matrix out = np::dot(a, b);
            if (!almost_equal(out[0][0], 4.0) || !almost_equal(out[1][0], 10.0)) {
                throw std::runtime_error("np::dot test failed");
            }
        }

        {
            np::Matrix y_true = {{1.0}, {0.0}};
            np::Matrix y_pred = {{0.9}, {0.2}};
            double l = loss::binary_cross_entropy(y_true, y_pred);
            if (!(l > 0.0 && l < 1.0)) {
                throw std::runtime_error("binary_cross_entropy range test failed");
            }
        }

        {
            np::Matrix input = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
            np::Matrix kernel = {{1.0, 0.0}, {0.0, -1.0}};
            np::Matrix corr = scipy_signal::correlate2d(input, kernel, "valid");
            if (corr.size() != 1 || corr[0].size() != 2) {
                throw std::runtime_error("scipy_signal::correlate2d shape test failed");
            }
        }

        {
            np::Matrix a = {{1.5, -2.0, 0.5}, {3.0, 4.0, -1.0}};
            np::Matrix b = {{2.0, 1.0}, {-3.0, 0.5}, {4.0, -2.0}};
            np::Matrix cpu = np::dot(a, b);

            if (nn_cuda::is_available()) {
                np::Matrix gpu = nn_cuda::dot(a, b);
                if (!almost_equal(cpu[0][0], gpu[0][0], 1e-9) ||
                    !almost_equal(cpu[0][1], gpu[0][1], 1e-9) ||
                    !almost_equal(cpu[1][0], gpu[1][0], 1e-9) ||
                    !almost_equal(cpu[1][1], gpu[1][1], 1e-9)) {
                    throw std::runtime_error("nn_cuda::dot correctness test failed");
                }
            }
        }

        std::cout << "test_core passed\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "test_core failed: " << ex.what() << "\n";
        return 1;
    }
}
