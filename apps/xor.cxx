#include <array>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "nn/layers/activation.h"
#include "nn/layers/dense.h"
#include "nn/loss/losses.h"

using Matrix = np::Matrix;
using Sample = Matrix;
using Dataset = std::vector<Sample>;
using nn::Dense;
using nn::Tanh;

struct LayerOps {
    std::function<Matrix(const Matrix&)> forward;
    std::function<Matrix(const Matrix&, double)> backward;
};

using Network = std::vector<LayerOps>;

Matrix predict(Network& network, const Matrix& input) {
    Matrix output = input;
    for (auto& layer : network) {
        output = layer.forward(output);
    }
    return output;
}

void train(
    Network& network,
    const Dataset& x_train,
    const Dataset& y_train,
    int epochs = 1000,
    double learning_rate = 0.01,
    bool verbose = true) {
    if (x_train.size() != y_train.size()) {
        throw std::invalid_argument("x_train and y_train must have the same length");
    }
    if (x_train.empty()) {
        throw std::invalid_argument("x_train cannot be empty");
    }

    for (int e = 0; e < epochs; ++e) {
        double error = 0.0;

        for (std::size_t i = 0; i < x_train.size(); ++i) {
            const Matrix& x = x_train[i];
            const Matrix& y = y_train[i];

            // forward
            Matrix output = predict(network, x);

            // error
            error += loss::mse(y, output);

            // backward
            Matrix grad = loss::mse_prime(y, output);
            for (auto it = network.rbegin(); it != network.rend(); ++it) {
                grad = it->backward(grad, learning_rate);
            }
        }

        error /= static_cast<double>(x_train.size());
        if (verbose) {
            std::cout << (e + 1) << "/" << epochs << ", error=" << error << "\n";
        }
    }
}

int main(int argc, char** argv) {
    int epochs = 10000;
    bool verbose = true;
    if (argc > 1) {
        epochs = std::atoi(argv[1]);
        if (epochs <= 0)
            epochs = 10000;
    }
    if (argc > 2) {
        verbose = (std::atoi(argv[2]) != 0);
    }

    const Dataset X = {
        {{0.0}, {0.0}},
        {{0.0}, {1.0}},
        {{1.0}, {0.0}},
        {{1.0}, {1.0}},
    };

    const Dataset Y = {
        {{0.0}},
        {{1.0}},
        {{1.0}},
        {{0.0}},
    };

    Dense dense1(2, 3);
    Tanh tanh1;
    Dense dense2(3, 1);
    Tanh tanh2;

    Network network = {
        LayerOps{
            [&](const Matrix& m) { return dense1.forward(m); },
            [&](const Matrix& g, double lr) { return dense1.backward(g, lr); }},
        LayerOps{
            [&](const Matrix& m) { return tanh1.forward(m); },
            [&](const Matrix& g, double lr) { return tanh1.backward(g, lr); }},
        LayerOps{
            [&](const Matrix& m) { return dense2.forward(m); },
            [&](const Matrix& g, double lr) { return dense2.backward(g, lr); }},
        LayerOps{
            [&](const Matrix& m) { return tanh2.forward(m); },
            [&](const Matrix& g, double lr) { return tanh2.backward(g, lr); }},
    };

    train(network, X, Y, epochs, 0.1, verbose);

    std::vector<std::array<double, 3>> points;
    points.reserve(20 * 20);

    for (int xi = 0; xi < 20; ++xi) {
        const double x = static_cast<double>(xi) / 19.0;
        for (int yi = 0; yi < 20; ++yi) {
            const double y = static_cast<double>(yi) / 19.0;
            const Matrix z = predict(network, {{x}, {y}});
            points.push_back({x, y, z[0][0]});
        }
    }

    std::ofstream out("data/xor/xor_points.csv");
    out << "x,y,z\n";
    for (const auto& p : points) {
        out << p[0] << "," << p[1] << "," << p[2] << "\n";
    }
    out.close();

    std::cout << "Saved decision boundary points to xor_points.csv (" << points.size() << " rows).\n";

    for (std::size_t i = 0; i < X.size(); ++i) {
        const Matrix pred = predict(network, X[i]);
        std::cout << "Input(" << X[i][0][0] << ", " << X[i][1][0] << ") -> " << pred[0][0]
                  << " (target=" << Y[i][0][0] << ")\n";
    }

    return 0;
}
