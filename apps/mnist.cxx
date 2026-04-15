#include <algorithm>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "nn/layers/activation.h"
#include "nn/layers/dense.h"
#include "nn/loss/losses.h"

using Matrix = np::Matrix;
using Sample = Matrix;
using Dataset = std::vector<Sample>;
using nn::Dense;
using nn::Tanh;

struct MnistDenseData {
    Dataset x;
    Dataset y;
};

struct LayerOps {
    std::function<Matrix(const Matrix&)> forward;
    std::function<Matrix(const Matrix&, double)> backward;
};

using Network = std::vector<LayerOps>;

static std::uint32_t read_be_u32(std::istream& in) {
    unsigned char b[4] = {0, 0, 0, 0};
    in.read(reinterpret_cast<char*>(b), 4);
    if (!in)
        throw std::runtime_error("Failed to read 4 bytes from IDX stream");
    return (static_cast<std::uint32_t>(b[0]) << 24) |
           (static_cast<std::uint32_t>(b[1]) << 16) |
           (static_cast<std::uint32_t>(b[2]) << 8) |
           static_cast<std::uint32_t>(b[3]);
}

static void load_idx_dense(const std::string& images_path, const std::string& labels_path, Dataset& images, std::vector<int>& labels) {
    std::ifstream img(images_path.c_str(), std::ios::binary);
    std::ifstream lbl(labels_path.c_str(), std::ios::binary);
    if (!img || !lbl)
        throw std::runtime_error("Could not open IDX files: " + images_path + " and/or " + labels_path);

    const std::uint32_t img_magic = read_be_u32(img);
    const std::uint32_t img_count = read_be_u32(img);
    const std::uint32_t rows = read_be_u32(img);
    const std::uint32_t cols = read_be_u32(img);

    const std::uint32_t lbl_magic = read_be_u32(lbl);
    const std::uint32_t lbl_count = read_be_u32(lbl);

    if (img_magic != 2051 || lbl_magic != 2049)
        throw std::runtime_error("Invalid IDX magic numbers");
    if (img_count != lbl_count)
        throw std::runtime_error("Image and label counts differ in IDX files");

    images.clear();
    labels.clear();
    images.reserve(img_count);
    labels.reserve(lbl_count);

    for (std::uint32_t i = 0; i < img_count; ++i) {
        Matrix m(rows, np::vd(cols, 0.0));
        for (std::uint32_t r = 0; r < rows; ++r) {
            for (std::uint32_t c = 0; c < cols; ++c) {
                unsigned char px = 0;
                img.read(reinterpret_cast<char*>(&px), 1);
                if (!img)
                    throw std::runtime_error("Unexpected EOF while reading image bytes");
                m[r][c] = static_cast<double>(px);
            }
        }
        images.push_back(m);

        unsigned char y = 0;
        lbl.read(reinterpret_cast<char*>(&y), 1);
        if (!lbl)
            throw std::runtime_error("Unexpected EOF while reading label bytes");
        labels.push_back(static_cast<int>(y));
    }
}

static void generate_synthetic_dense(Dataset& images, std::vector<int>& labels, std::size_t count) {
    images.clear();
    labels.clear();
    images.reserve(count);
    labels.reserve(count);

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 8.0);

    for (std::size_t i = 0; i < count; ++i) {
        const int label = static_cast<int>(i % 10);
        Matrix img(28, np::vd(28, 0.0));

        // Encode class identity in one row band so the network has a learnable pattern.
        const int band_row = (label * 2) % 28;
        for (int c = 0; c < 28; ++c) {
            img[band_row][c] = 180.0 + noise(rng);
        }

        // Add low-level background noise.
        for (int r = 0; r < 28; ++r)
            for (int c = 0; c < 28; ++c)
                img[r][c] = std::max(0.0, std::min(255.0, img[r][c] + noise(rng)));

        images.push_back(img);
        labels.push_back(label);
    }
}

MnistDenseData preprocess_dense(const Dataset& raw_images, const std::vector<int>& raw_labels, std::size_t limit) {
    if (raw_images.size() != raw_labels.size())
        throw std::invalid_argument("raw_images and raw_labels must have same size");

    const std::size_t n = std::min(limit, raw_images.size());
    MnistDenseData out;
    out.x.reserve(n);
    out.y.reserve(n);

    for (std::size_t i = 0; i < n; ++i) {
        const Matrix& img = raw_images[i];
        if (img.size() != 28 || img[0].size() != 28)
            throw std::invalid_argument("Expected each image to be 28x28");

        Matrix x(28 * 28, np::vd(1, 0.0));
        std::size_t k = 0;
        for (std::size_t r = 0; r < 28; ++r) {
            if (img[r].size() != 28)
                throw std::invalid_argument("Expected rectangular 28x28 images");
            for (std::size_t c = 0; c < 28; ++c) {
                const double v = img[r][c];
                x[k++][0] = (v > 1.0) ? (v / 255.0) : v;
            }
        }

        Matrix y(10, np::vd(1, 0.0));
        const int label = raw_labels[i];
        if (label < 0 || label > 9)
            throw std::invalid_argument("Label out of range [0,9]");
        y[static_cast<std::size_t>(label)][0] = 1.0;

        out.x.push_back(x);
        out.y.push_back(y);
    }

    return out;
}

int argmax_column_vector(const Matrix& v) {
    if (v.empty() || v[0].empty())
        throw std::invalid_argument("argmax_column_vector expects non-empty matrix");

    std::size_t best_i = 0;
    double best_v = v[0][0];
    for (std::size_t i = 1; i < v.size(); ++i) {
        if (v[i].empty())
            throw std::invalid_argument("argmax_column_vector expects rectangular column vector");
        if (v[i][0] > best_v) {
            best_v = v[i][0];
            best_i = i;
        }
    }
    return static_cast<int>(best_i);
}

Matrix predict_dense(Network& network, const Matrix& input) {
    Matrix output = input;
    for (std::size_t i = 0; i < network.size(); ++i)
        output = network[i].forward(output);
    return output;
}

void train_dense(Network& network, const Dataset& x_train, const Dataset& y_train, int epochs, double learning_rate) {
    if (x_train.size() != y_train.size())
        throw std::invalid_argument("x_train and y_train must have same size");
    if (x_train.empty())
        throw std::invalid_argument("x_train cannot be empty");

    for (int e = 0; e < epochs; ++e) {
        double error = 0.0;
        for (std::size_t i = 0; i < x_train.size(); ++i) {
            Matrix output = predict_dense(network, x_train[i]);
            error += loss::mse(y_train[i], output);

            Matrix grad = loss::mse_prime(y_train[i], output);
            for (std::size_t rev = network.size(); rev-- > 0;)
                grad = network[rev].backward(grad, learning_rate);
        }

        error /= static_cast<double>(x_train.size());
        std::cout << (e + 1) << "/" << epochs << ", error=" << error << "\n";
    }
}

int main() {
    Dataset raw_train_images;
    std::vector<int> raw_train_labels;
    Dataset raw_test_images;
    std::vector<int> raw_test_labels;

    bool used_real_mnist = true;
    try {
        try {
            load_idx_dense("data/mnist/train-images-idx3-ubyte", "data/mnist/train-labels-idx1-ubyte", raw_train_images, raw_train_labels);
            load_idx_dense("data/mnist/t10k-images-idx3-ubyte", "data/mnist/t10k-labels-idx1-ubyte", raw_test_images, raw_test_labels);
        } catch (...) {
            // Also support common file naming with ".idx*" instead of "-idx*".
            load_idx_dense("data/mnist/train-images.idx3-ubyte", "data/mnist/train-labels.idx1-ubyte", raw_train_images, raw_train_labels);
            load_idx_dense("data/mnist/t10k-images.idx3-ubyte", "data/mnist/t10k-labels.idx1-ubyte", raw_test_images, raw_test_labels);
        }
    } catch (const std::exception& ex) {
        used_real_mnist = false;
        std::cerr << "MNIST IDX files not found/invalid, using synthetic fallback: " << ex.what() << "\n";
        generate_synthetic_dense(raw_train_images, raw_train_labels, 1500);
        generate_synthetic_dense(raw_test_images, raw_test_labels, 200);
    }

    const MnistDenseData train = preprocess_dense(raw_train_images, raw_train_labels, 1000);
    const MnistDenseData test = preprocess_dense(raw_test_images, raw_test_labels, 20);

    Dense dense1(28 * 28, 40);
    Tanh tanh1;
    Dense dense2(40, 10);
    Tanh tanh2;

    Network network;
    network.push_back(LayerOps{
        [&](const Matrix& x) { return dense1.forward(x); },
        [&](const Matrix& g, double lr) { return dense1.backward(g, lr); }});
    network.push_back(LayerOps{
        [&](const Matrix& x) { return tanh1.forward(x); },
        [&](const Matrix& g, double lr) { return tanh1.backward(g, lr); }});
    network.push_back(LayerOps{
        [&](const Matrix& x) { return dense2.forward(x); },
        [&](const Matrix& g, double lr) { return dense2.backward(g, lr); }});
    network.push_back(LayerOps{
        [&](const Matrix& x) { return tanh2.forward(x); },
        [&](const Matrix& g, double lr) { return tanh2.backward(g, lr); }});

    train_dense(network, train.x, train.y, 20, 0.1);

    int correct = 0;
    for (std::size_t i = 0; i < test.x.size(); ++i) {
        const Matrix out = predict_dense(network, test.x[i]);
        const int pred = argmax_column_vector(out);
        const int truth = argmax_column_vector(test.y[i]);
        if (pred == truth)
            ++correct;
        std::cout << "pred: " << pred << "\ttrue: " << truth << "\n";
    }

    const double acc = 100.0 * static_cast<double>(correct) / static_cast<double>(test.x.size());
    std::cout << "Accuracy(" << (used_real_mnist ? "real MNIST" : "synthetic fallback") << "): " << acc << "%\n";
    return 0;
}
