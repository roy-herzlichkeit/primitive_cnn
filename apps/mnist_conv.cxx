#include <algorithm>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "nn/layers/activation.h"
#include "nn/layers/convolutional.h"
#include "nn/layers/dense.h"
#include "nn/layers/reshape.h"
#include "nn/loss/losses.h"

using Matrix = np::Matrix;
using Tensor3D = std::vector<Matrix>;
using TensorDataset = std::vector<Tensor3D>;
using LabelDataset = std::vector<Matrix>;
using nn::Convolutional;
using nn::Dense;
using nn::Reshape;
using nn::Sigmoid;

struct MnistConvData {
    TensorDataset x;
    LabelDataset y;
};

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

static void load_idx_tensor(const std::string& images_path, const std::string& labels_path, TensorDataset& images, std::vector<int>& labels) {
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
        Tensor3D t(1, m);
        images.push_back(t);

        unsigned char y = 0;
        lbl.read(reinterpret_cast<char*>(&y), 1);
        if (!lbl)
            throw std::runtime_error("Unexpected EOF while reading label bytes");
        labels.push_back(static_cast<int>(y));
    }
}

static void generate_synthetic_binary(TensorDataset& images, std::vector<int>& labels, std::size_t per_class) {
    images.clear();
    labels.clear();
    images.reserve(per_class * 2);
    labels.reserve(per_class * 2);

    std::mt19937 rng(123);
    std::normal_distribution<double> noise(0.0, 10.0);

    for (std::size_t i = 0; i < per_class; ++i) {
        Matrix zero(28, np::vd(28, 0.0));
        Matrix one(28, np::vd(28, 0.0));

        // Digit 0 proxy: ring-like border.
        for (int r = 6; r <= 21; ++r) {
            zero[r][6] = 220.0;
            zero[r][21] = 220.0;
        }
        for (int c = 6; c <= 21; ++c) {
            zero[6][c] = 220.0;
            zero[21][c] = 220.0;
        }

        // Digit 1 proxy: vertical bar.
        for (int r = 5; r <= 22; ++r)
            one[r][14] = 230.0;

        for (int r = 0; r < 28; ++r) {
            for (int c = 0; c < 28; ++c) {
                zero[r][c] = std::max(0.0, std::min(255.0, zero[r][c] + noise(rng)));
                one[r][c] = std::max(0.0, std::min(255.0, one[r][c] + noise(rng)));
            }
        }

        images.push_back(Tensor3D(1, zero));
        labels.push_back(0);
        images.push_back(Tensor3D(1, one));
        labels.push_back(1);
    }
}

MnistConvData preprocess_binary_01(const TensorDataset& raw_images, const std::vector<int>& raw_labels, std::size_t limit_per_class) {
    if (raw_images.size() != raw_labels.size())
        throw std::invalid_argument("raw_images and raw_labels must have same size");

    std::vector<std::size_t> zero_idx;
    std::vector<std::size_t> one_idx;
    zero_idx.reserve(limit_per_class);
    one_idx.reserve(limit_per_class);

    for (std::size_t i = 0; i < raw_labels.size(); ++i) {
        if (raw_labels[i] == 0 && zero_idx.size() < limit_per_class)
            zero_idx.push_back(i);
        else if (raw_labels[i] == 1 && one_idx.size() < limit_per_class)
            one_idx.push_back(i);
        if (zero_idx.size() >= limit_per_class && one_idx.size() >= limit_per_class)
            break;
    }

    std::vector<std::size_t> all_idx;
    all_idx.reserve(zero_idx.size() + one_idx.size());
    all_idx.insert(all_idx.end(), zero_idx.begin(), zero_idx.end());
    all_idx.insert(all_idx.end(), one_idx.begin(), one_idx.end());

    std::mt19937 rng(7);
    std::shuffle(all_idx.begin(), all_idx.end(), rng);

    MnistConvData out;
    out.x.reserve(all_idx.size());
    out.y.reserve(all_idx.size());

    for (std::size_t k = 0; k < all_idx.size(); ++k) {
        const std::size_t i = all_idx[k];
        if (raw_images[i].size() != 1 || raw_images[i][0].size() != 28 || raw_images[i][0][0].size() != 28)
            throw std::invalid_argument("Expected each conv input image shape to be (1,28,28)");

        Tensor3D x = raw_images[i];
        for (std::size_t r = 0; r < 28; ++r)
            for (std::size_t c = 0; c < 28; ++c) {
                const double v = x[0][r][c];
                x[0][r][c] = (v > 1.0) ? (v / 255.0) : v;
            }

        Matrix y(2, np::vd(1, 0.0));
        if (raw_labels[i] == 0)
            y[0][0] = 1.0;
        else if (raw_labels[i] == 1)
            y[1][0] = 1.0;
        else
            continue;

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
        if (v[i][0] > best_v) {
            best_v = v[i][0];
            best_i = i;
        }
    }
    return static_cast<int>(best_i);
}

static Tensor3D sigmoid_tensor(const Tensor3D& in) {
    Tensor3D out = in;
    for (std::size_t d = 0; d < out.size(); ++d)
        for (std::size_t r = 0; r < out[d].size(); ++r)
            for (std::size_t c = 0; c < out[d][r].size(); ++c)
                out[d][r][c] = 1.0 / (1.0 + std::exp(-out[d][r][c]));
    return out;
}

static Tensor3D sigmoid_tensor_backward(const Tensor3D& output_gradient, const Tensor3D& sigmoid_output) {
    Tensor3D grad = output_gradient;
    for (std::size_t d = 0; d < grad.size(); ++d)
        for (std::size_t r = 0; r < grad[d].size(); ++r)
            for (std::size_t c = 0; c < grad[d][r].size(); ++c)
                grad[d][r][c] *= sigmoid_output[d][r][c] * (1.0 - sigmoid_output[d][r][c]);
    return grad;
}

Matrix predict_conv(
    Convolutional& conv,
    Reshape& reshape,
    Dense& dense1,
    Sigmoid& sig1,
    Dense& dense2,
    Sigmoid& sig2,
    const Tensor3D& x) {
    const Tensor3D conv_out = conv.forward(x);
    const Tensor3D conv_act = sigmoid_tensor(conv_out);
    const Matrix flat = reshape.forward(conv_act);
    const Matrix h1 = dense1.forward(flat);
    const Matrix a1 = sig1.forward(h1);
    const Matrix h2 = dense2.forward(a1);
    const Matrix out = sig2.forward(h2);
    return out;
}

void train_conv(
    Convolutional& conv,
    Reshape& reshape,
    Dense& dense1,
    Sigmoid& sig1,
    Dense& dense2,
    Sigmoid& sig2,
    const TensorDataset& x_train,
    const LabelDataset& y_train,
    int epochs,
    double learning_rate) {
    if (x_train.size() != y_train.size())
        throw std::invalid_argument("x_train and y_train must have same size");
    if (x_train.empty())
        throw std::invalid_argument("x_train cannot be empty");

    for (int e = 0; e < epochs; ++e) {
        double error = 0.0;

        for (std::size_t i = 0; i < x_train.size(); ++i) {
            const Tensor3D conv_out = conv.forward(x_train[i]);
            const Tensor3D conv_act = sigmoid_tensor(conv_out);
            const Matrix flat = reshape.forward(conv_act);
            const Matrix h1 = dense1.forward(flat);
            const Matrix a1 = sig1.forward(h1);
            const Matrix h2 = dense2.forward(a1);
            const Matrix out = sig2.forward(h2);

            error += loss::binary_cross_entropy(y_train[i], out);

            Matrix grad = loss::binary_cross_entropy_prime(y_train[i], out);
            grad = sig2.backward(grad, learning_rate);
            grad = dense2.backward(grad, learning_rate);
            grad = sig1.backward(grad, learning_rate);
            grad = dense1.backward(grad, learning_rate);

            Tensor3D grad3d = reshape.backward(grad, learning_rate);
            grad3d = sigmoid_tensor_backward(grad3d, conv_act);
            conv.backward(grad3d, learning_rate);
        }

        error /= static_cast<double>(x_train.size());
        std::cout << (e + 1) << "/" << epochs << ", error=" << error << "\n";
    }
}

int main(int argc, char** argv) {
    bool force_cpu = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--cpu")
            force_cpu = true;
    }

    TensorDataset raw_train_images;
    std::vector<int> raw_train_labels;
    TensorDataset raw_test_images;
    std::vector<int> raw_test_labels;

    bool used_real_mnist = true;
    try {
        try {
            load_idx_tensor("data/mnist/train-images-idx3-ubyte", "data/mnist/train-labels-idx1-ubyte", raw_train_images, raw_train_labels);
            load_idx_tensor("data/mnist/t10k-images-idx3-ubyte", "data/mnist/t10k-labels-idx1-ubyte", raw_test_images, raw_test_labels);
        } catch (...) {
            // Also support common file naming with ".idx*" instead of "-idx*".
            load_idx_tensor("data/mnist/train-images.idx3-ubyte", "data/mnist/train-labels.idx1-ubyte", raw_train_images, raw_train_labels);
            load_idx_tensor("data/mnist/t10k-images.idx3-ubyte", "data/mnist/t10k-labels.idx1-ubyte", raw_test_images, raw_test_labels);
        }
    } catch (const std::exception& ex) {
        used_real_mnist = false;
        std::cerr << "MNIST IDX files not found/invalid, using synthetic fallback: " << ex.what() << "\n";
        generate_synthetic_binary(raw_train_images, raw_train_labels, 120);
        generate_synthetic_binary(raw_test_images, raw_test_labels, 30);
    }

    const MnistConvData train = preprocess_binary_01(raw_train_images, raw_train_labels, 40);
    const MnistConvData test = preprocess_binary_01(raw_test_images, raw_test_labels, 20);

    Convolutional conv({1, 28, 28}, 3, 5);
    Reshape reshape({5, 26, 26}, {5 * 26 * 26, 1});
    Dense dense1(5 * 26 * 26, 100);
    Sigmoid sig1;
    Dense dense2(100, 2);
    Sigmoid sig2;

    if (force_cpu) {
        conv.set_use_gpu(false);
        dense1.set_use_gpu(false);
        dense2.set_use_gpu(false);
    }

    std::cout << "Backend (conv/dense): "
              << (conv.is_using_gpu() ? "GPU" : "CPU")
              << "/"
              << (dense1.is_using_gpu() ? "GPU" : "CPU")
              << (force_cpu ? " (forced CPU)" : " (auto)")
              << "\n";

    train_conv(conv, reshape, dense1, sig1, dense2, sig2, train.x, train.y, 3, 0.1);

    int correct = 0;
    for (std::size_t i = 0; i < test.x.size(); ++i) {
        const Matrix out = predict_conv(conv, reshape, dense1, sig1, dense2, sig2, test.x[i]);
        const int pred = argmax_column_vector(out);
        const int truth = argmax_column_vector(test.y[i]);
        if (pred == truth)
            ++correct;
        std::cout << "pred: " << pred << "\ttrue: " << truth << "\n";
    }

    const double acc = 100.0 * static_cast<double>(correct) / static_cast<double>(test.x.size());
    std::cout << "Binary Accuracy(" << (used_real_mnist ? "real MNIST" : "synthetic fallback") << "): " << acc << "%\n";
    return 0;
}
