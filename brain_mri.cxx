#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cerrno>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

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

struct Dataset {
    TensorDataset x;
    LabelDataset y;
};

struct Split {
    Dataset train;
    Dataset test;
};

struct Config {
    std::string data_root = "data/brain_mri";
    std::string image_path;
    std::string save_path = "checkpoints/brain_mri.ckpt";
    std::string load_path;
    int image_size = 64;
    int epochs = 10;
    double learning_rate = 0.01;
    bool has_image = false;
    bool save_checkpoint = false;
    bool load_checkpoint = false;
    bool infer_only = false;
    bool force_cpu = false;
};

static void write_matrix(std::ostream& out, const Matrix& m) {
    out << m.size() << " " << (m.empty() ? 0 : m[0].size()) << "\n";
    for (std::size_t r = 0; r < m.size(); ++r) {
        for (std::size_t c = 0; c < m[r].size(); ++c) {
            out << m[r][c];
            if (c + 1 < m[r].size())
                out << " ";
        }
        out << "\n";
    }
}

static Matrix read_matrix(std::istream& in) {
    std::size_t rows = 0;
    std::size_t cols = 0;
    if (!(in >> rows >> cols))
        throw std::runtime_error("Failed to read matrix shape from checkpoint");

    Matrix m(rows, np::vd(cols, 0.0));
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < cols; ++c)
            if (!(in >> m[r][c]))
                throw std::runtime_error("Failed to read matrix value from checkpoint");
    return m;
}

static void write_tensor3d(std::ostream& out, const std::vector<Matrix>& t) {
    out << t.size() << "\n";
    for (std::size_t i = 0; i < t.size(); ++i)
        write_matrix(out, t[i]);
}

static std::vector<Matrix> read_tensor3d(std::istream& in) {
    std::size_t d = 0;
    if (!(in >> d))
        throw std::runtime_error("Failed to read tensor3d depth from checkpoint");
    std::vector<Matrix> t;
    t.reserve(d);
    for (std::size_t i = 0; i < d; ++i)
        t.push_back(read_matrix(in));
    return t;
}

static void write_tensor4d(std::ostream& out, const std::vector<std::vector<Matrix>>& t) {
    out << t.size() << "\n";
    for (std::size_t i = 0; i < t.size(); ++i)
        write_tensor3d(out, t[i]);
}

static std::vector<std::vector<Matrix>> read_tensor4d(std::istream& in) {
    std::size_t d = 0;
    if (!(in >> d))
        throw std::runtime_error("Failed to read tensor4d depth from checkpoint");
    std::vector<std::vector<Matrix>> t;
    t.reserve(d);
    for (std::size_t i = 0; i < d; ++i)
        t.push_back(read_tensor3d(in));
    return t;
}

static void save_checkpoint(
    const std::string& path,
    int image_size,
    const Convolutional& conv,
    const Dense& dense1,
    const Dense& dense2) {
    auto make_dir_if_missing = [](const std::string& dir) {
        if (dir.empty())
            return;
#if defined(_WIN32)
        const int rc = _mkdir(dir.c_str());
#else
        const int rc = mkdir(dir.c_str(), 0755);
#endif
        if (rc != 0 && errno != EEXIST)
            throw std::runtime_error("Failed to create directory: " + dir);
    };

    auto ensure_parent_dirs = [&](const std::string& file_path) {
        std::size_t sep = file_path.find_last_of("/\\");
        if (sep == std::string::npos)
            return;

        const std::string parent = file_path.substr(0, sep);
        std::string current;
        for (std::size_t i = 0; i < parent.size(); ++i) {
            const char ch = parent[i];
            current.push_back(ch);
            if (ch == '/' || ch == '\\') {
                if (current.size() > 1)
                    make_dir_if_missing(current);
            }
        }
        make_dir_if_missing(current);
    };

    ensure_parent_dirs(path);

    std::ofstream out(path.c_str());
    if (!out)
        throw std::runtime_error("Could not open checkpoint for writing: " + path);

    out << "NN_BRAIN_MRI_CKPT_V1\n";
    out << image_size << "\n";

    write_tensor4d(out, conv.get_kernels());
    write_tensor3d(out, conv.get_biases());
    write_matrix(out, dense1.get_weights());
    write_matrix(out, dense1.get_bias());
    write_matrix(out, dense2.get_weights());
    write_matrix(out, dense2.get_bias());
}

static void load_checkpoint(
    const std::string& path,
    int expected_image_size,
    Convolutional& conv,
    Dense& dense1,
    Dense& dense2) {
    std::ifstream in(path.c_str());
    if (!in)
        throw std::runtime_error("Could not open checkpoint for reading: " + path);

    std::string magic;
    if (!(in >> magic) || magic != "NN_BRAIN_MRI_CKPT_V1")
        throw std::runtime_error("Invalid checkpoint format");

    int stored_image_size = 0;
    if (!(in >> stored_image_size))
        throw std::runtime_error("Failed to read checkpoint image size");
    if (stored_image_size != expected_image_size)
        throw std::runtime_error("Checkpoint image size does not match --size argument");

    conv.set_kernels(read_tensor4d(in));
    conv.set_biases(read_tensor3d(in));
    dense1.set_weights(read_matrix(in));
    dense1.set_bias(read_matrix(in));
    dense2.set_weights(read_matrix(in));
    dense2.set_bias(read_matrix(in));
}

static std::string to_lower_copy(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return out;
}

static std::string basename_of(const std::string& p) {
    std::size_t pos = p.find_last_of("/\\");
    if (pos == std::string::npos)
        return p;
    return p.substr(pos + 1);
}

static std::vector<cv::String> safe_glob(const std::string& pattern) {
    std::vector<cv::String> out;
    try {
        cv::glob(pattern, out, false);
    } catch (...) {
        out.clear();
    }
    return out;
}

static Matrix make_label(double v) {
    Matrix y(1, np::vd(1, 0.0));
    y[0][0] = v;
    return y;
}

static Tensor3D load_image_tensor(const std::string& path, int image_size) {
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (img.empty())
        throw std::runtime_error("Failed to read image: " + path);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(image_size, image_size), 0.0, 0.0, cv::INTER_LINEAR);

    Tensor3D t(1, Matrix(static_cast<std::size_t>(image_size), np::vd(static_cast<std::size_t>(image_size), 0.0)));
    for (int r = 0; r < image_size; ++r) {
        for (int c = 0; c < image_size; ++c) {
            const unsigned char px = resized.at<unsigned char>(r, c);
            t[0][static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = static_cast<double>(px) / 255.0;
        }
    }
    return t;
}

static void append_class_samples(
    const std::string& class_dir,
    double class_value,
    int image_size,
    TensorDataset& x,
    LabelDataset& y) {
    const std::vector<cv::String> files = safe_glob(class_dir + "/*");

    for (std::size_t i = 0; i < files.size(); ++i) {
        try {
            x.push_back(load_image_tensor(files[i], image_size));
            y.push_back(make_label(class_value));
        } catch (...) {
            // Skip unreadable image files.
        }
    }
}

static Split load_brain_mri_dataset(const Config& cfg, double train_ratio = 0.8) {
    Dataset all;

    std::vector<std::string> yes_dirs;
    std::vector<std::string> no_dirs;

    // Common direct folder names.
    yes_dirs.push_back(cfg.data_root + "/yes");
    yes_dirs.push_back(cfg.data_root + "/Yes");
    yes_dirs.push_back(cfg.data_root + "/YES");
    no_dirs.push_back(cfg.data_root + "/no");
    no_dirs.push_back(cfg.data_root + "/No");
    no_dirs.push_back(cfg.data_root + "/NO");

    // Also inspect one level below data_root for label-like directory names.
    const std::vector<cv::String> children = safe_glob(cfg.data_root + "/*");
    for (std::size_t i = 0; i < children.size(); ++i) {
        const std::string child = children[i];
        const std::string name = to_lower_copy(basename_of(child));

        if (name == "yes" || name == "tumor" || name == "tumour" || name == "positive") {
            yes_dirs.push_back(child);
        } else if (name == "no" || name == "notumor" || name == "no_tumor" || name == "negative") {
            no_dirs.push_back(child);
        }
    }

    for (std::size_t i = 0; i < yes_dirs.size(); ++i)
        append_class_samples(yes_dirs[i], 1.0, cfg.image_size, all.x, all.y);
    for (std::size_t i = 0; i < no_dirs.size(); ++i)
        append_class_samples(no_dirs[i], 0.0, cfg.image_size, all.x, all.y);

    if (all.x.empty()) {
        throw std::runtime_error(
            "No readable images found under " + cfg.data_root + ". Expected class folders like yes/no (any case) with image files.");
    }

    std::vector<std::size_t> idx(all.x.size());
    for (std::size_t i = 0; i < idx.size(); ++i)
        idx[i] = i;

    std::mt19937 rng(123);
    std::shuffle(idx.begin(), idx.end(), rng);

    const std::size_t train_count = static_cast<std::size_t>(train_ratio * static_cast<double>(idx.size()));

    Split s;
    s.train.x.reserve(train_count);
    s.train.y.reserve(train_count);
    s.test.x.reserve(idx.size() - train_count);
    s.test.y.reserve(idx.size() - train_count);

    for (std::size_t k = 0; k < idx.size(); ++k) {
        const std::size_t i = idx[k];
        if (k < train_count) {
            s.train.x.push_back(all.x[i]);
            s.train.y.push_back(all.y[i]);
        } else {
            s.test.x.push_back(all.x[i]);
            s.test.y.push_back(all.y[i]);
        }
    }

    return s;
}

static Tensor3D sigmoid_tensor(const Tensor3D& in) {
    Tensor3D out = in;
    for (std::size_t d = 0; d < out.size(); ++d)
        for (std::size_t r = 0; r < out[d].size(); ++r)
            for (std::size_t c = 0; c < out[d][r].size(); ++c)
                out[d][r][c] = 1.0 / (1.0 + std::exp(-out[d][r][c]));
    return out;
}

static Tensor3D sigmoid_tensor_backward(const Tensor3D& grad, const Tensor3D& act) {
    Tensor3D out = grad;
    for (std::size_t d = 0; d < out.size(); ++d)
        for (std::size_t r = 0; r < out[d].size(); ++r)
            for (std::size_t c = 0; c < out[d][r].size(); ++c)
                out[d][r][c] *= act[d][r][c] * (1.0 - act[d][r][c]);
    return out;
}

static Matrix forward_binary(
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
    return sig2.forward(h2);
}

static void train_binary(
    Convolutional& conv,
    Reshape& reshape,
    Dense& dense1,
    Sigmoid& sig1,
    Dense& dense2,
    Sigmoid& sig2,
    const Dataset& train,
    int epochs,
    double learning_rate) {
    for (int e = 0; e < epochs; ++e) {
        double loss_sum = 0.0;
        for (std::size_t i = 0; i < train.x.size(); ++i) {
            const Tensor3D conv_out = conv.forward(train.x[i]);
            const Tensor3D conv_act = sigmoid_tensor(conv_out);
            const Matrix flat = reshape.forward(conv_act);
            const Matrix h1 = dense1.forward(flat);
            const Matrix a1 = sig1.forward(h1);
            const Matrix h2 = dense2.forward(a1);
            const Matrix out = sig2.forward(h2);

            loss_sum += loss::binary_cross_entropy(train.y[i], out);

            Matrix grad = loss::binary_cross_entropy_prime(train.y[i], out);
            grad = sig2.backward(grad, learning_rate);
            grad = dense2.backward(grad, learning_rate);
            grad = sig1.backward(grad, learning_rate);
            grad = dense1.backward(grad, learning_rate);

            Tensor3D grad3 = reshape.backward(grad, learning_rate);
            grad3 = sigmoid_tensor_backward(grad3, conv_act);
            conv.backward(grad3, learning_rate);
        }

        loss_sum /= static_cast<double>(train.x.size());
        std::cout << (e + 1) << "/" << epochs << ", loss=" << loss_sum << "\n";
    }
}

static double evaluate_binary(
    Convolutional& conv,
    Reshape& reshape,
    Dense& dense1,
    Sigmoid& sig1,
    Dense& dense2,
    Sigmoid& sig2,
    const Dataset& test) {
    int correct = 0;
    for (std::size_t i = 0; i < test.x.size(); ++i) {
        const Matrix out = forward_binary(conv, reshape, dense1, sig1, dense2, sig2, test.x[i]);
        const int pred = (out[0][0] >= 0.5) ? 1 : 0;
        const int truth = (test.y[i][0][0] >= 0.5) ? 1 : 0;
        if (pred == truth)
            ++correct;
    }
    return 100.0 * static_cast<double>(correct) / static_cast<double>(test.x.size());
}

static Config parse_args(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) {
            c.data_root = argv[++i];
        } else if (arg == "--image" && i + 1 < argc) {
            c.image_path = argv[++i];
            c.has_image = true;
        } else if (arg == "--size" && i + 1 < argc) {
            c.image_size = std::atoi(argv[++i]);
        } else if (arg == "--epochs" && i + 1 < argc) {
            c.epochs = std::atoi(argv[++i]);
        } else if (arg == "--lr" && i + 1 < argc) {
            c.learning_rate = std::atof(argv[++i]);
        } else if (arg == "--save" && i + 1 < argc) {
            c.save_path = argv[++i];
            c.save_checkpoint = true;
        } else if (arg == "--load" && i + 1 < argc) {
            c.load_path = argv[++i];
            c.load_checkpoint = true;
        } else if (arg == "--infer-only" || arg == "--infer_only" || arg == "--infer") {
            c.infer_only = true;
        } else if (arg == "--cpu") {
            c.force_cpu = true;
        }
    }
    return c;
}

int main(int argc, char** argv) {
    const Config cfg = parse_args(argc, argv);

    try {
        std::cout << "Mode: " << (cfg.infer_only ? "infer-only" : "train+eval") << "\n";

        const int h = cfg.image_size;
        const int w = cfg.image_size;

        Convolutional conv({1, h, w}, 3, 4);
        Reshape reshape({4, h - 2, w - 2}, {4 * (h - 2) * (w - 2), 1});
        Dense dense1(4 * (h - 2) * (w - 2), 64);
        Sigmoid sig1;
        Dense dense2(64, 1);
        Sigmoid sig2;

        if (cfg.force_cpu) {
            conv.set_use_gpu(false);
            dense1.set_use_gpu(false);
            dense2.set_use_gpu(false);
        }

        std::cout << "Backend (conv/dense): "
                  << (conv.is_using_gpu() ? "GPU" : "CPU")
                  << "/"
                  << (dense1.is_using_gpu() ? "GPU" : "CPU")
                  << (cfg.force_cpu ? " (forced CPU)" : " (auto)")
                  << "\n";

        if (cfg.load_checkpoint) {
            load_checkpoint(cfg.load_path, cfg.image_size, conv, dense1, dense2);
            std::cout << "Loaded checkpoint: " << cfg.load_path << "\n";
        }

        if (cfg.infer_only) {
            if (!cfg.load_checkpoint)
                throw std::runtime_error("--infer-only requires --load <checkpoint_path>");

            if (cfg.has_image) {
                const Tensor3D x = load_image_tensor(cfg.image_path, cfg.image_size);
                const Matrix out = forward_binary(conv, reshape, dense1, sig1, dense2, sig2, x);
                const double p = out[0][0];
                std::cout << "Arbitrary image probability: " << p << "\n";
                std::cout << "Prediction: " << ((p >= 0.5) ? "yes" : "no") << "\n";
            } else {
                std::cout << "Checkpoint loaded. Provide --image <path> to run inference.\n";
            }
            return 0;
        }

        if (!cfg.infer_only) {
            const Split split = load_brain_mri_dataset(cfg, 0.8);

            std::cout << "Train samples: " << split.train.x.size() << ", Test samples: " << split.test.x.size() << "\n";

            train_binary(conv, reshape, dense1, sig1, dense2, sig2, split.train, cfg.epochs, cfg.learning_rate);

            const double acc = evaluate_binary(conv, reshape, dense1, sig1, dense2, sig2, split.test);
            std::cout << "Brain MRI binary accuracy: " << acc << "%\n";

            if (cfg.save_checkpoint) {
                save_checkpoint(cfg.save_path, cfg.image_size, conv, dense1, dense2);
                std::cout << "Saved checkpoint: " << cfg.save_path << "\n";
            }
        }

        if (cfg.has_image) {
            const Tensor3D x = load_image_tensor(cfg.image_path, cfg.image_size);
            const Matrix out = forward_binary(conv, reshape, dense1, sig1, dense2, sig2, x);
            const double p = out[0][0];
            std::cout << "Arbitrary image probability: " << p << "\n";
            std::cout << "Prediction: " << ((p >= 0.5) ? "yes" : "no") << "\n";
        } else if (cfg.infer_only) {
            std::cout << "Checkpoint loaded. Provide --image <path> to run inference.\n";
        }

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        std::cerr << "Usage: ./brain_mri [--data data/brain_mri] [--size 64] [--epochs 10] [--lr 0.01] [--image path/to/file.jpg] [--save checkpoint_path] [--load checkpoint_path] [--infer-only] [--cpu]\n";
        return 1;
    }

    return 0;
}
