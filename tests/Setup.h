#include "GlobalUsings.h"

namespace NeuralNetwork {

struct SetupGradients {
    uint64_t seed = 42;
    SizeType iters = 1e3;
    SizeType dim = 1e2;
    DataType dx = 1e-6;
    DataType epsilon = 1e-8;
};

struct SetupPredictSinx {
    uint64_t seed = 42;

    SizeType sample_size = 1e5;
    SizeType epochs = 5e1;
    SizeType batch_size = 64;
    SizeType dim = 1;

    DataType test_part = 0.3;
    DataType learning_rate = 4e-2;
    DataType momentum = 1e-5;
    DataType epsilon = 1e-3;
    DataType ans = 4e-3;
};

struct SetupPredictMNIST {
    const std::string train_path = "../tests/data/mnist_train.csv";
    const std::string test_path = "../tests/data/mnist_test.csv";

    uint64_t seed = 42;

    SizeType sample_size = 1e5;
    SizeType epochs = 5e1;
    SizeType batch_size = 64;
    SizeType dim = 1;

    DataType test_part = 0.3;
    DataType learning_rate = 3e-4;
    DataType momentum = 0;
    DataType epsilon = 1e-3;
    DataType ans = 4e-3;
};

}  // namespace NeuralNetwork
