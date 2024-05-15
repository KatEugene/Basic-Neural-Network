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

    SizeType each_epoch = 20;
    SizeType sample_size = 1e5;
    SizeType epochs = 1e2;
    SizeType batch_size = 64;
    SizeType dim = 1;

    DataType test_part = 0.3;
    DataType learning_rate = 4e-2;
    DataType momentum = 1e-5;
    DataType epsilon = 1e-3;
    DataType ans = 0.063;

    std::vector<SizeType> layer_sizes = {1, 8, 8, 1};
    std::vector<ActivationFunction> act_funcs = {Id(), Id(), Id()};
};

struct SetupStressAllComponents {
    uint64_t seed = 42;

    SizeType iters = 1e3;

    SizeType sample_size = 1e2;
    SizeType epochs = 10;
    SizeType batch_size = 64;
    SizeType dim = 3;

    DataType test_part = 0.3;
    DataType learning_rate = 4e-2;
    DataType momentum = 1e-3;

    std::vector<SizeType> layer_sizes = {1, 2, 4, 8, 8, 4, 2, 1};
    std::vector<ActivationFunction> act_funcs = {Id(),      ReLU(), Sigmoid(), Tanh(),
                                                 Softmax(), ReLU(), Id()};
};

struct SetupPredictMNIST {
    const std::string train_path = "../tests/data/mnist_train.csv";
    const std::string test_path = "../tests/data/mnist_test.csv";

    uint64_t seed = 42;

    SizeType epochs = 10;
    SizeType batch_size = 128;
    SizeType dim = 10;

    DataType test_part = 0.2;
    DataType learning_rate = 1e-2;
    DataType momentum = 0;

    std::vector<SizeType> layer_sizes = {784, 256, 128, 10};
    std::vector<ActivationFunction> act_funcs = {ReLU(), ReLU(), Softmax()};
};

}  // namespace NeuralNetwork
