#include <gtest/gtest.h>

#include "NeuralNetwork.h"
#include "Setup.h"

using namespace NeuralNetwork;

TEST(NNCorrection, Gradients) {
    auto [seed, iters, dim, dx, epsilon] = SetupGradients();
    Random rnd(seed);
    std::vector<ActivationFunction> act_funcs = {Id(), ReLU(), Sigmoid(), Tanh(), Softmax()};
    for (SizeType _ = 0; _ < iters; ++_) {
        Vector x = rnd.GenVector(dim);
        for (const auto& act_func : act_funcs) {
            Matrix error(dim, dim);
            for (SizeType i = 0; i < dim; ++i) {
                Vector x_plus = x;
                Vector x_minus = x;
                x_plus(i) += dx;
                x_minus(i) -= dx;
                error.col(i) = (act_func->Compute(x_plus) - act_func->Compute(x_minus)) / (2 * dx);
            }
            error -= act_func->ComputeGradient(x);
            EXPECT_NEAR(error.cwiseAbs().maxCoeff(), epsilon, epsilon);
        }
    }
}

TEST(NNCorrection, PredictSinx) {
    auto [seed, sample_size, epochs, batch_size, dim, test_part, learning_rate, momentum, epsilon, ans] =
        SetupPredictSinx();

    Random rnd(seed);
    std::vector<SizeType> layer_sizes = {1, 8, 8, 1};

    VectorSet X = rnd.GenDataset(sample_size, dim);
    VectorSet y = ApplyFunc(X, [](const Vector& x) { return Vector(x.array().sin()); });

    auto [X_train, y_train, X_test, y_test] = SplitTrainTest(X, y, test_part);

    DataLoader train_data_loader(X_train, y_train, batch_size);

    std::vector<Layer> layers;

    layers.emplace_back(layer_sizes[0], layer_sizes[1], Id());
    layers.emplace_back(layer_sizes[1], layer_sizes[2], Sigmoid());
    layers.emplace_back(layer_sizes[2], layer_sizes[3], Id());

    Net basic_nn(layers);
    Optimizer optimizer = SGD(learning_rate, momentum);
    LossFunction loss_function = MSE();

    basic_nn.Fit(train_data_loader, optimizer, loss_function, epochs);

    VectorSet preds = basic_nn.Predict(X_test);

    DataType loss = loss_function->Compute(preds, y_test);

    EXPECT_NEAR(loss, ans, epsilon);
}

TEST(NNCorrection, PredictMNIST) {
    auto [train_path, test_path, seed, sample_size, epochs, batch_size, dim, test_part, learning_rate, momentum, epsilon, ans] =
        SetupPredictMNIST();
    
    std::vector<SizeType> layer_sizes = {784, 256, 128, 10};

    auto [X_train, y_train] = ReadCSV(train_path);
    auto [X_test, y_test] = ReadCSV(test_path);
    
    DataLoader train_data_loader(X_train, y_train, batch_size);

    std::vector<ActivationFunction> act_funcs = {ReLU(), ReLU(), Sigmoid()};
    std::vector<Layer> layers;
    
    for (SizeType i = 0; i + 1 < std::ssize(layer_sizes); ++i) {
        layers.emplace_back(layer_sizes[i], layer_sizes[i + 1], act_funcs[i]);
    }

    Net basic_nn(layers);
    Optimizer optimizer = SGD(learning_rate, momentum);
    LossFunction loss_function = MSE();
    basic_nn.Fit(train_data_loader, optimizer, loss_function, epochs);

    VectorSet preds = basic_nn.Predict(X_test);

    std::cout << Accuracy(preds, y_test) << '\n';
}


