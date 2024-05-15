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
    auto [seed, each_epoch, sample_size, epochs, batch_size, dim, test_part, learning_rate,
          momentum, epsilon, ans, layer_sizes, act_funcs] = SetupPredictSinx();

    Random rnd(seed);

    VectorSet X = rnd.GenDataset(sample_size, dim);
    VectorSet y = ApplyFunc(X, [](const Vector& x) { return Vector(x.array().sin()); });
    auto [X_train, y_train, X_test, y_test] = SplitTrainTest(X, y, test_part);

    DataLoader train_data_loader(X_train, y_train, batch_size);

    std::vector<Layer> layers;
    for (SizeType i = 0; i + 1 < std::ssize(layer_sizes); ++i) {
        layers.emplace_back(layer_sizes[i], layer_sizes[i + 1], act_funcs[i]);
    }

    Net model(layers);
    Optimizer optimizer = SGD(learning_rate, momentum);
    LossFunction loss_function = MSE();

    model.Fit(train_data_loader, optimizer, loss_function, epochs, TrainingInfo::On, X_test, y_test,
              each_epoch);

    VectorSet preds = model.Predict(X_test);
    DataType loss = loss_function->Compute(preds, y_test);

    EXPECT_NEAR(loss, ans, epsilon);
}

TEST(NNCorrection, StressAllComponents) {
    auto [seed, iters, sample_size, epochs, batch_size, dim, test_part, learning_rate, momentum,
          layer_sizes, act_funcs] = SetupStressAllComponents();

    Random rnd(seed);
    for (SizeType _ = 0; _ < iters; ++_) {
        VectorSet X = rnd.GenDataset(sample_size, dim);
        VectorSet y = ApplyFunc(X, [](const Vector& x) { return Vector(x.array().sin()); });
        auto [X_train, y_train, X_test, y_test] = SplitTrainTest(X, y, test_part);

        DataLoader train_data_loader(X_train, y_train, batch_size);

        std::vector<Layer> layers;
        for (SizeType i = 0; i + 1 < std::ssize(layer_sizes); ++i) {
            layers.emplace_back(layer_sizes[i], layer_sizes[i + 1], act_funcs[i]);
        }

        Net model(layers);
        Optimizer optimizer = Adagrad(learning_rate, momentum);
        LossFunction loss_function = MAE();

        model.Fit(train_data_loader, optimizer, loss_function, epochs);

        VectorSet preds = model.Predict(X_test);
        DataType loss = loss_function->Compute(preds, y_test);
    }
}

TEST(NNCorrection, PredictMNIST) {
    auto [train_path, test_path, seed, epochs, batch_size, dim, test_part, learning_rate, momentum,
          layer_sizes, act_funcs] = SetupPredictMNIST();

    auto [X_train, labels_train] = ReadCSV(train_path);
    auto [X_test, labels_test] = ReadCSV(test_path);

    SizeType train_size = std::ssize(X_train);
    SizeType test_size = std::ssize(X_test);

    VectorSet y_train = GetUnitVectors(dim, labels_train);
    VectorSet y_test = GetUnitVectors(dim, labels_test);

    DataLoader train_data_loader(X_train, y_train, batch_size);

    std::vector<Layer> layers;

    for (SizeType i = 0; i + 1 < std::ssize(layer_sizes); ++i) {
        layers.emplace_back(layer_sizes[i], layer_sizes[i + 1], act_funcs[i]);
    }

    Net model(layers);
    Optimizer optimizer = SGD(learning_rate, momentum);
    LossFunction loss_function = CrossEntropy();
    model.Fit(train_data_loader, optimizer, loss_function, epochs, TrainingInfo::On, X_test,
              y_test);

    VectorSet logits = model.Predict(X_test);
    Labels preds;
    preds.reserve(logits.size());

    for (const auto& logit : logits) {
        preds.push_back(Argmax(logit));
    }

    std::cout << Accuracy(preds, labels_test) << '\n';
}
