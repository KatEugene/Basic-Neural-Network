#include <gtest/gtest.h>

#include "NeuralNetwork.h"
#include "GlobalUsings.h"
#include "Helpers.h"

using namespace NeuralNetwork;

TEST(NNCorrection, PredictSinx) {
    Eigen::Rand::Vmt19937_64 urng{42};

    SizeType sample_size = 1e5;
    SizeType epochs = 100;
    SizeType batch_size = 64;
    SizeType dim = 1;

    DataType test_part = 0.3;
    DataType learning_rate = 4e-2;
    DataType epsilon = 1e-10;

    std::vector<SizeType> layer_sizes = {1, 8, 8, 1};

    Eigen::Rand::UniformRealGen<DataType> norm_gen{-5, 5};
    VectorSet X = GenDistr(sample_size, dim, norm_gen, urng);
    VectorSet y = ApplyFunc(X, [](const Vector& x) { return x.array().sin(); });

    for (int32_t i = 0; i < sample_size; ++i) {
        X[i] = Eigen::Rand::uniformReal<Matrix>(1, 1, urng, -1, 1);
        y[i] = X[i].array().sin();
    }

    auto [X_train, y_train, X_test, y_test] = TrainTestSplit(X, y, test_part);

    DataLoader train_data_loader(X_train, y_train, batch_size);

    Layer layer1(layer_sizes[0], layer_sizes[1], Id());
    Layer layer2(layer_sizes[1], layer_sizes[2], Sigmoid());
    Layer layer3(layer_sizes[2], layer_sizes[3], Id());

    Net basic_nn({layer1, layer2, layer3});
    Optimizer optimizer = SGD(learning_rate);
    LossFunction loss_function = MSE();

    basic_nn.Fit(train_data_loader, optimizer, loss_function, epochs, X, y);

    VectorSet preds = basic_nn.Predict(X_test);

    DataType loss = loss_function->Compute(preds, y_test);
    DataType ans = 1e-11;

    EXPECT_NEAR(loss, ans, epsilon);
}

/*
TODO
TEST_CASE("Predict MNIST") {
}
*/
