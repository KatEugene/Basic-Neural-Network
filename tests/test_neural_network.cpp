#include <gtest/gtest.h>

#include <NeuralNetwork.h>
#include <GlobalUsings.h>

using namespace NeuralNetwork;

TEST("Predict sin(x)") {
	Eigen::Rand::Vmt19937_64 urng{42};
	int32_t sample_size = 1e5; 

	VectorSet X(sample_size);
	VectorSet y(sample_size);

	for (int32_t i = 0; i < sample_size; ++i) {
		X[i] = Eigen::Rand::uniformReal(1, urng);
		y[i] = 2 * X[i];
	}

	auto [X_train, y_train, X_test, y_test] = TrainTestSplit(X, y, test_size=0.3);

	DataLoader train_data_loader(X_train, y_train);

	Layer layer1(1, 8, Id());
	Layer layer2(8, 8, Sigmoid());
	Layer layer3(8, 1, Id());

	NeuralNetwork basic_nn({layer1, layer2, layer3});
	Optimizer optimizer = SGD(learning_rate=4e-2);
	LossFunction loss_function = MSE();

	basic_nn.fit(train_data_loader, optimizer, loss_function, 100);
	VectorSet preds = basic_nn.predict(X_test);

	DataType loss = loss_function(preds, y_test);
	DataType ans = 0;

	std::cout << loss << '\n'; 
	EXPECT_DOUBLE_EQ(loss, ans);
}

/*
TODO
TEST_CASE("Predict linear func R^n --> R^m") {
}
TEST_CASE("Predict MNIST") {
}
*/