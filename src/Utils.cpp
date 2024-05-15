#include "Utils.h"

namespace NeuralNetwork {

SplitResult SplitTrainTest(const VectorSet& dataset_x, const VectorSet& dataset_y,
                           DataType test_part) {
    int32_t test_size = dataset_x.size() * test_part;
    VectorSet x_train, y_train, x_test, y_test;
    x_train = {dataset_x.begin() + test_size, dataset_x.end()};
    y_train = {dataset_y.begin() + test_size, dataset_y.end()};
    x_test = {dataset_x.begin(), dataset_x.begin() + test_size};
    y_test = {dataset_y.begin(), dataset_y.begin() + test_size};
    return std::tuple(std::move(x_train), std::move(y_train), std::move(x_test), std::move(y_test));
}

DataPair<VectorSet, Labels> ReadCSV(const std::string& path) {
    std::ifstream file(path);
    std::string line;
    VectorSet data;
    Labels labels;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        SizeType label = -1;
        std::vector<DataType> values;

        while (std::getline(ss, cell, ',')) {
            if (label == -1) {
                label = std::stoi(cell);
            } else {
                values.push_back(std::stod(cell));
            }
        }

        labels.emplace_back(label);
        data.emplace_back(Eigen::Map<Vector>(values.data(), std::ssize(values)));
    }

    return {data, labels};
}
DataType Accuracy(const Labels& predicted, const Labels& expected) {
    assert(predicted.size() == expected.size() && "Predicted and expected size must be same");

    DataType sum = 0;
    SizeType size = std::ssize(predicted);
    for (SizeType i = 0; i < size; ++i) {
        sum += predicted[i] == expected[i];
    }
    return sum / size;
}
VectorSet GetUnitVectors(SizeType dim, const Labels& labels) {
    SizeType labels_size = std::ssize(labels);
    VectorSet data(labels_size);
    for (SizeType i = 0; i < labels_size; ++i) {
        data[i] = Vector::Zero(dim);
        assert(labels[i] < dim && "Labels must be in range [0, dim - 1]");
        data[i](labels[i]) = 1;
    }
    return data;
}
SizeType Argmax(const Vector& x) {
    int arg_max = 0;
    for (SizeType i = 1; i < x.size(); ++i) {
        if (x(i) > x(arg_max)) {
            arg_max = i;
        }
    }
    return arg_max;
}

}  // namespace NeuralNetwork
