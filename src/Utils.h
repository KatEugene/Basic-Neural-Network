#pragma once

#include <tuple>
#include <string>
#include "DataLoader.h"
#include "GlobalUsings.h"

namespace NeuralNetwork {

template <typename Function>
VectorSet ApplyFunc(const VectorSet& data, Function&& function) {
    VectorSet y(data.size());
    std::transform(data.cbegin(), data.cend(), y.begin(), std::forward<Function>(function));
    return y;
}

using SplitResult = std::tuple<VectorSet, VectorSet, VectorSet, VectorSet>;
SplitResult SplitTrainTest(const VectorSet& dataset_x, const VectorSet& dataset_y,
                           DataType test_part);

using Labels = std::vector<SizeType>;
DataPair<VectorSet, Labels> ReadCSV(const std::string& filename);
DataType Accuracy(const Labels& predicted, const Labels& expected);
VectorSet GetUnitVectors(SizeType dim, const Labels& labels);
SizeType Argmax(const Vector& x);

}  // namespace NeuralNetwork
