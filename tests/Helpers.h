#include <algorithm>
#include "GlobalUsings.h"

namespace NeuralNetwork {

template <typename RandGen, typename RandNumGen>
VectorSet GenDistr(SizeType sample_size, SizeType dim, RandGen rand_gen, RandNumGen urng) {
    VectorSet X(sample_size);
    for (int i = 0; i < sample_size; ++i) {
        X[i].resize(dim);
        X[i] = rand_gen.generateLike(X[i], urng);
    }
    return X;
}

template <typename Function>
VectorSet ApplyFunc(const VectorSet& X, Function function) {
    VectorSet y(X.size());
    std::transform(X.cbegin(), X.cend(), y.begin(), function);
    return y;
}

}  // namespace NeuralNetwork
