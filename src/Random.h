#pragma once

#include <tuple>
#include <string>
#include "GlobalUsings.h"

namespace NeuralNetwork {

template <typename RandDistr = Eigen::Rand::NormalGen<DataType>,
          typename RandGen = Eigen::Rand::Vmt19937_64>
class Random {
    static constexpr uint64_t k_default_seed = 42;

    RandGen urng_;
    RandDistr gen_;

public:
    Random(uint64_t seed = k_default_seed) : urng_(seed) {
    }
    Matrix GenLike(const Matrix& matrix) {
        return gen_.generateLike(matrix, urng_);
    }
    Vector GenVector(SizeType dim) {
        return gen_.template generate<Matrix>(dim, 1, urng_);
    }
    Matrix GenMatrix(SizeType dim1, SizeType dim2) {
        return gen_.template generate<Matrix>(dim1, dim2, urng_);
    }
    VectorSet GenDataset(SizeType sample_size, SizeType dim) {
        VectorSet data(sample_size);
        for (int i = 0; i < sample_size; ++i) {
            data[i] = GenVector(dim);
        }
        return data;
    }
};
}  // namespace NeuralNetwork
