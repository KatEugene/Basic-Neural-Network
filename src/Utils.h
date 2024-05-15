#pragma once

#include <tuple>
#include <string>
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

template <typename RandDistr = Eigen::Rand::NormalGen<DataType>,
          typename RandGen = Eigen::Rand::Vmt19937_64>
class Random {
    static constexpr uint64_t k_default_seed = 42;

    mutable RandGen urng_;
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

template<class T1, class T2>
struct DataPair {
    T1 x;
    T2 y;
};

class Batch {
    using Iterator = VectorSet::const_iterator;
    
    class BIterator {
        Iterator cur_iter_x_, cur_iter_y_;

    public:
        BIterator(Iterator begin_x, Iterator begin_y);
        DataPair<const Vector&, const Vector&> operator*() const;
        BIterator& operator++();
        bool operator==(const BIterator& other) const;
        bool operator!=(const BIterator& other) const;
    };

    SizeType size_;
    BIterator begin_, end_;

public:
    Batch(Iterator begin_x, Iterator end_x, Iterator begin_y, Iterator end_y);
    BIterator begin() const;
    BIterator end() const;
    SizeType GetSize() const;

private:
    int GetSafeSize(SizeType size1, SizeType size2) const;
};


class DataLoader {
    class BatchIterator {
        using Iterator = VectorSet::const_iterator;

        Iterator begin_x_, begin_y_;
        SizeType cur_pos_ = 0;
        SizeType batch_size_;
        SizeType max_size_;

    public:
        BatchIterator(Iterator begin_x, Iterator begin_y, SizeType batch_size, SizeType max_size);
        Batch operator*() const;
        BatchIterator& operator++();
        bool operator==(const BatchIterator& other) const;
        bool operator!=(const BatchIterator& other) const;
    };

    BatchIterator begin_, end_;

public:
    DataLoader(const VectorSet& dataset_x, const VectorSet& dataset_y, SizeType batch_size);
    BatchIterator begin() const;
    BatchIterator end() const;
};

DataPair<VectorSet, VectorSet> ReadCSV(const std::string& filename);
DataType Accuracy(const VectorSet& predicted, const VectorSet& expected);

}  // namespace NeuralNetwork
