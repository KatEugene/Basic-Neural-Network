#pragma once

#include <tuple>
#include "GlobalUsings.h"

namespace NeuralNetwork {

class Batch {
    class BIterator {
        using Iterator = VectorSet::const_iterator;

        Iterator cur_iter_X_, cur_iter_y_;

    public:
        BIterator(Iterator begin_X, Iterator begin_y);
        std::pair<Vector, Vector> operator*();
        BIterator& operator++();
        bool operator!=(const BIterator& other) const;
    };

public:
    Batch(const VectorSet& batch_X, const VectorSet& batch_y);
    BIterator begin() const;
    BIterator end() const;
    SizeType GetSize() const {
        return std::ssize(batch_X_);
    }

    VectorSet batch_X_, batch_y_;
    BIterator begin_, end_;
};

std::tuple<VectorSet, VectorSet, VectorSet, VectorSet> TrainTestSplit(const VectorSet& dataset_X,
                                                                      const VectorSet& dataset_y,
                                                                      DataType test_part);

class DataLoader {

    class BatchIterator {
        using Iterator = VectorSet::const_iterator;

        Iterator begin_X_, begin_y_;
        SizeType cur_pos_ = 0;
        SizeType batch_size_;
        SizeType max_size_;

    public:
        BatchIterator(Iterator begin_X, Iterator begin_y, SizeType batch_size, SizeType max_size);
        Batch operator*();
        BatchIterator& operator++();
        bool operator!=(const BatchIterator& other) const;
    };

public:
    DataLoader(const VectorSet& dataset_X, const VectorSet& dataset_y, SizeType batch_size);
    BatchIterator begin() const;
    BatchIterator end() const;

    BatchIterator begin_, end_;
};

}  // namespace NeuralNetwork
