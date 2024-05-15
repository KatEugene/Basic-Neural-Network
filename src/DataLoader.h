#pragma once

#include "GlobalUsings.h"

namespace NeuralNetwork {

template <class T1, class T2>
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

}  // namespace NeuralNetwork
