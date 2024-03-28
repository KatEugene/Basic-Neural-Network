#pragma once

#include <tuple>

namespace NeuralNetwork {

struct Batch {
	VectorSet X;
	VectorSet y;
}

std::tuple<VectorSet, VectorSet, VectorSet, VectorSet> TrainTestSplit(const VectorSet& dataset_X, const VectorSet& dataset_y, DataType test_part) {
	int32_t test_size = dataset_X.size() * test_part;
	VectorSet X_train, y_train, X_test, y_test;
	X_test = {dataset_X.begin(), dataset_X.begin() + test_size};
	y_test = {dataset_y.begin(), dataset_y.begin() + test_size};
	X_train = {dataset_X.begin() + test_size, dataset_X.end()};
	y_train = {dataset_y.begin() + test_size, dataset_y.end()};
	return std::tie(X_test, y_test, X_train, y_train);
}

class DataLoader {
    BatchIterator begin_, end_;
	
	template<typename Iterator>
	class BatchIterator {
	public:
	    BatchIterator(Iterator iterator_begin, Iterator iterator_end)
	        : iterator_end_(iterator_begin), global_end_(iterator_end) {
	        ShiftEnd();
	    }
	    auto operator*() const {
	        return IteratorRange(iterator_begin_, iterator_end_);
	    }
	    BatchIterator& operator++() {
	        ShiftEnd();
	        return *this;
	    }
	    bool operator!=(const BatchIterator& other) const {
	        return iterator_begin_ != other.iterator_begin_;
	    }

	private:
	    void ShiftEnd() {
	        iterator_begin_ = iterator_end_;
	        while (iterator_end_ != global_end_ && *iterator_end_ == *iterator_begin_) {
	            ++iterator_end_;
	        }
	    }

	    Iterator iterator_begin_;
	    Iterator iterator_end_;
	    Iterator global_end_;
	};

public:
	DataLoader(VectorSet ) {

	}
	BatchIterator begin() const {  // NOLINT
        return BatchIterator();
    }

    BatchIterator end() const {  // NOLINT
        return BatchIterator();
    }
};

} // namespace NeuralNetwork