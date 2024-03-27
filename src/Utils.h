#pragma once

#include <tuple>

namespace NeuralNetwork {

struct Batch {
	VectorSet X;
	VectorSet y;
}

std::tuple<VectorSet, VectorSet, VectorSet, VectorSet> TrainTestSplit(VectorSet dataset_X, VectorSet dataset_y, test_size) {

}

class DataLoader {
	class BatchIterator {
	public:
	    BatchIterator() = default;
	    BatchIterator(int64_t pos, size_t step) : pos_(pos), step_(step) {
	    }
	    Batch operator*() const {
	        return pos_;
	    }
	    BatchIterator& operator++() {
	        pos_ += step_;
	        return *this;
	    }
	    bool operator!=(const BatchIterator& other) const {
	        return pos_ < other.pos_;
	    }

	private:
	    int64_t pos_;
	    size_t step_;
	};

    Iterator begin_, end_;
public:
	BatchIterator begin() const {  // NOLINT
        return begin_;
    }

    BatchIterator end() const {  // NOLINT
        return end_;
    }
};

} // namespace NeuralNetwork