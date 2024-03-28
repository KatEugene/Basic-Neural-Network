#include <Utils.h>

namespace NeuralNetwork {

std::tuple<VectorSet, VectorSet, VectorSet, VectorSet> TrainTestSplit(const VectorSet& dataset_X,
                                                                      const VectorSet& dataset_y,
                                                                      DataType test_part) {
    int32_t test_size = dataset_X.size() * test_part;
    VectorSet X_train, y_train, X_test, y_test;
    X_test = {dataset_X.begin(), dataset_X.begin() + test_size};
    y_test = {dataset_y.begin(), dataset_y.begin() + test_size};
    X_train = {dataset_X.begin() + test_size, dataset_X.end()};
    y_train = {dataset_y.begin() + test_size, dataset_y.end()};
    return std::tie(X_test, y_test, X_train, y_train);
}

Batch::BIterator::BIterator(Iterator iter_X, Iterator iter_y)
    : cur_iter_X_(iter_X), cur_iter_y_(iter_y) {
}
std::pair<Vector, Vector> Batch::BIterator::operator*() {
    return {*cur_iter_X_, *cur_iter_y_};
}
Batch::BIterator& Batch::BIterator::operator++() {
    ++cur_iter_X_;
    ++cur_iter_y_;
    return *this;
}
bool Batch::BIterator::operator!=(const Batch::BIterator& other) const {
    return cur_iter_X_ != other.cur_iter_X_ || cur_iter_y_ != other.cur_iter_y_;
}

Batch::Batch(const VectorSet& batch_X, const VectorSet& batch_y)
    : batch_X_(batch_X),
      batch_y_(batch_y),
      begin_(BIterator(batch_X_.begin(), batch_y_.begin())),
      end_(BIterator(batch_X_.end(), batch_y_.end())) {
    assert(batch_X_.size() == batch_y_.size() && "Batch X and y parts are not the same size");
}
Batch::BIterator Batch::begin() const {
    return begin_;
}
Batch::BIterator Batch::end() const {
    return end_;
}

DataLoader::BatchIterator::BatchIterator(Iterator begin_X, Iterator begin_y, SizeType batch_size,
                                         SizeType max_size)
    : begin_X_(begin_X), begin_y_(begin_y), batch_size_(batch_size), max_size_(max_size) {
}
Batch DataLoader::BatchIterator::operator*() {
    if (cur_pos_ + batch_size_ > max_size_) {
        return Batch{VectorSet{begin_X_ + cur_pos_, begin_X_ + max_size_},
                     VectorSet{begin_y_ + cur_pos_, begin_y_ + max_size_}};
    }
    return Batch{VectorSet{begin_X_ + cur_pos_, begin_X_ + cur_pos_ + batch_size_},
                 VectorSet{begin_y_ + cur_pos_, begin_y_ + cur_pos_ + batch_size_}};
}
DataLoader::BatchIterator& DataLoader::BatchIterator::operator++() {
    cur_pos_ = std::min(cur_pos_ + batch_size_, max_size_);
    return *this;
}
bool DataLoader::BatchIterator::operator!=(const BatchIterator& other) const {
    return begin_X_ + cur_pos_ != other.begin_X_ + other.cur_pos_;
}

DataLoader::DataLoader(const VectorSet& dataset_X, const VectorSet& dataset_y, SizeType batch_size)
    : begin_(BatchIterator(dataset_X.begin(), dataset_y.begin(), batch_size, dataset_X.size())),
      end_(BatchIterator(dataset_X.end(), dataset_y.end(), batch_size, dataset_X.size())) {
    assert(dataset_X.size() == dataset_y.size());
}
DataLoader::BatchIterator DataLoader::begin() const {
    return begin_;
}
DataLoader::BatchIterator DataLoader::end() const {
    return end_;
}

}  // namespace NeuralNetwork
