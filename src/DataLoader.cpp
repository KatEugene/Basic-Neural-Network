#include "DataLoader.h"

namespace NeuralNetwork {

Batch::BIterator::BIterator(Iterator iter_x, Iterator iter_y)
    : cur_iter_x_(std::move(iter_x)), cur_iter_y_(std::move(iter_y)) {
}
DataPair<const Vector&, const Vector&> Batch::BIterator::operator*() const {
    return {*cur_iter_x_, *cur_iter_y_};
}
Batch::BIterator& Batch::BIterator::operator++() {
    ++cur_iter_x_;
    ++cur_iter_y_;
    return *this;
}
bool Batch::BIterator::operator==(const Batch::BIterator& other) const {
    return cur_iter_x_ == other.cur_iter_x_ && cur_iter_y_ == other.cur_iter_y_;
}
bool Batch::BIterator::operator!=(const Batch::BIterator& other) const {
    return cur_iter_x_ != other.cur_iter_x_ || cur_iter_y_ != other.cur_iter_y_;
}

Batch::Batch(Iterator begin_x, Iterator end_x, Iterator begin_y, Iterator end_y)
    : size_(GetSafeSize(end_x - begin_x, end_y - begin_y)),
      begin_(BIterator(std::move(begin_x), std::move(begin_y))),
      end_(BIterator(std::move(end_x), std::move(end_y))) {
}
Batch::BIterator Batch::begin() const {
    return begin_;
}
Batch::BIterator Batch::end() const {
    return end_;
}
SizeType Batch::GetSize() const {
    return size_;
}
int Batch::GetSafeSize(SizeType size1, SizeType size2) const {
    assert(size1 == size2 && "Batch x and y parts are not the same size");
    return size1;
}

DataLoader::BatchIterator::BatchIterator(Iterator begin_x, Iterator begin_y, SizeType batch_size,
                                         SizeType max_size)
    : begin_x_(std::move(begin_x)),
      begin_y_(std::move(begin_y)),
      batch_size_(batch_size),
      max_size_(max_size) {
}
Batch DataLoader::BatchIterator::operator*() const {
    if (cur_pos_ + batch_size_ > max_size_) {
        return Batch{begin_x_ + cur_pos_, begin_x_ + max_size_, begin_y_ + cur_pos_,
                     begin_y_ + max_size_};
    }
    return Batch{begin_x_ + cur_pos_, begin_x_ + cur_pos_ + batch_size_, begin_y_ + cur_pos_,
                 begin_y_ + cur_pos_ + batch_size_};
}
DataLoader::BatchIterator& DataLoader::BatchIterator::operator++() {
    cur_pos_ = std::min(cur_pos_ + batch_size_, max_size_);
    return *this;
}
bool DataLoader::BatchIterator::operator==(const BatchIterator& other) const {
    return begin_x_ + cur_pos_ == other.begin_x_ + other.cur_pos_;
}
bool DataLoader::BatchIterator::operator!=(const BatchIterator& other) const {
    return begin_x_ + cur_pos_ != other.begin_x_ + other.cur_pos_;
}

DataLoader::DataLoader(const VectorSet& dataset_x, const VectorSet& dataset_y, SizeType batch_size)
    : begin_(BatchIterator(dataset_x.begin(), dataset_y.begin(), batch_size, dataset_x.size())),
      end_(BatchIterator(dataset_x.end(), dataset_y.end(), batch_size, dataset_x.size())) {
    assert(dataset_x.size() == dataset_y.size() && "Sizes of datasets must be same");
}
DataLoader::BatchIterator DataLoader::begin() const {
    return begin_;
}
DataLoader::BatchIterator DataLoader::end() const {
    return end_;
}

}  // namespace NeuralNetwork
