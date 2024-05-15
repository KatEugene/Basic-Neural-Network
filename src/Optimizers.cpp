#include "Optimizers.h"

namespace NeuralNetwork {

SGD::SGD(DataType learning_rate, DataType momentum)
    : learning_rate_(learning_rate), momentum_(momentum) {
}

void SGD::DoStep(const std::vector<Matrix>& weights_grads, const std::vector<Vector>& bias_grads,
                 std::vector<Layer>* layers) {

    assert(weights_grads.size() == bias_grads.size() &&
           "Count of weights and bias gradients must be same");

    if (weights_deltas_.empty()) {
        InitDeltas(weights_grads, bias_grads);
    }
    for (size_t i = 0; i < layers->size(); ++i) {
        weights_deltas_[i] = momentum_ * weights_deltas_[i] - learning_rate_ * weights_grads[i];
        bias_deltas_[i] = momentum_ * bias_deltas_[i] - learning_rate_ * bias_deltas_[i];
        (*layers)[i].Update(weights_deltas_[i], bias_deltas_[i]);
    }
}

void SGD::InitDeltas(const std::vector<Matrix>& weights, const std::vector<Vector>& bias) {
    SizeType layers_size = std::ssize(weights);

    weights_deltas_.reserve(layers_size);
    bias_deltas_.reserve(layers_size);

    for (SizeType i = 0; i < layers_size; ++i) {
        weights_deltas_.push_back(Matrix::Zero(weights[i].rows(), weights[i].cols()));
        bias_deltas_.push_back(Vector::Zero(bias[i].rows()));
    }
}

// Adagrad::Adagrad(DataType learning_rate, DataType momentum) : learning_rate_(learning_rate),
// momentum_(momentum) {
// }

// void Adagrad::DoStep(const std::vector<Matrix>& weights_grads, const std::vector<Vector>&
// bias_grads,
//                  std::vector<Layer>* layers) const {

//     assert(weights_grads.size() == bias_grads.size() && "Count of weights and bias gradients must
//     be same");

//     if (weights_deltas_.empty()) {
//         InitDeltas(weights_grads, bias_grads);
//     }

//     for (size_t i = 0; i < layers->size(); ++i) {
//         weights_deltas_[i] = momentum_ * weights_deltas_[i] - learning_rate_ * weights_grads[i];
//         bias_deltas_[i] = momentum_ * bias_deltas_[i] - learning_rate_ * bias_deltas_[i];

//         (*layers)[i].Update(weights_deltas_[i], bias_deltas_[i]);
//     }
// }

// void Adagrad::InitDeltas(const std::vector<Matrix>& weights, const std::vector<Vector>& bias) {
//     SizeType layers_size = std::ssize(weights);

//     weights_deltas_.reserve(layers_size);
//     bias_deltas_.reserve(layers_size);

//     for (SizeType i = 0; i < layers_size; ++i) {
//         weights_deltas_.push_back(Matrix::ZeroLike(weights[i]));
//         bias_deltas_.push_back(Vector::ZeroLike(bias[i]));
//     }
// }

}  // namespace NeuralNetwork
