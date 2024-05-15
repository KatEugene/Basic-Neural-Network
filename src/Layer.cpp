#include "Layer.h"

namespace NeuralNetwork {

Layer::Layer(Index in, Index out, ActivationFunction activation_function, Random<> gen)
    : activation_function_(std::move(activation_function)),
      weights_(SafeGen(in, out, gen)),
      bias_(SafeGen(out, gen)) {
    assert(activation_function_.isDefined() && "Activation function is not defined");
    weights_ /= weights_.norm();
    bias_ /= bias_.norm();
}
Index Layer::GetIn() const {
    return weights_.cols();
}
Index Layer::GetOut() const {
    return weights_.rows();
}
const Matrix& Layer::GetWeights() const {
    return weights_;
}
const Vector& Layer::GetBias() const {
    return bias_;
}

Vector Layer::Compute(const Vector& x) const {
    return activation_function_->Compute(weights_ * x + bias_);
}
Matrix Layer::GetActFuncJacobian(const Vector& x) const {
    return activation_function_->ComputeGradient(x);
}
void Layer::Update(const Matrix& weights_delta, const Vector& bias_delta) {
    weights_ += weights_delta;
    bias_ += bias_delta;
}

Matrix Layer::SafeGen(Index in, Index out, Random<>& rnd) {
    assert(in > 0 && out > 0 && "Layer dimensions must be positive numbers");
    return rnd.GenMatrix(out, in);
}
Vector Layer::SafeGen(Index out, Random<>& rnd) {
    assert(in > 0 && out > 0 && "Layer dimensions must be positive numbers");
    return rnd.GenVector(out);
}

}  // namespace NeuralNetwork
