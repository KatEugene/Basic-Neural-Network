#include "ActivationFunctions.h"

namespace NeuralNetwork {

Vector Id::Compute(const Vector& x) const {
    return x;
}
Matrix Id::ComputeGradient(const Vector& x) const {
    return Matrix::Identity(x.size(), x.size());
}

Vector ReLU::Compute(const Vector& x) const {
    return x.cwiseMax(0);
}
Matrix ReLU::ComputeGradient(const Vector& x) const {
    return Matrix((x.array() > 0).cast<DataType>().matrix().asDiagonal());
}

Vector Sigmoid::Compute(const Vector& x) const {
    return x.unaryExpr([](DataType xi) { return 1 / (1 + exp(-xi)); });
}
Matrix Sigmoid::ComputeGradient(const Vector& x) const {
    return (Compute(x).cwiseProduct(Vector::Ones(x.size()) - Compute(x))).asDiagonal();
}

Vector Tanh::Compute(const Vector& x) const {
    return x.unaryExpr([](DataType xi) { return tanh(xi); });
}
Matrix Tanh::ComputeGradient(const Vector& x) const {
    Vector l_fact = Vector::Ones(x.size()) - Compute(x);
    Vector r_fact = Vector::Ones(x.size()) + Compute(x);
    return (l_fact.cwiseProduct(r_fact)).asDiagonal();
}

Vector Softmax::Compute(const Vector& x) const {
    Vector exps = (x.array() - x.maxCoeff()).exp();
    return exps / exps.sum();
}
Matrix Softmax::ComputeGradient(const Vector& x) const {
    Vector smax = Compute(x);
    return Matrix(smax.asDiagonal()) - smax * smax.transpose();
}

}  // namespace NeuralNetwork
