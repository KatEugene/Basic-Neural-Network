#include "LossFunctions.h"

namespace NeuralNetwork {

DataType MSE::Compute(const Vector& predicted, const Vector& expected) const {
    return (predicted - expected).squaredNorm();
}
DataType MSE::Compute(const VectorSet& predicted, const VectorSet& expected) const {
    DataType sum = 0;
    for (size_t i = 0; i < predicted.size(); ++i) {
        sum += Compute(predicted[i], expected[i]);
    }
    return sum / predicted.size();
}
Vector MSE::ComputeGradient(const Vector& predicted, const Vector& expected) const {
    return 2 * (predicted - expected);
}

DataType MAE::Compute(const Vector& predicted, const Vector& expected) const {
    return (predicted - expected).cwiseAbs().sum();
}
DataType MAE::Compute(const VectorSet& predicted, const VectorSet& expected) const {
    DataType sum = 0;
    for (size_t i = 0; i < predicted.size(); ++i) {
        sum += Compute(predicted[i], expected[i]);
    }
    return sum / predicted.size();
}
Vector MAE::ComputeGradient(const Vector& predicted, const Vector& expected) const {
    return Vector((predicted - expected).array().sign());
}

}  // namespace NeuralNetwork