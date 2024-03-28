#pragma once

#include "ActivationFunctions.h"
#include "GlobalUsings.h"

namespace NeuralNetwork {

class Layer {
    Matrix weights_;
    Vector bias_;
    ActivationFunction activation_function_;

public:
    template <typename RandGen, typename RandNumGen>
    Layer(Index in, Index out, ActivationFunction activation_function = Id()
          RandGen rand_gen = Eigen::Rand::NormalGen<DataType>{}, RandNumGen urng = Eigen::Rand::Vmt19937_64{42})
        : activation_function_(activation_function) {
        assert(activation_function_.isDefined() && "Activation function is not defined");
        assert(in > 0 && out > 0 && "Layer dimensions must be positive numbers");

        weights_.resize(out, in);
        bias_.resize(out);

        weights_ = rand_gen.generateLike(weights_, urng);
        bias_ = rand_gen.generateLike(bias_, urng);
    }
    Index GetIn() const {
        return weights_.cols();
    }
    Index GetOut() const {
        return weights_.rows();
    }
    Matrix GetWeights() const {
        return weights_;
    }
    Vector GetBias() const {
        return bias_;
    }
    Matrix& GetWeightsReference() {
        return weights_;
    }
    Vector& GetBiasReference() {
        return bias_;
    }

    Vector Compute(const Vector& x) const {
        return activation_function_->Compute(weights_ * x + bias_);
    }
    Matrix GetActFuncJacobian(const Vector& x) const {
        return activation_function_->ComputeGradient(x);
    }
};

}  // namespace NeuralNetwork
