#pragma once

#include "ActivationFunctions.h"
#include "GlobalUsings.h"
#include "Utils.h"

namespace NeuralNetwork {

class Layer {
    Matrix weights_;
    Vector bias_;
    ActivationFunction activation_function_;

public:
    Layer(Index in, Index out, ActivationFunction activation_function = Id(),
          Random<> rnd = Random(42));
    Index GetIn() const;
    Index GetOut() const;
    const Matrix& GetWeights() const;
    const Vector& GetBias() const;

    Vector Compute(const Vector& x) const;
    Matrix GetActFuncJacobian(const Vector& x) const;
    void Update(const Matrix& weights_delta, const Vector& bias_delta);

private:
    static Matrix SafeGen(Index in, Index out, Random<>& rnd);
    static Vector SafeGen(Index out, Random<>& rnd);
};

}  // namespace NeuralNetwork
