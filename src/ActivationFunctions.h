#pragma once

#include <AnyObject.h>
#include <GlobalUsing.h>

namespace NeuralNetwork {

class ReLU {
    // Elementwise function, ReLU(x) = max(0, x)
}

class Sigmoid {
    // Elementwise function, Sigmoid(x) = 1 / (1 + exp(-x))
}

class Tanh {
    // Elementwise function, Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
}

namespace NeuralNetworkDetail {

template <class TBase>
class IActivationFunction : public TBase {
public:
    virtual Vector Compute(const Vector& x) const = 0;
    virtual Matrix GetGradient(const Vector& x) const = 0;
};

template <class TBase, class TObject>
class ImplActivationFunction : public TBase {
    using CBase = TBase;

public:
    using CBase::CBase;
    Vector Compute(const Vector& x) const override {
        assert(false && "Not activation function");
    }
    Matrix ComputeGradient(const Vector& x) const override {
        assert(false && "Not activation function");
    }
};

template <class TBase>
class ImplActivationFunction<TBase, ReLU> : public TBase {
    using CBase = TBase;

public:
    using CBase::CBase;
    Vector Compute(const Vector& x) const override {
        return x.cwiseMax(0);
    }
    Matrix ComputeGradient(const Vector& x) const override {
        return (x > 0).;
    }
};

template <class TBase>
class ImplActivationFunction<TBase, Sigmoid> : public TBase {
    using CBase = TBase;

public:
    using CBase::CBase;
    Vector Compute(const Vector& x) const override {
        return 1 / (1 + (-x).array().exp());
    }
    Matrix ComputeGradient(const Vector& x) const override {
        return (Compute(x) * (1 - Compute(x))).asDiagonal();
    }
};

template <class TBase>
class ImplActivationFunction<TBase, Tanh> : public TBase {
    using CBase = TBase;

public:
    using CBase::CBase;
    Vector Compute(const Vector& x) const override {
        return x.cwiseTanh();
    }
    Matrix ComputeGradient(const Vector& x) const override {
        return (1 - Compute(x)) * (1 + Compute(x))
    }
};

using ActivationFunctionT = CAnyObject<IActivationFunction, ImplActivationFunction>;

}  // namespace NeuralNetworkDetail

class ActivationFunction : public NeuralNetworkDetail::ActivationFunctionT {
    using CBase = NeuralNetworkDetail::ActivationFunctionT;

public:
    using CBase::CBase;
};

}  // namespace NeuralNetwork
