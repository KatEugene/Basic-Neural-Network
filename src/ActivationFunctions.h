#pragma once

#include "AnyObject.h"
#include "GlobalUsings.h"

namespace NeuralNetwork {

class Id {
    // Identical function, Id(x) = x
};

class ReLU {
    // Elementwise function, ReLU(x) = max(0, x)
};

class Sigmoid {
    // Elementwise function, Sigmoid(x) = 1 / (1 + exp(-x))
};

class Tanh {
    // Elementwise function, Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
};

namespace NNActivationFuncDetail {

template <class TBase>
class IActivationFunction : public TBase {
public:
    virtual Vector Compute(const Vector& x) const = 0;
    virtual Matrix ComputeGradient(const Vector& x) const = 0;
};

template <class TBase, class TObject>
class ImplActivationFunction : public TBase {
    using CBase = TBase;

public:
    using CBase::CBase;
    Vector Compute(const Vector& x) const override {
        // это очень костыльно, но лучше я не придумал
        assert(false && "Not activation function");
    }
    Matrix ComputeGradient(const Vector& x) const override {
        assert(false && "Not activation function");
    }
};

template <class TBase>
class ImplActivationFunction<TBase, Id> : public TBase {
    using CBase = TBase;

public:
    using CBase::CBase;
    Vector Compute(const Vector& x) const override {
        return x;
    }
    Matrix ComputeGradient(const Vector& x) const override {
        return Matrix::Identity(x.size(), x.size());
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
        return Matrix((x.array() > 0).matrix().asDiagonal());
    }
};

template <class TBase>
class ImplActivationFunction<TBase, Sigmoid> : public TBase {
    using CBase = TBase;

public:
    using CBase::CBase;
    Vector Compute(const Vector& x) const override {
        return Vector(1 / (1 + (-x).array().exp()));
    }
    Matrix ComputeGradient(const Vector& x) const override {
        return (Compute(x).cwiseProduct(1 - Compute(x))).asDiagonal();
    }
};

template <class TBase>
class ImplActivationFunction<TBase, Tanh> : public TBase {
    using CBase = TBase;

public:
    using CBase::CBase;
    Vector Compute(const Vector& x) const override {
        return Vector(x.array().tanh());
    }
    Matrix ComputeGradient(const Vector& x) const override {
        return ((1 - Compute(x)).cwiseProduct(1 + Compute(x))).asDiagonal();
    }
};

using ActivationFunctionT = CAnyObject<IActivationFunction, ImplActivationFunction>;

}  // namespace NNActivationFuncDetail

class ActivationFunction : public NNActivationFuncDetail::ActivationFunctionT {
    using CBase = NNActivationFuncDetail::ActivationFunctionT;

public:
    using CBase::CBase;
};

}  // namespace NeuralNetwork
