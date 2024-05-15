#pragma once

#include "AnyObject.h"
#include "GlobalUsings.h"

namespace NeuralNetwork {

namespace NNActivationFuncDetail {

template <class Base>
class InterfaceAF : public Base {
public:
    virtual Vector Compute(const Vector& x) const = 0;
    virtual Matrix ComputeGradient(const Vector& x) const = 0;
};

template <class Base, class Type>
class AFImpl : public Base {
public:
    using Base::Base;
    Vector Compute(const Vector& x) const override {
        return Base::Object().Compute(x);
    }
    Matrix ComputeGradient(const Vector& x) const override {
        return Base::Object().ComputeGradient(x);
    }
};

using ActivationFunctionT = CAnyObject<InterfaceAF, AFImpl>;

}  // namespace NNActivationFuncDetail

class ActivationFunction : public NNActivationFuncDetail::ActivationFunctionT {
    using Base = NNActivationFuncDetail::ActivationFunctionT;

public:
    using Base::Base;
};

class Id {
    // Identical function, Id(x) = x
public:
    Vector Compute(const Vector& x) const;
    Matrix ComputeGradient(const Vector& x) const;
};

class ReLU {
    // Elementwise function, ReLU(x) = max(0, x)
public:
    Vector Compute(const Vector& x) const;
    Matrix ComputeGradient(const Vector& x) const;
};

class Sigmoid {
    // Elementwise function, Sigmoid(x) = 1 / (1 + exp(-x))
public:
    Vector Compute(const Vector& x) const;
    Matrix ComputeGradient(const Vector& x) const;
};

class Tanh {
    // Elementwise function, Tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
public:
    Vector Compute(const Vector& x) const;
    Matrix ComputeGradient(const Vector& x) const;
};

class Softmax {
    // Softmax(x) = {exp(x_i) / sum(exp(x_i))}
public:
    Vector Compute(const Vector& x) const;
    Matrix ComputeGradient(const Vector& x) const;
};

}  // namespace NeuralNetwork
