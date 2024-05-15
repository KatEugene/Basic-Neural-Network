#pragma once

#include <vector>

#include "AnyObject.h"
#include "GlobalUsings.h"

namespace NeuralNetwork {

namespace NNLossFuncDetail {

template <class Base>
class InterfaceLF : public Base {
public:
    virtual DataType Compute(const Vector& predicted, const Vector& expected) const = 0;
    virtual DataType Compute(const VectorSet& predicted, const VectorSet& expected) const = 0;
    virtual Vector ComputeGradient(const Vector& predicted, const Vector& expected) const = 0;
};

template <class Base, class Type>
class ImplLossFunction : public Base {
public:
    using Base::Base;

    DataType Compute(const Vector& predicted, const Vector& expected) const override {
        return Base::Object().Compute(predicted, expected);
    }
    DataType Compute(const VectorSet& predicted, const VectorSet& expected) const override {
        return Base::Object().Compute(predicted, expected);
    }
    Vector ComputeGradient(const Vector& predicted, const Vector& expected) const override {
        return Base::Object().ComputeGradient(predicted, expected);
    }
};

using LossFunctionT = CAnyObject<InterfaceLF, ImplLossFunction>;

}  // namespace NNLossFuncDetail

class LossFunction : public NNLossFuncDetail::LossFunctionT {
    using Base = NNLossFuncDetail::LossFunctionT;

public:
    using Base::Base;
};

class MSE {
    // Given vectors x, y in R^n, computes sum((x_i - y_i)^2)

public:
    DataType Compute(const Vector& predicted, const Vector& expected) const;
    DataType Compute(const VectorSet& predicted, const VectorSet& expected) const;
    Vector ComputeGradient(const Vector& predicted, const Vector& expected) const;
};

class MAE {
    // Given vectors x, y in R^n, computes sum(abs(x_i - y_i))

public:
    DataType Compute(const Vector& predicted, const Vector& expected) const;
    DataType Compute(const VectorSet& predicted, const VectorSet& expected) const;
    Vector ComputeGradient(const Vector& predicted, const Vector& expected) const;
};

}  // namespace NeuralNetwork
