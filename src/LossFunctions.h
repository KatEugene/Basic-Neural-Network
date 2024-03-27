#pragma once

#include <vector>
#include <AnyObject.h>
#include <GlobalUsing.h>

namespace NeuralNetwork {

class MSE {
    // Given vectors x, y in R^n, computes their squared 2nd norm : ||x - y||^2
};

namespace NNLossFuncDetail {

template <class TBase>
class ILossFunction : public TBase {
public:
    virtual DataType Compute(const Vector& predicted, const Vector& expected) const = 0;
    virtual DataType Compute(const VectorSet& predicted, const VectorSet& expected) const = 0;
    virtual Vector ComputeGradient(const Vector& predicted, const Vector& expected) const = 0;
};

template <class TBase, class TObject>
class ImplLossFunction : public TBase {
    using CBase = TBase;

public:
    using CBase::CBase;

    DataType Compute(const Vector& predicted, const Vector& expected) const override {
        assert(false && "Not loss function");
    }
    DataType Compute(const VectorSet& predicted, const VectorSet& expected) const override {
        assert(false && "Not loss function");
    }
    Vector ComputeGradient(const Vector& predicted, const Vector& expected) const override {
        assert(false && "Not loss function");
    }
};

template <class TBase>
class ImplLossFunction<TBase, MSE> : public TBase {
    using CBase = TBase;

public:
    using CBase::CBase;

    DataType Compute(const Vector& predicted, const Vector& expected) const override {
        return (predicted - expected).squaredNorm();
    }
    DataType Compute(const VectorSet& predicted, const VectorSet& expected) const override {
        DataType sum = 0;
        for (size_t i = 0; i < predicted.size(); ++i) {
            sum += Compute(predicted[i], expected[i]);
        }
        return sum / predicted.size();
    }
    Vector ComputeGradient(const Vector& predicted, const Vector& expected) const override {
        return 2 * (predicted - expected);
    }
};

using LossFunctionT = CAnyObject<ILossFunction, ImplLossFunction>;

}  // namespace NNLossFuncDetail

class LossFunction : public NNLossFuncDetail::LossFunctionT {
    using CBase = NNLossFuncDetail::LossFunctionT;

public:
    using CBase::CBase;
};

}  // namespace NeuralNetwork
