#pragma once

#include <AnyObject.h>
#include <GlobalUsing.h>

namespace NeuralNetwork {

class MSE {
	// Given vectors x, y in R^n, computes their squared 2nd norm : ||x - y||^2
};

namespace NeuralNetworkDetail {

template<class TBase>
class ILossFunction : public TBase {
public:
    virtual double Compute(const Vector& predicted, const Vector& expected) const = 0;
    virtual Vector ComputeGradient(const Vector& predicted, const Vector& expected) const = 0;
};

template<class TBase, class TObject>
class ImplLossFunction : public TBase {
  using CBase = TBase;

public:
  using CBase::CBase;

  double Compute(const Vector& predicted, const Vector& expected) const override {
    assert(false && "Not loss function");
  }
  Vector ComputeGradient(const Vector& predicted, const Vector& expected) const override {
    assert(false && "Not loss function");
  }
};

template<class TBase>
class ImplLossFunction<TBase, MSE> : public TBase {
  using CBase = TBase;

public:
  using CBase::CBase;

  double Compute(const Vector& predicted, const Vector& expected) const override {
    return (predicted - expected).squaredNorm();
  }
  Vector ComputeGradient(const Vector& predicted, const Vector& expected) const override {
    return 2 * (predicted - expected);
  }
};

using LossFunctionT = CAnyObject<ILossFunction, ImplLossFunction>;

} // namespace NeuralNetworkDetail

class LossFunction : public NeuralNetworkDetail::LossFunctionT {
  using CBase = NeuralNetworkDetail::LossFunctionT;
public:
  using CBase::CBase;
};

} // namespace NeuralNetwork
