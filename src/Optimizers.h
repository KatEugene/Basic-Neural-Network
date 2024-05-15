#pragma once

#include "AnyObject.h"
#include "GlobalUsings.h"
#include "Layer.h"

namespace NeuralNetwork {

namespace NNOptimizerDetail {

template <class Base>
class InterfaceOpt : public Base {
public:
    virtual void DoStep(const std::vector<Matrix>& weights_grads,
                        const std::vector<Vector>& bias_grads, std::vector<Layer>* layers) = 0;
};

template <class Base, class Type>
class ImplOptimizer : public Base {
public:
    using Base::Base;

    void DoStep(const std::vector<Matrix>& weights_grads, const std::vector<Vector>& bias_grads,
                std::vector<Layer>* layers) override {
        Base::Object().DoStep(weights_grads, bias_grads, layers);
    }
};

using OptimizerT = CAnyObject<InterfaceOpt, ImplOptimizer>;

}  // namespace NNOptimizerDetail

class Optimizer : public NNOptimizerDetail::OptimizerT {
    using Base = NNOptimizerDetail::OptimizerT;

public:
    using Base::Base;
};

class SGD {
    // Stochastic gradient descent

    static constexpr DataType k_default_learning_rate = 4e-2;
    static constexpr DataType k_default_momentum = 0;
    DataType learning_rate_;
    DataType momentum_;

    std::vector<Matrix> weights_deltas_;
    std::vector<Vector> bias_deltas_;

public:
    SGD(DataType learning_rate = k_default_learning_rate, DataType momentum = k_default_momentum);
    void DoStep(const std::vector<Matrix>& weights_grads, const std::vector<Vector>& bias_grads,
                std::vector<Layer>* layers);

private:
    void InitDeltas(const std::vector<Matrix>& weights, const std::vector<Vector>& bias);
};

// class Adagrad {
//     // Stochastic gradient descent

//     static constexpr DataType k_default_learning_rate = 4e-2;
//     static constexpr DataType k_default_epsilon = 1e-15;
//     DataType learning_rate_;
//     DataType epsilon_;

//     std::vector<Matrix> weights_deltas_;
//     std::vector<Vector> bias_deltas_;

// public:
//     SGD(DataType learning_rate = k_default_learning_rate, epsilon_ = k_default_epsilon);
//     void DoStep(const std::vector<Matrix>& weights_grads, const std::vector<Vector>& bias_grads,
//                 std::vector<Layer>* layers);
// };

}  // namespace NeuralNetwork
