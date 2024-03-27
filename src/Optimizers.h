#pragma once

#include <AnyObject.h>
#include <GlobalUsing.h>
#include <Layer.h>

namespace NeuralNetwork {

class SGD {
    // Stochastic gradient descent
	
	DataType learning_rate_;

public:
    SGD(DataType learning_rate = 1e-2) : learning_rate_(learning_rate) {
    }
    DataType GetLearningRate() const {
    	return learning_rate_;
    }
};

namespace NNOptimizerDetail {

template <class TBase>
class IOptimizer : public TBase {
public:
    virtual void DoStep(std::vector<Layer>* layers, const std::vector<Matrix>& weights_grads,
                        const std::vector<Vector>& bias_grads) const = 0;
};

template <class TBase, class TObject>
class ImplOptimizer : public TBase {
    using CBase = TBase;

public:
    using CBase::CBase;

    void DoStep(std::vector<Layer>* layers, const std::vector<Matrix>& weights_grads,
                        const std::vector<Vector>& bias_grads) const override {
        assert(false && "Not optimizer");
    }
};

template <class TBase>
class ImplOptimizer<TBase, SGD> : public TBase {
    using CBase = TBase;

public:
    using CBase::CBase;

    void DoStep(std::vector<Layer>* layers, const std::vector<Matrix>& weights_grads,
                        const std::vector<Vector>& bias_grads) const override {
    	DataType learning_rate = Cbase::Object().GetLearningRate();

    	for (size_t i = 0; i < layers->size(); ++i) {
    		layers->at(i).GetWeightsReference() -= learning_rate * weights_grads[i];
    		layers->at(i).GetBiasReference() -= learning_rate * bias_grads[i];

    		/*
			Matrix updated_weights = layers[i].GetWeights();
    		Vector updated_bias = layers[i].GetBias();

    		updated_weights -= learning_rate_ * weights_grads[i];
    		updated_bias -= learning_rate_ * bias_grads[i];

    		layers[i].SetWeights(updated_weights);
    		layers[i].SetBias(updated_bias);
    		*/
    	}
    }
};

using OptimizerT = CAnyObject<IOptimizer, ImplOptimizer>;

}  // namespace NNOptimizerDetail

class Optimizer : public NNOptimizerDetail::OptimizerT {
    using CBase = NNOptimizerDetail::OptimizerT;

public:
    using CBase::CBase;
};

}  // namespace NeuralNetwork
