#pragma once

#include <ActivationFunctions.h>
#include <GlobalUsing.h>

namespace NeuralNetwork {

class Layer {
	Matrix weights_;
	Vector bias_;
	ActivationFunction activation_function_;

public:
	Layer(size_t in, size_t out, ActivationFunction activation_function) : activation_function(activation_function_) {
		weights_.reshape(out, in);
		bias_.reshape(out);
	}

	Vector Compute(const Vector& x) {
		return activation_function_.Compute(weights_ * x + bias_);
	}
};

} // namespace NeuralNetwork
