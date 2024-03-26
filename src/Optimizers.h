#pragma once

#include <AnyObject.h>
#include <GlobalUsing.h>

namespace NeuralNetwork {

class StochasticGradientDescent {

public:
	Vector ComputeGradient() {

	}
	void Step(const Batch& batch) {
		for (auto x : batch) {
			for (int i = layers.ssize(); i >= 0; --i) {

			}
			Vector grad = ComputeGradient();

		}
	}
private:

};

} // namespace NeuralNetwork
