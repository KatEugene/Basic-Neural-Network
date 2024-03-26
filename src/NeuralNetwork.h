#pragma once

#include <vector>

#include <Layer.h>
#include <LossFunctions.h>
#include <Optimizers.h>
#include <GlobalUsing.h>

namespace NeuralNetwork {

class NeuralNetwork {
	std::vector<Layer> layers_;

public:
	NeuralNetwork(std::initializer_list<Layer> layers) : layers_(layers) {
	}

	void Fit() {

	}
	Vector Predict() {

	}

private:
		

};

} // namespace NeuralNetwork
