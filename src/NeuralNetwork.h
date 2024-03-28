#pragma once

#include <vector>

#include "Layer.h"
#include "LossFunctions.h"
#include "Optimizers.h"
#include "GlobalUsings.h"
#include "Utils.h"

namespace NeuralNetwork {

class NeuralNetwork {
    std::vector<Layer> layers_;

public:
    NeuralNetwork(std::initializer_list<Layer> layers) : layers_(layers);

    void Fit(const DataLoader& train_data_loader, const Optimizer& optimizer,
             const LossFunction& loss_function, int32_t epochs);
    Vector Predict(const Vector& x);
    VectorSet Predict(const VectorSet& X);

private:
    VectorSet ComputeOuts(const Vector& x);
    void BackPropogate(const Batch& batch, const Optimizer& optimizer, const LossFunction& loss_function);
};

}  // namespace NeuralNetwork
