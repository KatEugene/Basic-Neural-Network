#pragma once

#include <vector>

#include <Layer.h>
#include <LossFunctions.h>
#include <Optimizers.h>
#include <GlobalUsings.h>
#include <Utils.h>

namespace NeuralNetwork {

class Net {
    std::vector<Layer> layers_;

public:
    Net(std::initializer_list<Layer> layers);

    void Fit(const DataLoader& train_data_loader, const Optimizer& optimizer,
             const LossFunction& loss_function, int32_t epochs, const VectorSet& X_train,
             const VectorSet& y_train);
    Vector Predict(const Vector& x) const;
    VectorSet Predict(const VectorSet& X) const;

private:
    VectorSet ComputeOuts(const Vector& x) const;
    void BackPropogate(const Batch& batch, const Optimizer& optimizer,
                       const LossFunction& loss_function);
};

}  // namespace NeuralNetwork
