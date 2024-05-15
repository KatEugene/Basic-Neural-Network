#pragma once

#include <vector>
#include <span>

#include "Layer.h"
#include "LossFunctions.h"
#include "Optimizers.h"
#include "GlobalUsings.h"
#include "Utils.h"
#include "DataLoader.h"
#include "Random.h"

namespace NeuralNetwork {

enum class TrainingInfo { On, Off };

class Net {
    std::vector<Layer> layers_;

public:
    Net(std::span<Layer> layers);

    void Fit(const DataLoader& train_data_loader, const Optimizer& optimizer,
             const LossFunction& loss_function, SizeType epochs,
             TrainingInfo info = TrainingInfo::Off, const VectorSet& X_test = {},
             const VectorSet& y_test = {}, SizeType each_epoch = 1);
    Vector Predict(const Vector& x) const;
    VectorSet Predict(const VectorSet& X) const;
    void SaveWeights(std::fstream& out);

private:
    VectorSet ForwardPass(const Vector& x) const;
    void BackPropogate(SizeType batch_size, SizeType layers_size, const VectorSet& layer_outs,
                       const Vector& y, const Optimizer& optimizer,
                       const LossFunction& loss_function, MatrixSet* weights_grads_p,
                       VectorSet* bias_grads_p);
    void FitBatch(const Batch& batch, const Optimizer& optimizer,
                  const LossFunction& loss_function);
    MatrixSet InitZeroWeights();
    VectorSet InitZeroBias();
};

}  // namespace NeuralNetwork
