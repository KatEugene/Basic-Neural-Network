#include <NeuralNetwork.h>
#include <iostream>

namespace NeuralNetwork {

Net::Net(std::initializer_list<Layer> layers) : layers_(layers) {
    for (int32_t i = 0; i + 1 < std::ssize(layers_); ++i) {
        assert(layers_[i].GetOut() == layers_[i + 1].GetIn() &&
               "Out and in dimensions of adjacent layers are not equal");
    }
}

void Net::Fit(const DataLoader& train_data_loader, const Optimizer& optimizer,
              const LossFunction& loss_function, int32_t epochs, const VectorSet& X_train,
              const VectorSet& y_train) {
    assert(loss_function.isDefined() && "Loss function is not defined");
    assert(optimizer.isDefined() && "Optimizer is not defined");
    assert(epochs > 0 && "Number of epochs must be positive number");

    for (int32_t i = 0; i < epochs; ++i) {
        for (const Batch& batch : train_data_loader) {
            BackPropogate(batch, optimizer, loss_function);
        }
    }
}
Vector Net::Predict(const Vector& x) const {
    Vector pred = x;
    for (int32_t i = 0; i < std::ssize(layers_); ++i) {
        pred = layers_[i].Compute(pred);
    }
    return pred;
}
VectorSet Net::Predict(const VectorSet& X) const {
    VectorSet preds;
    for (const Vector& x : X) {
        preds.push_back(Predict(x));
    }
    return preds;
}

VectorSet Net::ComputeOuts(const Vector& x) const {
    VectorSet outs = {x};

    for (int32_t i = 0; i < std::ssize(layers_); ++i) {
        outs.push_back(layers_[i].Compute(outs.back()));
    }

    return outs;
}

void Net::BackPropogate(const Batch& batch, const Optimizer& optimizer,
                        const LossFunction& loss_function) {
    int32_t batch_size = batch.GetSize();
    int32_t layers_size = std::ssize(layers_);

    std::vector<Matrix> weights_grads(layers_size);
    VectorSet bias_grads(layers_size);

    for (int32_t i = 0; i < layers_size; ++i) {
        weights_grads[i] = Matrix::Zero(layers_[i].GetOut(), layers_[i].GetIn());
        bias_grads[i] = Vector::Zero(layers_[i].GetOut());
    }

    for (const auto& [x, y] : batch) {
        VectorSet layer_outs = ComputeOuts(x);
        RowVector last_grad = loss_function->ComputeGradient(layer_outs.back(), y);

        for (int32_t i = layers_size - 1; i >= 0; --i) {
            Matrix jacobian = layers_[i].GetActFuncJacobian(layer_outs[i + 1]);

            Matrix weights_grad = (layer_outs[i] * last_grad * jacobian).transpose();
            Vector bias_grad = (last_grad * jacobian).transpose();

            last_grad *= jacobian * layers_[i].GetWeights();

            weights_grads[i] += weights_grad / batch_size;
            bias_grads[i] += bias_grad / batch_size;
        }
    }
    optimizer->DoStep(&layers_, weights_grads, bias_grads);
}

}  // namespace NeuralNetwork
