#include <NeuralNetwork.h>
#include <iostream>
namespace NeuralNetwork {

Net::Net(std::span<Layer> layers) {
    assert(!layers.empty() && "Can't construct empty neural network");
    for (SizeType i = 0; i + 1 < std::ssize(layers); ++i) {
        layers_.push_back(std::move(layers[i]));
        assert(layers[i].GetOut() == layers[i + 1].GetIn() &&
               "Out and in dimensions of adjacent layers are not equal");
    }
    layers_.push_back(std::move(layers.back()));
}

void Net::Fit(const DataLoader& train_data_loader, const Optimizer& optimizer,
              const LossFunction& loss_function, SizeType epochs, TrainingInfo info,
              const VectorSet& X_test, const VectorSet& y_test, SizeType each_epoch) {
    assert(loss_function.isDefined() && "Loss function is not defined");
    assert(optimizer.isDefined() && "Optimizer is not defined");
    assert(epochs > 0 && "Number of epochs must be positive number");

    for (SizeType i = 0; i < epochs; ++i) {
        for (const Batch& batch : train_data_loader) {
            FitBatch(batch, optimizer, loss_function);
        }
        if (info == TrainingInfo::On && (i + 1) % each_epoch == 0) {
            VectorSet preds = Predict(X_test);
            std::cout << "Epoch: " << i + 1 << ", "
                      << "Loss: " << loss_function->Compute(preds, y_test) << '\n';
        }
    }
}
Vector Net::Predict(const Vector& x) const {
    Vector pred = x;
    for (SizeType i = 0; i < std::ssize(layers_); ++i) {
        pred = layers_[i].Compute(pred);
    }
    return pred;
}
VectorSet Net::Predict(const VectorSet& X) const {
    VectorSet preds;
    preds.reserve(X.size());
    for (const Vector& x : X) {
        preds.push_back(Predict(x));
    }
    return preds;
}
void Net::SaveWeights(std::fstream& out) {
    for (const auto& layer : layers_) {
        out << layer.GetWeights() << layer.GetBias();
    }
}

VectorSet Net::ForwardPass(const Vector& x) const {
    SizeType layers_size = std::size(layers_);

    VectorSet outs = {x};
    outs.reserve(layers_size);

    for (SizeType i = 0; i < layers_size; ++i) {
        outs.push_back(layers_[i].Compute(outs.back()));
    }

    return outs;
}
void Net::BackPropogate(SizeType batch_size, SizeType layers_size, const VectorSet& layer_outs,
                        const Vector& y, const Optimizer& optimizer,
                        const LossFunction& loss_function, MatrixSet* weights_grads_p,
                        VectorSet* bias_grads_p) {
    MatrixSet& weights_grads = *weights_grads_p;
    VectorSet& bias_grads = *bias_grads_p;
    RowVector last_grad = loss_function->ComputeGradient(layer_outs.back(), y);

    for (SizeType i = layers_size - 1; i >= 0; --i) {
        Matrix jacobian = layers_[i].GetActFuncJacobian(layer_outs[i + 1]);

        Matrix weights_grad = (layer_outs[i] * last_grad * jacobian).transpose();
        Vector bias_grad = (last_grad * jacobian).transpose();

        last_grad *= jacobian * layers_[i].GetWeights();

        weights_grads[i] += weights_grad / batch_size;
        bias_grads[i] += bias_grad / batch_size;
    }
}
void Net::FitBatch(const Batch& batch, const Optimizer& optimizer,
                   const LossFunction& loss_function) {
    SizeType batch_size = batch.GetSize();
    SizeType layers_size = std::ssize(layers_);

    MatrixSet weights_grads = InitZeroWeights();
    VectorSet bias_grads = InitZeroBias();
    for (const auto& [x, y] : batch) {
        VectorSet layer_outs = ForwardPass(x);
        BackPropogate(batch_size, layers_size, layer_outs, y, optimizer, loss_function,
                      &weights_grads, &bias_grads);
    }
    optimizer->DoStep(weights_grads, bias_grads, &layers_);
}

MatrixSet Net::InitZeroWeights() {
    SizeType layers_size = std::ssize(layers_);
    MatrixSet weights(layers_size);
    for (SizeType i = 0; i < layers_size; ++i) {
        weights[i] = Matrix::Zero(layers_[i].GetOut(), layers_[i].GetIn());
    }
    return weights;
}

VectorSet Net::InitZeroBias() {
    SizeType layers_size = std::ssize(layers_);
    VectorSet bias(layers_size);
    for (SizeType i = 0; i < layers_size; ++i) {
        bias[i] = Vector::Zero(layers_[i].GetOut());
    }
    return bias;
}

}  // namespace NeuralNetwork
