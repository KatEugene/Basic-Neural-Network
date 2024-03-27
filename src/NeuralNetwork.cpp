// #include <NeuralNetwork.h>

// namespace NeuralNetwork {

// NeuralNetwork::NeuralNetwork(std::initializer_list<Layer> layers) : layers_(layers) {
//     for (int32_t i = 0; i + 1 < layers_.ssize(); ++i) {
//         assert(layers_[i].GetOut() == layers_[i + 1].GetIn() &&
//                "Out and in dimensions of adjacent layers are not equal");
//     }
// }

// void NeuralNetwork::Fit(const DataLoader& train_data_loader, const Optimizer& optimizer,
//          const LossFunction& loss_function, int32_t epochs) {
//     assert(loss_function.IsDefined() && "Loss function is not defined");
//     assert(optimizer.IsDefined() && "Optimizer is not defined");
//     assert(epochs > 0 && "Number of epochs must be positive number");

//     for (int32_t i = 0; i < epochs; ++i) {
//         for (const auto& batch : train_data_loader) {
//             BackPropogate(batch, optimizer, loss_function);
//         }
//     }
// }
// Vector NeuralNetwork::Predict(const Vector& x) {
// 	Vector pred = x;
// 	for (int32_t i = 0; i < layers_.ssize(); ++i) {
// 		pred = layers_[i].Compute(pred);
// 	}
//     return pred;
// }
// VectorSet NeuralNetwork::Predict(const VectorSet& X) {
//     VectorSet preds;
//     for (const auto& x : X) {
//     	preds.push_back(Predict(x));
//     }
//     return preds;
// }

// VectorSet NeuralNetwork::ComputeOuts(const Vector& x) {
//     VectorSet outs = {x};

//     for (int32_t i = 0; i < layers_.size(); ++i) {
//         outs.push_back(layers_[i].Compute(outs.back()));
//     }

//     return out;
// }

// void NeuralNetwork::BackPropogate(const Batch& batch, const Optimizer& optimizer, const LossFunction& loss_function) {
//     int32_t batch_size = batch.X.ssize();
//     int32_t layers_size = layers_.ssize();

//     std::vector<Matrix> weights_grad(layers_size);
//     VectorSet bias_grads(layers_size);

//     for (int32_t i = 0; i < layers_size; ++i) {
//         weights_grads[i].resize(layers_[i].GetOut(), layers_[i].GetIn());
//         bias_grads[i].resize(layers_[i].GetOut());
//     }

//     for (int32_t i = 0; i < batch_size; ++i) {
//     	Vector x = batch.X[i];
//     	Vector y = batch.y[i];

//         VectorSet layer_outs = ComputeOuts(x);
//         RowVector last_grad = loss_function.ComputeGradient(layer_outs.back(), y);

//         for (int32_t i = layers_size - 1; i >= 1; --i) {
//             Matrix jacobian = layers_[i].GetActFuncJacobian(layer_outs[i]);

//             Matrix weights_grad = (layer_outs[i - 1] * last_grad * jacobian).transpose();
//             Vector bias_grad = (layer_outs[i - 1] * jacobian).transpose();
//             last_grad = last_grad * jacobian * layers_.GetWeights();

//             weights_grads[i] += weights_grad / batch_size;
//             bias_grads[i] += bias_grad / batch_size;
//         }
//     }

//     optimizer.DoStep(&layers_, weights_grads, bias_grads);
// }

// }  // namespace NeuralNetwork
