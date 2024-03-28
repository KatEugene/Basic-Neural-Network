#pragma once

#include <Eigen/Eigen>
#include <EigenRand/EigenRand>

namespace NeuralNetwork {

using Vector = Eigen::VectorXd;
using RowVector = Eigen::RowVectorXd;
using Matrix = Eigen::MatrixXd;
using Index = Eigen::Index;

using DataType = double;
using SizeType = int32_t;

using VectorSet = std::vector<Vector>;

}  // namespace NeuralNetwork
