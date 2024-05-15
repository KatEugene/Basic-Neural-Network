include_directories("${CMAKE_SOURCE_DIR}/libs/Eigen")
include_directories("${CMAKE_SOURCE_DIR}/libs/EigenRand")

add_library(
	neural_network 
	src/NeuralNetwork.cpp
	src/Utils.cpp
	src/Layer.cpp
	src/ActivationFunctions.cpp
	src/LossFunctions.cpp
	src/Optimizers.cpp
)
