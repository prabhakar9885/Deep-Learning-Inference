#include "DenseLayer.cuh"
#include "../cuBLAS/blasUtills.cuh"

DenseLayer::DenseLayer(int size, Activation activationFunc, std::string name) : Layer(std::vector<int>(1, size), name) {
	this->activationFunc = activationFunc;;
}

LayerType DenseLayer::getLayerType() {
	return LayerType::DENSE;
}

void DenseLayer::init() {
	this->weights = std::vector<std::vector<float>>(0);
	this->bias = std::vector<float>(0);
}

void DenseLayer::initWeight(const std::vector<float> &weights) {
	int numberOfRows = this->getSize()[0];
	int numberOfCols = weights.size() / numberOfRows;
	for (int i = 0; i < numberOfRows; i++) {
		std::vector<float> row(weights.begin() + i, weights.begin() + i + numberOfCols);
		this->weights.push_back(row);
	}
}

void DenseLayer::initBias(const std::vector<float>& bias) {
	this->bias = bias;
}

void DenseLayer::forward(ContextFactory contextFactory, std::vector<float>& input_sample) {
	//Z = W * X + B
	BlasUtils::axpby_vector_matrix(contextFactory, this->weights, input_sample, this->bias);
	//	A = f(Z)
	BlasUtils::computeActivation(input_sample, this->activationFunc);
}
