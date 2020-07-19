#include "ConvLayer.cuh"

ConvLayer::ConvLayer(std::vector<int> size, Activation activationFunc, std::string name, int padding, int stride, int dilation) : Layer(size, name) {
	this->activationFunc = activationFunc;
	this->padding = padding;
	this->stride = stride;
	this->dilation = dilation;
}

LayerType ConvLayer::getLayerType() {
	return LayerType::CONV;
}

void ConvLayer::init() {
	this->weights = std::vector<std::vector<float>>(0);
	this->bias = std::vector<float>(0);
}

void ConvLayer::initWeight(const std::vector<float>& weights) {
	int numberOfRows = this->getSize()[0];
	int numberOfCols = weights.size() / numberOfRows;
	for (int i = 0; i < numberOfRows; i++) {
		std::vector<float> row(weights.begin() + i, weights.begin() + i + numberOfCols);
		this->weights.push_back(row);
	}
}

void ConvLayer::initBias(const std::vector<float>& bias) {
	this->bias = bias;
}

void ConvLayer::forward(ContextFactory contextFactory, std::vector<float>& input_sample) {
	//Z = W * X + B
	//BlasUtils::axpby_vector_matrix(contextFactory, this->weights, input_sample, this->bias);
	//	A = f(Z)
	//BlasUtils::computeActivation(input_sample, this->activationFunc);
}
