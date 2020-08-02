#include "PoolingLayer.cuh"

PoolingLayer::PoolingLayer(std::vector<int> size, std::string name, PoolingType poolingType, int padding, int stride) : Layer(size, name) {
	this->poolingType = poolingType;
	this->padding = padding;
	this->stride = stride;
}

LayerType PoolingLayer::getLayerType() {
	return LayerType::POOL;
}

void PoolingLayer::init() {
}

void PoolingLayer::initWeight(const std::vector<float>& weights) {
}

void PoolingLayer::initBias(const std::vector<float>& bias) {
}

void PoolingLayer::forward(ContextFactory contextFactory, std::vector<float>& input_sample) {
	//Z = W * X + B
	//BlasUtils::axpby_vector_matrix(contextFactory, this->weights, input_sample, this->bias);
	//	A = f(Z)
	//BlasUtils::computeActivation(input_sample, this->activationFunc);
}
