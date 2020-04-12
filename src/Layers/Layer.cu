#include "Layer.cuh"
#include "../cuBLAS/blasUtills.cuh"

Layer::Layer(int size, Activation activationFunc) {
	this->size = size;
	this->value.resize(size);
	this->activationFunc = activationFunc;
	this->activationValue = nullptr;
	this->name = "unnamed";
}

Layer::Layer(int size, Activation activationFunc, std::string name) {
	this->size = size;
	this->value.resize(size);
	this->activationFunc = activationFunc;
	this->activationValue = nullptr;
	this->name = name;
}

void Layer::applyActivation() {
	BlasUtils::computeActivation(this->value, this->activationFunc);
}

void Layer::forward(cublasHandle_t handle, std::vector<float>& input_sample) {
	//Z = W * X + B
	BlasUtils::axpby_vector_matrix(handle, this->weights, input_sample, this->bias);
	//	A = f(Z)
	BlasUtils::computeActivation(input_sample, this->activationFunc);
}
