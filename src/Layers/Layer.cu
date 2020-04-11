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