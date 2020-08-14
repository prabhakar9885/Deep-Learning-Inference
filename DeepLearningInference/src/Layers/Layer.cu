#include "Layer.cuh"

Layer::Layer(std::vector<int> size) {
	this->size = size;
	this->name = "unnamed";
}

Layer::Layer(std::vector<int> size, std::string name) {
	this->size = size;
	this->name = name;
}

std::string Layer::getName()
{
	return this->name;
}

std::vector<int> Layer::getSize() {
	return this->size;
}


void* Layer::getOuputOnDevice() {
	return nullptr;
}


void* Layer::getOuputToHost() {
	int numberOfElements = accumulate(this->getSize().begin(), this->getSize().end(), 1, std::multiplies<int>());
	float* outputOnDevice = new float[numberOfElements];
	for (int i = 0; i < numberOfElements; i++)
		outputOnDevice[i] = ((float*)(this->outputOfCurrentLayer))[i];
	return outputOnDevice;
}