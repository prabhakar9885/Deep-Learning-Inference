#include "InputLayer.cuh"

InputLayer::InputLayer(int size, std::string name) : Layer(std::vector<int>(1, size), name) {
}

LayerType InputLayer::getLayerType() {
	return LayerType::INPUT;
}

void* InputLayer::allocAndInitDataOnDevice(void* inputDataOnHost, int inputElementCount, std::list<Layer*>::iterator layerIterator) {

	Layer* nextLayer = *(std::next(layerIterator, 1));

	switch (nextLayer->getLayerType()) {

	case LayerType::DENSE:
		float* inputVectorOnHost = (float*)inputDataOnHost;
		int inputSize = this->size[0];
		if (inputSize != inputElementCount)
			throw "Elements in input-data doesn't match the size of input layer.";
		float* inputVectorOnDevice; // n x 1
		cudaMallocManaged((void**)&inputVectorOnDevice, inputSize * 1 * sizeof(float));
		for (size_t i = 0; i < inputSize; i++)
		{
			inputVectorOnDevice[i] = inputVectorOnHost[i];
		}
		this->outputOfCurrentLayer = inputVectorOnDevice;
		break;

		/*case LayerType::CONV:
			break;

		case LayerType::POOL:
			break;*/

	}

	return this->outputOfCurrentLayer;
}


void* InputLayer::getOuputOnDevice() {
	return this->outputOfCurrentLayer;
}

void InputLayer::init() {
}

void InputLayer::initWeight(const std::vector<float>& weights) {
}

void InputLayer::initBias(const std::vector<float>& bias) {
}

void InputLayer::forward(ContextFactory contextFactory, void* inputSample, int inputElementCount , std::list<Layer*>::iterator layerIterator) {
	this->allocAndInitDataOnDevice(inputSample, inputElementCount, layerIterator);

	float* outputOfCurrentLayer = (float*)inputSample;
	std::cout << " Activation=> ";
	for (int i = 0; i < inputElementCount; i++)
		std::cout << " " << outputOfCurrentLayer[i];
}