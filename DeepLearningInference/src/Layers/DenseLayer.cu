#include "DenseLayer.cuh"
#include "../cuBLAS/blasUtills.cuh"

DenseLayer::DenseLayer(int size, Activation activationFunc, std::string name) : Layer(std::vector<int>(1, size), name) {
	this->activationFunc = activationFunc;;
}

LayerType DenseLayer::getLayerType() {
	return LayerType::DENSE;
}


void* DenseLayer::allocAndInitDataOnDevice(void* inputDataOnHost, int inputElementCount, std::list<Layer*>::iterator layerIterator) {

	Layer* prevLayer = *(std::prev(layerIterator, 1));

	float* inputVectorOnHost = (float*)inputDataOnHost;
	int inputSize = prevLayer->getSize().front();
	float* inputVectorOnDevice; // n x 1
	cudaMallocManaged((void**)&inputVectorOnDevice, this->getSize().front() * 1 * sizeof(float));
	for (size_t i = 0; i < inputSize; i++)
	{
		inputVectorOnDevice[i] = 0;
	}
	this->outputOfCurrentLayer = inputVectorOnDevice;

	return this->outputOfCurrentLayer;
}


// Allocates host memory for holding the weights and bias pertaining to links from previous layer to the current layer
void DenseLayer::init() {
	this->weights = std::vector<std::vector<float>>(0);
	this->bias = std::vector<float>(0);
}

void* DenseLayer::getOuputOnDevice() {
	return (void*)(this->outputOfCurrentLayer);
}

void DenseLayer::initWeight(const std::vector<float> &weights) {
	int numberOfRows = this->getSize()[0];
	int numberOfCols = weights.size() / numberOfRows;
	int startIndexOfIthRow = 0;
	for (int i = 0; i < numberOfRows; i++) {
		std::vector<float> row(weights.begin() + startIndexOfIthRow, weights.begin() + startIndexOfIthRow + numberOfCols);
		this->weights.push_back(row);
		startIndexOfIthRow += numberOfCols;
	}
}

void DenseLayer::initBias(const std::vector<float>& bias) {
	this->bias = bias;
}


void DenseLayer::forward(ContextFactory contextFactory, void* inputSample, int inputElementCount, std::list<Layer*>::iterator layerIterator) {
		
	Layer* prevLayer = *(std::prev(layerIterator, 1));
	float* outputFromPreviousLayer = (float*)prevLayer->getOuputOnDevice();
	int outputSizeFromPreviousLayer = prevLayer->getSize().front();
	
	float* outputForCurrentLayer;
	int outputSizeForCurrentLayer = this->getSize().front();

	//Z = W * X + B
	BlasUtils::axpby_vector_matrix(contextFactory,
		outputFromPreviousLayer, outputSizeFromPreviousLayer,
		this->weights, this->bias,
		outputForCurrentLayer, outputSizeForCurrentLayer
	);

	//	A = f(Z)
	BlasUtils::computeActivation(outputForCurrentLayer, outputSizeForCurrentLayer, this->activationFunc);

	std::cout << " Activation=> ";
	for (int i = 0; i < outputSizeForCurrentLayer; i++)
		std::cout << " " << outputForCurrentLayer[i];

	this->outputOfCurrentLayer = outputForCurrentLayer;
}