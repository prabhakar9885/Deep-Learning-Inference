#include "NN.cuh"


NN::NN(ContextFactory contextFactory) {
	this->contextFactory = contextFactory;
}


void NN::pushLayer(Layer* layer) {
	layers.push_back(layer);
	std::string layerSize;
	for (int num : layer->getSize())
		layerSize += std::to_string(num) + ",";
	layerSize.pop_back();
	std::cout << "\nPushed " << "Layer-" << (layers.size() - 1) << ": " << layer->getName() << "(" << layerSize << ")";
}


void NN::init(std::list<std::vector<float>> weightsAndBias) 
{
	std::cout << "\nInitiaizing the NN... ";
	unsigned __int64 noOfLayers = layers.size() - 1; // We are ignoring the input layer

	if (weightsAndBias.size() != noOfLayers * 2)
		throw "Weights dimension mismatches NN dimensions";

	auto weightsAndBiasIterator = weightsAndBias.begin();
	auto layerIterator = this->layers.begin();
	(*layerIterator)->init();
	layerIterator++;	// Skip the input layer, as there are no incoming weights for the input layer

	while (layerIterator != this->layers.end() && weightsAndBiasIterator != weightsAndBias.end())
	{
		Layer* currentLayer = *layerIterator;
		currentLayer->initWeight(*weightsAndBiasIterator);
		weightsAndBiasIterator++;
		currentLayer->initBias(*weightsAndBiasIterator);
		weightsAndBiasIterator++;
		layerIterator++;
	}

	if (layerIterator == this->layers.end() ^ weightsAndBiasIterator == weightsAndBias.end())
		throw "Initializing the Weights and biases failed.";

	std::cout << "\ndone";
}

float NN::forward(std::vector<float>& input_sample) {
	std::cout << "\nPredicting...";

	std::cout << "\nCreating the context... ";
	this->contextFactory.createContext(ContextType::cuBLAS);
	this->contextFactory.createContext(ContextType::cuDNN);
	std::cout << "done";

	float* inputData = input_sample.data();
	int inputElementCount = input_sample.size();
	std::list<Layer*>::iterator layerIterator = this->layers.begin();

	while (layerIterator != this->layers.end())
	{
		Layer* currentLayer = *layerIterator;
		std::cout << "\n\tComputing for layer-" << std::distance(this->layers.begin(), layerIterator);
		currentLayer->forward(this->contextFactory, (void*)inputData, inputElementCount, layerIterator);
		layerIterator++;
	}

	std::cout << "\nForword-prop is done";
	std::cout << "\nDestroying the context... ";
	this->contextFactory.releaseContext(ContextType::cuBLAS);
	this->contextFactory.releaseContext(ContextType::cuDNN);
	std::cout << "done";
	return input_sample[0];
}