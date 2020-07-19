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
	std::cout << "\nPushed " << "Layer-" << (layers.size() - 1) << ": " << layer->name << "(" << layerSize << ")";
}


void NN::init(std::vector<std::vector<float>> weightsAndBias) {
	std::cout << "\nInitiaizing the NN... ";
	unsigned __int64 noOfLayers = layers.size() - 1; // We are ignoring the input layer
	this->layers[0]->init();

	if (weightsAndBias.size() != noOfLayers * 2)
		throw "Weights dimension mismatches NN dimensions";

	for (size_t indexOfCurrentLayer = 1; indexOfCurrentLayer <= noOfLayers; indexOfCurrentLayer++) {
		this->layers[indexOfCurrentLayer]->initWeight(weightsAndBias[(indexOfCurrentLayer - 1) * 2]);
		this->layers[indexOfCurrentLayer]->initBias(weightsAndBias[(indexOfCurrentLayer - 1) * 2 + 1]);
	}
	std::cout << "\ndone";
}


float NN::forward(std::vector<float>& input_sample) {
	std::cout << "\nPredicting...";

	std::cout << "\nCreating the context... ";
	this->contextFactory.createContext(ContextType::cuBLAS);
	this->contextFactory.createContext(ContextType::cuDNN);
	std::cout << "done";

	for (size_t i = 1; i < layers.size(); i++)	{
		std::cout << "\n\tComputing for layer-" << i;
		this->layers[i]->forward(this->contextFactory, input_sample);
	}

	std::cout << "\nForword-prop is done";
	std::cout << "\nDestroying the context... ";
	this->contextFactory.releaseContext(ContextType::cuBLAS);
	this->contextFactory.releaseContext(ContextType::cuDNN);
	std::cout << "done";
	return input_sample[0];
}