#include "NN.cuh"


NN::NN(ContextFactory contextFactory) {
	this->contextFactory = contextFactory;
}


void NN::pushLayer(Layer& layer) {
	layers.push_back(layer);
	std::cout << "\nPushed " << "Layer-" << (layers.size() - 1) << ": " << layer.name << "(" << layer.size << ")";
}


void NN::init(std::vector<std::vector<std::vector<float>>> weights) {
	std::cout << "\nInitiaizing the NN... ";
	unsigned __int64 noOfLayers = layers.size();
	this->layers[0].weights = std::vector<std::vector<float>>(0);
	this->layers[0].bias = std::vector<float>(0);

	if (weights.size() != noOfLayers)
		throw "Weights dimension mismatches NN dimensions";

	for (size_t indexOfCurrentLayer = 1; indexOfCurrentLayer < noOfLayers; indexOfCurrentLayer++) {
		std::vector<std::vector<float>>& weightsOfCurrentLayer = weights[indexOfCurrentLayer];
		this->layers[indexOfCurrentLayer].bias = weightsOfCurrentLayer[0];
		weightsOfCurrentLayer.erase(weightsOfCurrentLayer.begin());
		this->layers[indexOfCurrentLayer].weights = weightsOfCurrentLayer;
	}
	std::cout << "\ndone";
}


float NN::forward(std::vector<float>& input_sample) {
	std::cout << "\nPredicting...";

	std::cout << "\nCreating the context... ";
	this->contextFactory.createContext(ContextType::cuBLAS);
	std::cout << "done";

	for (size_t i = 1; i < layers.size(); i++)
	{
		std::cout << "\n\tComputing for layer-" << i;
		this->layers[i].forward(this->contextFactory, input_sample);
	}

	std::cout << "\nForword-prop is done";
	std::cout << "\nDestroying the context... ";
	this->contextFactory.releaseContext(ContextType::cuBLAS);
	std::cout << "done";
	return input_sample[0];
}