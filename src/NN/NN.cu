#include "NN.cuh"


NN::NN(cublasHandle_t handle) {
	this->handle = handle;
}


void NN::pushLayer(Layer& layer) {
	layers.push_back(layer);
	std::cout << "\nPushed " << "Layer-" << (layers.size() - 1) << ": " << layer.name << "(" << layer.size << ")";
}


void NN::init(std::vector<std::vector<std::vector<float>>> weights) {
	std::cout << "\nInitiaizing the NN... ";
	unsigned __int64 noOfLayers = layers.size();
	this->weights.push_back(std::vector<std::vector<float>>(0));
	this->bias.push_back(std::vector<float>(0));
	if (weights.size() != noOfLayers)
		throw "Weights dimension mismatches NN dimensions";
	for (size_t indexOfCurrentLayer = 1; indexOfCurrentLayer < noOfLayers; indexOfCurrentLayer++) {
		std::vector<std::vector<float>>& weightsOfCurrentLayer = weights[indexOfCurrentLayer];
		this->bias.push_back(weightsOfCurrentLayer[0]);
		weightsOfCurrentLayer.erase(weightsOfCurrentLayer.begin());
		this->weights.push_back(weightsOfCurrentLayer);
		//cout << "\nWts for Layer-" << indexOfCurrentLayer << ": " << weightsOfCurrentLayer.size() << "x" << weightsOfCurrentLayer[0].size();
	}
	std::cout << "\ndone";
}


float NN::forword(std::vector<float>& input_sample) {
	std::cout << "\nPredicting...";

	for (size_t i = 1; i < layers.size(); i++)
	{
		std::cout << "\n\tComputing for layer-" << i;
		//  Z = W*X + B
		BlasUtils::axpby_vector_matrix(this->handle, weights[i], input_sample, bias[i]);
		BlasUtils::computeActivation(input_sample, layers[i].activationFunc);
		//	A = f(Z)
	}

	std::cout << "\ndone";
	return input_sample[0];
}