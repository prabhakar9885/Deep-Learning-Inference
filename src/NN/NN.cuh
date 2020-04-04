
#include <iostream>
#include "cuda.h"
#include "cuda_runtime_api.h"
#include <vector>
#include "../Layers/Layer.cuh"
#include "../Activation/Activation.cuh"

using namespace std;

class NN {
public:
	vector<Layer> layers;
	vector<vector<vector<float>>> weights;
	vector<vector<float>> bias;
	NN() {

	}

	void pushLayer(Layer& layer) {
		layers.push_back(layer);
		cout << "\nPushed " << "Layer-" << (layers.size() - 1) << ": " << layer.name << "(" << layer.size << ")";
	}

	void init(vector<vector<vector<float>>> weights) {
		cout << "\nInitiaizing the NN... ";
		unsigned __int64 noOfLayers = layers.size();
		this->weights.push_back(vector<vector<float>>(0));
		this->bias.push_back(vector<float>(0));
		if (weights.size() != noOfLayers )
			throw "Weights dimension mismatches NN dimensions";
		for (size_t indexOfCurrentLayer = 1; indexOfCurrentLayer < noOfLayers; indexOfCurrentLayer++) {
			vector<vector<float>> &weightsOfCurrentLayer = weights[indexOfCurrentLayer];
			this->bias.push_back(weightsOfCurrentLayer[0]);
			weightsOfCurrentLayer.erase(weightsOfCurrentLayer.begin());
			this->weights.push_back(weightsOfCurrentLayer);
			//cout << "\nWts for Layer-" << indexOfCurrentLayer << ": " << weightsOfCurrentLayer.size() << "x" << weightsOfCurrentLayer[0].size();
		}
		cout << "\ndone";
	}

	float forword(vector<float>& input_sample) {
		cout << "\nPredicting...";
		
		for (size_t i = 1; i < layers.size(); i++)
		{
			cout << "\n\tComputing for layer-" << i;
			//  Z = W*X + B
			axpby_vector_matrix(weights[i], input_sample, bias[i]);
			computeActivation(input_sample, layers[i].activationFunc);
			//	A = f(Z)
		}

		cout << "\ndone";
		return input_sample[0];
	}
};