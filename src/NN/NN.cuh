
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
	NN() {

	}

	void pushLayer(Layer& layer) {
		layers.push_back(layer);
		cout << "Pushed: " << layer.name << "\n";
	}

	void init(vector<vector<vector<float>>> weights) {
		cout << "Initiaizing the NN... ";
		unsigned __int64 noOfLayers = layers.size();
		this->weights.push_back(vector<vector<float>>(0));
		if (weights.size() != noOfLayers )
			throw "Weights dimension mismatches NN dimensions";
		for (size_t indexOfCurrentLayer = 0; indexOfCurrentLayer < noOfLayers; indexOfCurrentLayer++) {
			this->weights.push_back( weights[indexOfCurrentLayer] );
		}
		cout << "done\n";
	}

	int forword(vector<float>& input_sample) {
		cout << "Predicting...";
		
		for (size_t i = 1; i < layers.size(); i++)
		{
			cout << "\n\tComputing for layer-" << i;
			//  Z = W*X + B
			axpb_vector_matrix( weights[i], input_sample, 1);
			computeActivation(input_sample, layers[i].activationFunc);
			//	A = f(Z)
		}

		cout << "\ndone";
		return -1;
	}
};