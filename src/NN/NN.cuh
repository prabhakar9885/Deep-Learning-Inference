
#include <iostream>
#include "cuda.h"
#include "cuda_runtime_api.h"
#include <vector>
#include "../Layers/Layer.cuh"

using namespace std;

class NN {
public:
	vector<Layer> layers;
	vector<vector<float>> weights;
	NN() {

	}

	void pushLayer(Layer& layer) {
		layers.push_back(layer);
		cout << "Pushed: " << layer.name << "\n";
	}

	void init(vector<vector<float>>& weights) {
		cout << "Initiaizing the NN... ";
		unsigned __int64 noOfLayers = layers.size();
		if (weights.size() != noOfLayers )
			throw "Weights dimension mismatches NN dimensions";
		for (unsigned __int64 indexOfCurrentLayer = 1u; indexOfCurrentLayer < noOfLayers; indexOfCurrentLayer++) {
			this->weights.push_back( weights[indexOfCurrentLayer] );
		}
		cout << "done\n";
	}

	int predict(vector<int>& input_sample) {
		cout << "Predicting...";
		cout << "done \n";
		return -1;
	}
};