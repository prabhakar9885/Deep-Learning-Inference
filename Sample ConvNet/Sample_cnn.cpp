#include "../../DeepLearningInference/src/NN/NN.cuh"
#include "../../DeepLearningInference/src/Context/ContextFactory.cuh"
#include "../../DeepLearningInference/src/Context/ContextObject.cuh"
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <chrono> 

using namespace std;

int main() {

	vector<int> layers_dims({ 12288, 128, 128, 128, 128, 128, 64, 64, 32, 1 });
	//vector<int> layers_dims({ 6, 4, 3, 2, 2, 2, 3, 4, 2, 1 });
	unordered_map<string, vector<float>> weights_data; // = parse_and_load_data(R"(Model\weights.lst)");

	try {
		ContextFactory contextFactory;
		NN* cnnObj = new NN(contextFactory);

		// Push Input layer
		Layer inputLayer(layers_dims[0], Activation::NONE, "Input layer");
		cnnObj->pushLayer(inputLayer);

		ConvLayer convLayer1(layers_dims[1], Activation::ReLU, "Hidden " + to_string(i));
		cnnObj->pushLayer(convLayer1);

		PoolLayer poolLayer1(layers_dims[1], Activation::ReLU, "Hidden " + to_string(i));
		cnnObj->pushLayer(poolLayer1);

		ConvLayer convLayer2(layers_dims[1], Activation::ReLU, "Hidden " + to_string(i));
		cnnObj->pushLayer(convLayer2);

		PoolLayer poolLayer2(layers_dims[1], Activation::ReLU, "Hidden " + to_string(i));
		cnnObj->pushLayer(poolLayer2);

		// Push Hidden layers
		for (int i = 1; i < layers_dims.size() - 1; i++) {
			DenseLayer hiddenLayer(layers_dims[i], Activation::ReLU, "Hidden " + to_string(i));
			cnnObj->pushLayer(hiddenLayer);
		}

		// Push Output layer
		Layer outputLayer(layers_dims[layers_dims.size() - 1], Activation::SIGMOID, "Output layer");
		cnnObj->pushLayer(outputLayer);

		// Init NN with weights
		vector<vector<vector<float>>> weights = get_weights(layers_dims, weights_data);
		cnnObj->init(weights);
		weights.erase(weights.begin(), weights.end());

		// Do inference
		vector<float> inputSample(layers_dims[0], 1);
		cout << "\n" << cnnObj->forward(inputSample) << "\n";
	}
	catch (string msg) {
		cout << "Exception: " << msg << "\n";
	}

	return 0;
}