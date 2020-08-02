#include "../../DeepLearningInference/src/NN/NN.cuh"
#include "../../DeepLearningInference/src/Layers/InputLayer.cuh"
#include "../../DeepLearningInference/src/Context/ContextFactory.cuh"
#include "../../DeepLearningInference/src/Context/ContextObject.cuh"
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <numeric> 
#include <chrono> 

using namespace std;

int main() {

	int inputHeight = 7;
	int inputWidth = 7;
	int inputChannels = 3;

	vector<vector<int>> cnn_layers_dims;						//					outChannel x inChannel x channelHeight x channelWidth
	cnn_layers_dims.push_back(vector<int>({ 4, 3, 3, 3 }));		// convLayer1:		OutChannels = 4	;	InChannels = 3	;	channelHeight = 3	;	channelWidth = 3
	cnn_layers_dims.push_back(vector<int>({ 5, 4, 3, 3 }));		// convLayer2:		OutChannels = 5	;	InChannels = 4	;	channelHeight = 3	;	channelWidth = 3
	cnn_layers_dims.push_back(vector<int>({ 2, 4, 3, 3 }));		// convLayer3:		OutChannels = 2	;	InChannels = 4	;	channelHeight = 3	;	channelWidth = 3

	vector<vector<float>> weightsAndBiases;
	for (int i = 0; i < cnn_layers_dims.size(); i++) {
		
		// Pushing the Weights
		size_t countOfAllWeightsInLayerI = accumulate(cnn_layers_dims[i].begin(), cnn_layers_dims[i].end(), 1, multiplies<int>());
		weightsAndBiases.push_back(vector<float>(countOfAllWeightsInLayerI, 1));

		// Pushing the Bias
		size_t countOfAllBiasInLayerI = cnn_layers_dims[i][0];
		weightsAndBiases.push_back(vector<float>(countOfAllBiasInLayerI, 1));
	}

	try {
		ContextFactory contextFactory;
		NN* cnnObj = new NN(contextFactory);

		Layer* inputLayer = new InputLayer(inputChannels * inputHeight * inputWidth, "Input layer");	// Input is an RGB image of size 5x5
		cnnObj->pushLayer(inputLayer);

		Layer* convLayer1 = new ConvLayer(cnn_layers_dims[0], Activation::ReLU, "Conv 1");
		cnnObj->pushLayer(convLayer1);

		Layer* poolLayer1 = new PoolingLayer(cnn_layers_dims[1], "Pool 1");
		cnnObj->pushLayer(poolLayer1);

		Layer* convLayer2 = new ConvLayer(cnn_layers_dims[2], Activation::ReLU, "Conv 2");
		cnnObj->pushLayer(convLayer2);

		Layer* convLayer3 = new ConvLayer(cnn_layers_dims[2], Activation::ReLU, "Conv 3");
		cnnObj->pushLayer(convLayer3);

		Layer* hiddenLayer1 = new DenseLayer(5, Activation::ReLU, "Dense 1");
		cnnObj->pushLayer(hiddenLayer1);

		Layer* hiddenLayer2 = new DenseLayer(4, Activation::ReLU, "Dense 2");
		cnnObj->pushLayer(hiddenLayer2);

		Layer* hiddenLayer3 = new DenseLayer(3, Activation::SIGMOID, "Dense 3");
		cnnObj->pushLayer(hiddenLayer2);

		vector<vector<float>> weights;
		cnnObj->init(weights);

		// Do inference
		vector<float> inputSample({ 
				1,1,1,
				1,1,1,
				1,1,1,

				0,1,0,
				1,1,1,
				1,1,0,

				1,0,1,
				0,1,1,
				1,1,0
			}
		);

		cout << "\nOutput: " << cnnObj->forward(inputSample) << "\n";
	}
	catch (string msg) {
		cout << "Exception: " << msg << "\n";
	}

	return 0;
}