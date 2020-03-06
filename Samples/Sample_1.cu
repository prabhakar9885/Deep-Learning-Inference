
#include "../src/NN/NN.cuh"
#include "../src/Layers/Layer.cuh"
#include "../src/Layers/Input.cuh"
#include "../src/Layers/Hidden.cuh"
#include "../src/Layers/DenseLayer.cuh"
#include "../src/Layers/Output.cuh"
#include "../src/Activation/Sigmoid.cuh"
#include "../src/Activation/Softmax.cuh"
#include "iostream"

using namespace std;

int main() {
	cout << "Hello World.\n";

	NN* nnObj = new NN();

	// Push Input layer
	Layer* inputLayer = new Input(10);
	nnObj->pushLayer(inputLayer);

	// Push Hidden layers
	vector<int> hiddenLayersSize = {};
	for (int i = 0; i < 4; i++) {
		Layer* hiddenLayer = new DenseLayer(10, new Sigmoid());
		nnObj->pushLayer(hiddenLayer);
	}

	// Push Output layer
	Layer* outputLayer = new Output(10, new Softmax());
	nnObj->pushLayer(outputLayer);

	// Init NN with weights
	vector<vector<float>> *weights = nullptr;
	nnObj->init(weights);

	// Do inference
	vector<int> *inputSample = nullptr;
	cout << nnObj->predict(inputSample) << "\n";

	return 0;
}