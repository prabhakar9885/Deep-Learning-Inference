
#include "../src/NN/NN.cuh"
#include "../src/Layers/Layer.cuh"
#include "../src/Layers/Input.cuh"
#include "../src/Layers/Hidden.cuh"
#include "../src/Layers/DenseLayer.cuh"
#include "../src/Layers/Output.cuh"
#include "../src/Activation/Activation.cuh"
#include "iostream"

using namespace std;

int main() {
	try {
		NN* nnObj = new NN();

		// Push Input layer
		Input inputLayer(10, "Input layer");
		nnObj->pushLayer(inputLayer);

		// Push Hidden layers
		for (int i = 0; i < 8; i++) {
			DenseLayer hiddenLayer(10, Activation::ReLU, "Hidden " + to_string(i));
			nnObj->pushLayer(hiddenLayer);
		}

		// Push Output layer
		Output outputLayer(10, Activation::SOFTMAX, "Output layer");
		nnObj->pushLayer(outputLayer);

		// Init NN with weights
		vector<vector<float>> weights;
		weights.resize( 10, vector<float>(0));
		nnObj->init(weights);

		// Do inference
		vector<int> inputSample;
		cout << nnObj->forword(inputSample) << "\n";
	}
	catch (string msg) {
		cout << "Exception: " << msg << "\n";
	}

	return 0;
}