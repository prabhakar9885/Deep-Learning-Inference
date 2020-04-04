
#include "../src/NN/NN.cuh"
#include "../src/Layers/Layer.cuh"
#include "../src/Layers/Input.cuh"
#include "../src/Layers/Hidden.cuh"
#include "../src/Layers/DenseLayer.cuh"
#include "../src/Layers/Output.cuh"
#include "../src/Activation/Activation.cuh"
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <chrono> 


using namespace std;
using namespace std::chrono;


unordered_map<string, vector<float>> parse_and_load_data(string fileName)
{
	string line;
	ifstream myfile(fileName);
	unordered_map<string, vector<float>> um;

	if (myfile.is_open())
	{
		bool isBias = false;
		int index = 1;
		cout << "\nParsing the weights file...";
		auto start = high_resolution_clock::now();
		while (getline(myfile, line))
		{
			stringstream streamObj(line);
			string intermediate;

			vector<float> tokens;
			while (getline(streamObj, intermediate, ' '))
			{
				tokens.push_back(stof(intermediate));
			}
			if (isBias)
			{
				um["B" + to_string(index)] = tokens;
				index++;
				isBias = !isBias;
			}
			else
			{
				um["W" + to_string(index)] = tokens;
				isBias = !isBias;
			}
		}
		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<milliseconds>(stop - start);

		cout << "\nDone... " << duration.count() << " milliseconds";
		myfile.close();
	}
	else
		cout << "Unable to open file";
	return um;
}


vector<vector<vector<float>>> get_weights(std::vector<int>& layers_dims, unordered_map<string, vector<float>> weights_data)
{
	vector<vector<vector<float>>> weights;
	int prev_layer_size = 1;
	vector<vector<float>> weight(layers_dims[0], vector<float>(prev_layer_size, 1.0));
	weights.push_back(weight);
	prev_layer_size = layers_dims[0];
	for (size_t i = 1; i < layers_dims.size(); i++)
	{
		int current_layer_size = layers_dims[i];
		vector<vector<float>> weight(current_layer_size, vector<float>(prev_layer_size, 1.0));
		weight.insert(weight.begin(), vector<float>(current_layer_size, 1.0));
		weights.push_back(weight);
		prev_layer_size = current_layer_size;
	}
	return weights;
}


int main() {

	vector<int> layers_dims({ 12288, 128, 128, 128, 128, 128, 64, 64, 32, 1 });
	//vector<int> layers_dims({ 6, 4, 3, 2, 2, 2, 3, 4, 2, 1 });
	unordered_map<string, vector<float>> weights_data = parse_and_load_data(R"(C:\Users\prabhakarb\source\repos\Deep Learning Inference\Model\weights.lst)");

	try {
		NN* nnObj = new NN();

		// Push Input layer
		Input inputLayer(layers_dims[0], "Input layer");
		nnObj->pushLayer(inputLayer);

		// Push Hidden layers
		for (int i = 1; i < layers_dims.size()-1; i++) {
			DenseLayer hiddenLayer(layers_dims[i], Activation::ReLU, "Hidden " + to_string(i));
			nnObj->pushLayer(hiddenLayer);
		}

		// Push Output layer
		Output outputLayer(layers_dims[layers_dims.size() - 1], Activation::SIGMOID, "Output layer");
		nnObj->pushLayer(outputLayer);

		// Init NN with weights
		vector<vector<vector<float>>> weights = get_weights(layers_dims, weights_data);
		nnObj->init(weights);

		// Do inference
		vector<float> inputSample(layers_dims[0], 1);
		cout << "\n" << nnObj->forword(inputSample) << "\n";
	}
	catch (string msg) {
		cout << "Exception: " << msg << "\n";
	}

	return 0;
}