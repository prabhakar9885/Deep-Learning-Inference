
#include "DenseNetwork.cuh"

using namespace std;
using namespace std::chrono;


vector<vector<float>> parse_and_load_data(string fileName)
{
	string line;
	ifstream myfile(fileName);
	vector<vector<float>> weightsAndBiases;

	if (myfile.is_open())
	{
		std::cout << "\nParsing the weights file...";
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
			weightsAndBiases.push_back(tokens);
		}
		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<milliseconds>(stop - start);

		std::cout << "\nDone... " << duration.count() << " milliseconds";
		myfile.close();
	}
	else
		std::cout << "Unable to open file";
	return weightsAndBiases;
}



int main() {

	vector<int> layers_dims({ 12288, 128, 128, 128, 128, 128, 64, 64, 32, 1 });
	//vector<int> layers_dims({ 6, 4, 3, 2, 2, 2, 3, 4, 2, 1 });
	vector<vector<float>> weightsAndBiases = parse_and_load_data(R"(Model\weights.lst)");

	try {
		ContextFactory contextFactory;
		NN* nnObj = new NN(contextFactory);

		// Push Input layer
		Layer* inputLayer = new InputLayer(vector<int>(1,layers_dims[0]), "Input layer");
		nnObj->pushLayer(inputLayer);

		// Push Hidden layers
		for (int i = 1; i < layers_dims.size() - 1; i++) {
			DenseLayer* hiddenLayer = new DenseLayer(layers_dims[i], Activation::ReLU, "Hidden " + to_string(i));
			nnObj->pushLayer(hiddenLayer);
		}

		// Push Output layer
		Layer* outputLayer = new DenseLayer(layers_dims[layers_dims.size() - 1], Activation::SIGMOID, "Output layer");
		nnObj->pushLayer(outputLayer);

		// Init NN with weights
		nnObj->init(weightsAndBiases);
		weightsAndBiases.erase(weightsAndBiases.begin(), weightsAndBiases.end());

		// Do inference
		vector<float> inputSample(layers_dims[0], 1);
		std::cout << "\n" << nnObj->forward(inputSample) << "\n";
	}
	catch (string msg) {
		std::cout << "Exception: " << msg << "\n";
	}

	return 0;
}