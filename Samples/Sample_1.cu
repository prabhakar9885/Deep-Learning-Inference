
#include "../src/NN/NN.cuh"
#include "../src/Layers/Layer.cuh"
#include "../src/Layers/DenseLayer.cuh"
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
	weight.clear();

	prev_layer_size = layers_dims[0];
	for (size_t i = 1; i < layers_dims.size(); i++)
	{
		int current_layer_size = layers_dims[i];

		// Insert Bias (B)
		vector<float> b_params_for_layer_i = weights_data["B" + to_string(i)];
		weight.push_back(b_params_for_layer_i);

		// Insert Weights (W)
		vector<float> w_params_for_layer_i = weights_data["W" + to_string(i)];
		int num_of_params_for_layer_i = w_params_for_layer_i.size();
		vector<float> inp_wts_for_a_node_in_layer_i;
		for (size_t start_index = 0; start_index < num_of_params_for_layer_i; start_index += prev_layer_size)
		{
			vector<float> inp_wts_for_a_node_in_layer_i(w_params_for_layer_i.begin() + start_index, w_params_for_layer_i.begin() + start_index + prev_layer_size);
			weight.push_back(inp_wts_for_a_node_in_layer_i);
		}
		
		weights.push_back(weight);
		weight.clear();
		prev_layer_size = current_layer_size;
	}
	return weights;
}


