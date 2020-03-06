
#include "cuda.h"
#include "cuda_runtime_api.h"
#include <vector>
#include "../Layers/Layer.cuh"

using namespace std;

class NN {
public:
	vector<Layer> layers;
	vector<dim3> weights;
	NN() {

	}

	void pushLayer(Layer* layer) {

	}

	void init(vector<vector<float>>* weights) {

	}

	int predict(vector<int> *input_sample) {
		return -1;
	}
};