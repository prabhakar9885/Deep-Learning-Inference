
#ifndef LAYER
#define LAYER

#include <vector>
#include <string>
#include "../Activation/Activation.cuh"
#include "../cuBLAS/blasUtills.cuh"

using namespace std;
using namespace utils;

class Layer {
public:
	int size;
	vector<float> value;
	string name;
	Activation activationFunc;
	float* activationValue;

	Layer(int size, Activation activationFunc) {
		this->size = size;
		this->value.resize(size);
		this->activationFunc = activationFunc;
		this->activationValue = nullptr;
		this->name = "unnamed";
	}

	Layer(int size, Activation activationFunc, string name) {
		this->size = size;
		this->value.resize(size);
		this->activationFunc = activationFunc;
		this->activationValue = nullptr;
		this->name = name;
	}

	void applyActivation() {
		computeActivation(this->value, this->activationFunc);
	}
};

#endif // !LAYER

