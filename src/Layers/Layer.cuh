
#ifndef LAYER
#define LAYER

#include "../Activation/Activation.cuh"
#include <string>

using namespace std;

class Layer {
public:
	int size;
	float* value;
	string name;
	Activation* activationFunc;
	float* activationValue;

	Layer(int size, Activation* activationFunc) {
		this->size = size;
		this->value = nullptr;
		this->activationFunc = activationFunc;
		this->activationValue = nullptr;
		this->name = "unnamed";
	}

	Layer(int size, Activation* activationFunc, string name) {
		this->size = size;
		this->value = nullptr;
		this->activationFunc = activationFunc;
		this->activationValue = nullptr;
		this->name = name;
	}

	void applyActivation() {
		this->activationValue = activationFunc->compute(value);
	}
};

#endif // !LAYER