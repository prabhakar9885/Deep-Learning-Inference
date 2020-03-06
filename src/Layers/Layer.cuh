
#ifndef LAYER
#define LAYER

#include "../Activation/Activation.cuh"

class Layer {
public:
	int size;
	float* value;
	Activation* activationFunc;
	float* activationValue;

	Layer(int size, Activation* activationFunc) {
		this->size = size;
		this->value = nullptr;
		this->activationFunc = activationFunc;
		this->activationValue = nullptr;
	}

	void applyActivation() {
		this->activationValue = activationFunc->compute(value);
	}
};

#endif // !LAYER