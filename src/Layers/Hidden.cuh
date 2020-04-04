
#ifndef HIDDEN
#define HIDDEN


#include "Layer.cuh"
#include "../Activation/Activation.cuh"


class Hidden : public Layer {
public:

	vector<float> bias;

	Hidden(int size, Activation activationFunc):Layer(size, activationFunc) {
	}

	Hidden(int size, Activation activationFunc, string name) :Layer(size, activationFunc, name) {
	}
};

#endif // !HIDDEN