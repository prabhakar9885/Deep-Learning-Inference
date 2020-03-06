
#ifndef HIDDEN
#define HIDDEN


#include "Layer.cuh"
#include "../Activation/Activation.cuh"
#include "../Activation/Sigmoid.cuh"


class Hidden : public Layer {
public:
	Hidden(int size, Activation* activationFunc):Layer(size, activationFunc) {
	}
};

#endif // !HIDDEN