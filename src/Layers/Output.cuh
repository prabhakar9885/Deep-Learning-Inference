
#ifndef OUTPUT
#define OUTPUT

#include "Layer.cuh"
#include "../Activation/Activation.cuh"

class Output : public Layer {
public:
	Output(int size, Activation activationFunc) :Layer(size, activationFunc) {
	}
	Output(int size, Activation activationFunc, string name) :Layer(size, activationFunc, name) {
	}
};

#endif // !OUTPUT