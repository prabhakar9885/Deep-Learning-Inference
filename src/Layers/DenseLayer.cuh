
#ifndef DENSELAYER
#define DENSELAYER


#include "Layer.cuh"
#include "../Activation/Activation.cuh"

class DenseLayer : public Layer {
public:
	DenseLayer(int size, Activation activationFunc) :Layer(size, activationFunc) {
	}
	DenseLayer(int size, Activation activationFunc, std::string name) :Layer(size, activationFunc, name) {
	}
};

#endif // !DENSELAYER