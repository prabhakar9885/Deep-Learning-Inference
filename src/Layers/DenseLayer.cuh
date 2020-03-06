
#ifndef DENSELAYER
#define DENSELAYER


#include "Hidden.cuh"
#include "Layer.cuh"
#include "../Activation/Activation.cuh"

class DenseLayer : public Hidden {
public:
	DenseLayer(int size, Activation* activationFunc) :Hidden(size, activationFunc) {
	}
};

#endif // !DENSELAYER