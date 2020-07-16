
#ifndef CONVLAYER
#define CONVLAYER

#include "Layer.cuh"
#include "../Activation/Activation.cuh"
#include "../Context/ContextFactory.cuh"

class ConvLayer : public Layer {
public:
	
	Activation activationFunc;

	ConvLayer(std::vector<int> size, Activation activationFunc) :Layer(size) {
		this->activationFunc = activationFunc;
	}
	ConvLayer(std::vector<int> size, Activation activationFunc, std::string name) :Layer(size, name) {
	}
};

#endif // !CONVLAYER