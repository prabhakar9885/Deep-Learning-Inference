
#ifndef DENSELAYER
#define DENSELAYER


#include "Layer.cuh"
#include "../Activation/Activation.cuh"
#include "../Context/ContextFactory.cuh"

class DenseLayer : public Layer {
public:
	std::vector<std::vector<float>> weights;
	std::vector<float> bias;
	Activation activationFunc;

	DenseLayer(int size, Activation activationFunc);
	DenseLayer(int size, Activation activationFunc, std::string name);

	void init();
	void initWeight(const std::vector<float>& weights);
	void initBias(const std::vector<float>& bias);
	void forward(ContextFactory contextFactory, std::vector<float>& input_sample);
};

#endif // !DENSELAYER