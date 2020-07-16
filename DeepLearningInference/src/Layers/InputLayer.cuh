
#ifndef INPUTLAYER
#define INPUTLAYER

#include "Layer.cuh"
#include "../Activation/Activation.cuh"
#include "../Context/ContextFactory.cuh"

class InputLayer: public Layer {
public:
	InputLayer::InputLayer(std::vector<int> size);
	InputLayer::InputLayer(std::vector<int> size, std::string name);
	void init();
	void initWeight(const std::vector<float>& weights);
	void initBias(const std::vector<float>& bias);
	void forward(ContextFactory contextFactory, std::vector<float>& input);
};

#endif // !INPUTLAYER