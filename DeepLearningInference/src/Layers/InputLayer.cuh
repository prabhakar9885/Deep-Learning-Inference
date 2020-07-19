
#ifndef INPUTLAYER
#define INPUTLAYER

#include "Layer.cuh"
#include "../Activation/Activation.cuh"
#include "../Context/ContextFactory.cuh"

class InputLayer: public Layer {
public:
	InputLayer(int size, std::string name="Input");
	void init();
	LayerType getLayerType();
	void initWeight(const std::vector<float>& weights);
	void initBias(const std::vector<float>& bias);
	void forward(ContextFactory contextFactory, std::vector<float>& input);
};

#endif // !INPUTLAYER