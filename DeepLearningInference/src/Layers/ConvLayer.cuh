
#ifndef CONVLAYER
#define CONVLAYER

#include "Layer.cuh"
#include "../Activation/Activation.cuh"
#include "../Context/ContextFactory.cuh"


class ConvLayer : public Layer {
public:
	std::vector<std::vector<float>> weights;
	std::vector<float> bias;
	Activation activationFunc;
	int padding = 0;
	int stride = 0;
	int dilation = 0;

	ConvLayer(std::vector<int> size, Activation activationFunc, std::string name="ConvLayer", int padding = 0, int stride = 1, int dilation = 1);

	void init();
	LayerType getLayerType();
	void initWeight(const std::vector<float>& weights);
	void initBias(const std::vector<float>& bias);
	void forward(ContextFactory contextFactory, std::vector<float>& input_sample);

};

#endif // !CONVLAYER