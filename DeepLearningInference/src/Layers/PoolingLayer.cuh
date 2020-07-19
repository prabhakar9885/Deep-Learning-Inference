
#ifndef POOLINGLAYER
#define POOLINGLAYER

#include "Layer.cuh"
#include "../Activation/Activation.cuh"
#include "../Context/ContextFactory.cuh"


class PoolingLayer : public Layer {
public:
	int padding = 0;
	int stride = 0;

	PoolingLayer(std::vector<int> size, std::string name="Pooling", int padding = 0, int stride = 1);

	void init();
	LayerType getLayerType();
	void initWeight(const std::vector<float>& weights);
	void initBias(const std::vector<float>& bias);
	void forward(ContextFactory contextFactory, std::vector<float>& input_sample);

};

#endif // !POOLINGLAYER