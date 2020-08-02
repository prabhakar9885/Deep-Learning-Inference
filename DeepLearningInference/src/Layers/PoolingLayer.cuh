
#ifndef POOLINGLAYER
#define POOLINGLAYER

#include "Layer.cuh"
#include "../Activation/Activation.cuh"
#include "../Context/ContextFactory.cuh"

enum PoolingType {
	MAX_POOLING = CUDNN_POOLING_MAX,
	AVERAGE_COUNT_EXCLUDE_PADDING = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
	AVERAGE_COUNT_INCLUDE_PADDING = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
};


class PoolingLayer : public Layer {
public:
	int padding = 0;
	int stride = 0;
	PoolingType poolingType;

	PoolingLayer(std::vector<int> size, std::string name = "Pooling", PoolingType poolingType = PoolingType::MAX_POOLING, int padding = 0, int stride = 1);

	void init();
	LayerType getLayerType();
	void initWeight(const std::vector<float>& weights);
	void initBias(const std::vector<float>& bias);
	void forward(ContextFactory contextFactory, std::vector<float>& input_sample);

};

#endif // !POOLINGLAYER