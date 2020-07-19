
#ifndef LAYER
#define LAYER

#include <vector>
#include <string>
#include <cublas_v2.h>
#include "../Context/ContextFactory.cuh"

enum LayerType {
	INPUT,
	DENSE,
	CONV,
	POOL
};

class Layer {
public:
	std::vector<int> size;
	std::string name;

	Layer(std::vector<int> size);
	Layer(std::vector<int> size, std::string name);

	std::vector<int> getSize();
	virtual void init() = 0;
	virtual LayerType getLayerType() = 0;
	virtual void initWeight(const std::vector<float>& weights) = 0;
	virtual void initBias(const std::vector<float>& bias) = 0;
	virtual void forward(ContextFactory contextFactory, std::vector<float>& input_sample) = 0;
};

#endif // !LAYER

