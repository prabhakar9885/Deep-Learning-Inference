
#ifndef LAYER
#define LAYER

#include <algorithm>
#include <numeric> 
#include <list>
#include <vector>
#include <string>
#include <cublas_v2.h>
#include <iostream>
#include "../Context/ContextFactory.cuh"

enum class LayerType {
	INPUT,
	DENSE,
	CONV,
	POOL
};

enum class MemoryLayout {
	NHWC = CUDNN_TENSOR_NHWC,
	NCHW = CUDNN_TENSOR_NCHW
};

class Layer {
protected:
	void* outputOfCurrentLayer;
	std::vector<int> size;
	std::string name;

	virtual void* allocAndInitDataOnDevice(void* inputDataOnHost, int inputElementCount, std::list<Layer*>::iterator layerIterator) = 0;

public:
	Layer(std::vector<int> size);
	Layer(std::vector<int> size, std::string name);
	std::string getName();
	std::vector<int> getSize();
	virtual LayerType getLayerType() = 0;
	virtual void init() = 0;
	virtual void initWeight(const std::vector<float>& weights) = 0;
	virtual void initBias(const std::vector<float>& bias) = 0;
	virtual void forward(ContextFactory contextFactory, void* inputSample, int inputElementCount, std::list<Layer*>::iterator layerIterator) = 0;
	virtual void* getOuputOnDevice() = 0;
	void* getOuputToHost();
};

#endif // !LAYER

