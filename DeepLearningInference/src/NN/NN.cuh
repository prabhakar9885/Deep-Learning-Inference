
#ifndef NN_CUH
#define NN_CUH


#include <iostream>
#include "cuda.h"
#include "cuda_runtime_api.h"
#include <vector>
#include <list>
#include "../Layers/Layer.cuh"
#include "../Activation/Activation.cuh"
#include "../Layers/ConvLayer.cuh"
#include "../Layers/PoolingLayer.cuh"
#include "../Layers/DenseLayer.cuh"
#include "../Layers/InputLayer.cuh"
#include "../cuBLAS/blasUtills.cuh"
#include "../Context/ContextFactory.cuh"


class NN {
private:
	void forward(Layer* previousLayer, Layer* currentLayer);
public:
	std::list<Layer*> layers;
	ContextFactory contextFactory;

	NN(ContextFactory contextFactory);

	void pushLayer(Layer* layer);

	void init(std::list<std::vector<float>> weightsAndBias);

	float forward(std::vector<float>& input_sample);
};


#endif // !NN_CUH