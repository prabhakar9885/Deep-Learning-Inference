
#ifndef NN_CUH
#define NN_CUH


#include <iostream>
#include "cuda.h"
#include "cuda_runtime_api.h"
#include <vector>
#include "../Layers/Layer.cuh"
#include "../Activation/Activation.cuh"
#include "../Layers/DenseLayer.cuh"
#include "../cuBLAS/blasUtills.cuh"
#include "../Context/ContextFactory.cuh"


class NN {
public:
	std::vector<Layer> layers;
	ContextFactory contextFactory;

	NN(ContextFactory contextFactory);

	void pushLayer(Layer& layer);

	void init(std::vector<std::vector<std::vector<float>>> weights);

	float forward(std::vector<float>& input_sample);
};


#endif // !NN_CUH