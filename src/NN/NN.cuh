
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


class NN {
public:
	std::vector<Layer> layers;
	std::vector<std::vector<std::vector<float>>> weights;
	std::vector<std::vector<float>> bias;
	cublasHandle_t handle;

	NN(cublasHandle_t handle);

	void pushLayer(Layer& layer);

	void init(std::vector<std::vector<std::vector<float>>> weights);

	float forword(std::vector<float>& input_sample);
};


#endif // !NN_CUH