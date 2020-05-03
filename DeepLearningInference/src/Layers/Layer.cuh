
#ifndef LAYER
#define LAYER

#include <vector>
#include <string>
#include <cublas_v2.h>
#include "../Activation/Activation.cuh"
#include "../Context/ContextFactory.cuh"

class Layer {
public:
	int size;
	std::vector<float> value;
	std::vector<std::vector<float>> weights;
	std::vector<float> bias;
	std::string name;
	Activation activationFunc;
	float* activationValue;

	Layer(int size, Activation activationFunc);

	Layer(int size, Activation activationFunc, std::string name);

	void applyActivation();

	void forward(ContextFactory contextFactory, std::vector<float>& input_sample);
};

#endif // !LAYER

