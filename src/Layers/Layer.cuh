
#ifndef LAYER
#define LAYER

#include <vector>
#include <string>
#include "../Activation/Activation.cuh"

class Layer {
public:
	int size;
	std::vector<float> value;
	std::string name;
	Activation activationFunc;
	float* activationValue;

	Layer(int size, Activation activationFunc);

	Layer(int size, Activation activationFunc, std::string name);

	void applyActivation();
};

#endif // !LAYER

