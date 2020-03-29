
#ifndef INPUT


#include "Layer.cuh"

class Input : public Layer {
public:
	Input(int size) :Layer(size, Activation::IDENTITY) {
		cout << "Created input layer\n";
	}

	Input(int size, string name) :Layer(size, Activation::IDENTITY, name) {
	}
};


#endif // !INPUT