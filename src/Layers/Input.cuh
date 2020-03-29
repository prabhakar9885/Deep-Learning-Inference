
#ifndef INPUT


#include "Layer.cuh"

class Input : public Layer {
public:
	Input(int size) :Layer(size, nullptr) {
		cout << "Created input layer\n";
	}

	Input(int size, string name) :Layer(size, nullptr, name) {
	}
};


#endif // !INPUT