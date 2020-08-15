
#ifndef DENSELAYER
#define DENSELAYER


#include "Layer.cuh"
#include "../Activation/Activation.cuh"
#include "../Context/ContextFactory.cuh"

class DenseLayer : public Layer {
protected:
	virtual void* DenseLayer::allocAndInitDataOnDevice(void* inputDataOnHost, int inputElementCount, std::list<Layer*>::iterator layerIterator);

public:
	std::vector<std::vector<float>> weights;
	std::vector<float> bias;
	Activation activationFunc;

	DenseLayer(int size, Activation activationFunc, std::string name="Dense");

	void init();
	LayerType getLayerType();
	void* getOuputOnDevice();
	void initWeight(const std::vector<float>& weights);
	void initBias(const std::vector<float>& bias);
	//void forward(ContextFactory contextFactory, std::vector<float>& input_sample);
	void forward(ContextFactory contextFactory, void* inputSample, int inputElementCount, std::list<Layer*>::iterator layerIterator);
	//virtual void forward(ContextFactory contextFactory, Layer* previousLayer);
};

#endif // !DENSELAYER