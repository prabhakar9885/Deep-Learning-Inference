
#ifndef INPUTLAYER
#define INPUTLAYER

#include "Layer.cuh"
#include "../Activation/Activation.cuh"
#include "../Context/ContextFactory.cuh"

class InputLayer: public Layer {
private:
	void* allocAndInitDataOnDevice(void* inputDataOnHost, int inputElementCount, std::list<Layer*>::iterator layerIterator);

public:
	InputLayer(int size, std::string name="Input");
	LayerType getLayerType();
	void init();
	void initWeight(const std::vector<float>& weights);
	void initBias(const std::vector<float>& bias);
	void forward(ContextFactory contextFactory, void* inputSample, int inputElementCount, std::list<Layer*>::iterator layerIterator);
	void* getOuputOnDevice();
};

#endif // !INPUTLAYER