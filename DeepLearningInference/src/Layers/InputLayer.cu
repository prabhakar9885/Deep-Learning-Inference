#include "InputLayer.cuh"

InputLayer::InputLayer(int size, std::string name) : Layer(std::vector<int>(1, size), name) {
}

LayerType InputLayer::getLayerType() {
    return LayerType::INPUT;
}

void InputLayer::init() {
}

void InputLayer::initWeight(const std::vector<float>& weights) {
}

void InputLayer::initBias(const std::vector<float>& bias) {
}

void InputLayer::forward(ContextFactory contextFactory, std::vector<float>& input) {
}