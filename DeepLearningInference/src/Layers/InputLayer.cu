#include "InputLayer.cuh"

InputLayer::InputLayer(std::vector<int> size): Layer(size) {
}

InputLayer::InputLayer(std::vector<int> size, std::string name) : Layer(size, name) {
}

void InputLayer::init() {
}

void InputLayer::initWeight(const std::vector<float>& weights) {
}

void InputLayer::initBias(const std::vector<float>& bias) {
}

void InputLayer::forward(ContextFactory contextFactory, std::vector<float>& input) {
}