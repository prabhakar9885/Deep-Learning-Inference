#include "Layer.cuh"

Layer::Layer(std::vector<int> size) {
	this->size = size;
	this->name = "unnamed";
}

Layer::Layer(std::vector<int> size, std::string name) {
	this->size = size;
	this->name = name;
}

std::vector<int> Layer::getSize() {
	return this->size;
}