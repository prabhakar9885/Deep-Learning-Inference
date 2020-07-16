
#include "../../DeepLearningInference/src/NN/NN.cuh"
#include "../../DeepLearningInference/src/Context/ContextFactory.cuh"
#include "../../DeepLearningInference/src/Context/ContextObject.cuh"
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <chrono> 


std::vector<std::vector<float>> parse_and_load_data(std::string fileName);