
#include "../../DeepLearningInference/src/NN/NN.cuh"
#include "../../DeepLearningInference/src/Context/ContextFactory.cuh"
#include "../../DeepLearningInference/src/Context/ContextObject.cuh"
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <chrono> 


std::unordered_map<std::string, std::vector<float>> parse_and_load_data(std::string fileName);


std::vector<std::vector<std::vector<float>>> get_weights(std::vector<int>& layers_dims, std::unordered_map<std::string, std::vector<float>> weights_data);