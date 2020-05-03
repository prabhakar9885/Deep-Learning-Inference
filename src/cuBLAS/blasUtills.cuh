
#ifndef BlasUtils_CUH
#define BlasUtils_CUH

#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <device_launch_parameters.h>
#include "../Activation/Activation.cuh"
#include "../Context/ContextFactory.cuh"

//using namespace std;

class BlasUtils
{
public:

	static const bool PRINT_TRACE = false;

	static void axpb_c(std::vector<float> a, std::vector<float>& x, float b);

	static void axpby_vector_matrix(ContextFactory contextFactory, std::vector<std::vector<float>> wt, std::vector<float>& x, std::vector<float>& bias);

	static void computeActivation( std::vector<float>& x, Activation activation);
};
#endif // BlasUtils_CUH
