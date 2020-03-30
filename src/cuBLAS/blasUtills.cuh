#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <device_launch_parameters.h>

#ifndef MathOps
#define MathOps

using namespace std;

namespace utils
{
	void axpb_c(vector<float> a, vector<float>& x, float b)
	{
		for (size_t i = 0; i < x.size(); i++)
		{
			x[i] = a[i] * x[i] + b;
		}
	}


	__global__ void sigmoid(float* inp, float* res, int N) {
		int indx = blockDim.x * blockIdx.x + threadIdx.x;
		if (indx < N)
			res[indx] = 1 / (1 + exp(-inp[indx]));
	}


	void axpb_vector_matrix(vector<vector<float>> a, vector<float>& x, float b) {
		
		return;
		
		size_t m = 5, n = 3;
		float* A; // m x n
		float* X; // n x 1
		float* C; // m x 1
		float alpha = 1;

		cudaMallocManaged((void**)&A, m * n * sizeof(float));
		cudaMallocManaged((void**)&X, n * 1 * sizeof(float));
		cudaMallocManaged((void**)&C, m * 1 * sizeof(float));

		cublasHandle_t handle;
		cublasCreate(&handle);
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, 1, n, &alpha, A, m, X, n, &b, C, m);
		cudaDeviceSynchronize();

		cublasDestroy(handle);
		cudaFree(A);
		cudaFree(X);
		cudaFree(C);
	}

	void computeActivation(vector<float>& x, Activation activation)
	{

		return;

		float* arr;
		float* res;
		int N = 10;

		cudaMallocManaged(&arr, N * sizeof(float));
		cudaMallocManaged(&res, N * sizeof(float));

		for (int i = 0; i < N; i++)
			arr[i] = N / 2 - i;

		int blockSize = 8;
		int blockCount = N / blockSize + (N % blockSize != 0);

		switch (activation)
		{
		case Activation::SIGMOID:
			sigmoid << < blockCount, blockSize >> > (arr, res, N);
			break;
		case Activation::ReLU:
			break;
		case Activation::SOFTMAX:
			break;
		default:
			throw "Unidentified Activation type";
		}

		
		cudaDeviceSynchronize();

		cout << "First 5 elements\n";
		for (int i = 0; i < min(5, N); i++)
			cout << "sigmoid(" << arr[i] << ") : " << res[i] << "\n";

		cout << "Last 5 elements\n";
		for (int i = max(N - 5, 0); i < max(5, N); i++)
			cout << "sigmoid(" << arr[i] << ") : " << res[i] << "\n";

		cudaFree(arr);
		cudaFree(res);

	}
}
#endif // MathOps
