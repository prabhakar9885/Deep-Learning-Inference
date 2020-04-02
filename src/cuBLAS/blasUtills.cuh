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


	__global__ void sigmoid(float* inp, int N) {
		int indx = blockDim.x * blockIdx.x + threadIdx.x;
		if (indx < N)
			inp[indx] = 1 / (1 + exp(-inp[indx]));
	}

	__global__ void relu(float* inp, int N) {
		int indx = blockDim.x * blockIdx.x + threadIdx.x;
		if (indx < N)
			inp[indx] = 0 > inp[indx] ? 0 : inp[indx];
	}


	void axpb_vector_matrix(vector<vector<float>> wt, vector<float>& x, float b) {

		size_t m = wt.size();
		size_t n = wt[0].size();
		float* W; // m x n
		float* X; // n x 1
		float* C; // m x 1
		float alpha = 1, beta = 1;

		cudaMallocManaged((void**)&W, m * n * sizeof(float));
		cudaMallocManaged((void**)&X, n * 1 * sizeof(float));
		cudaMallocManaged((void**)&C, m * 1 * sizeof(float));

		// Copy weight from host to device
		for (int i = 0, Ai = 0; i < m;i++)
		{
			C[i] = 0;
			for (int j = 0; j < n;j++)
			{
				W[i * n + j] = wt[i][j];
				Ai++;
			}
		}

		// Copy bias from host to device
		for (size_t i = 0; i < x.size(); i++)
		{
			X[i] = x[i];
		}

		cublasHandle_t handle;
		cublasCreate(&handle);
		cublasSgemv(handle, CUBLAS_OP_T, n, m, &alpha, W, n, X, 1, &beta, C, 1);
		cudaDeviceSynchronize();

		//cout << "\n";
		//for (size_t i = 0; i < m; i++)
		//{
		//	for (size_t j = 0; j < n; j++)
		//	{
		//		cout << wt[i][j] << " ";
		//	}
		//	//cout << "\n";
		//}
		//cout << "\n";

		//for (size_t i = 0; i < n; i++)
		//{
		//	cout << x[i] << "\n";
		//}
		//cout << "\n";

		x = vector<float>();
		for (size_t i = 0; i < m; i++)
		{
			//cout << C[i] << "\n";
			x.push_back(C[i]);
		}

		cublasDestroy(handle);
		cudaFree(W);
		cudaFree(X);
		cudaFree(C);
	}

	void computeActivation(vector<float>& x, Activation activation)
	{
		float* arr;
		float* res;
		int N = x.size();

		cudaMallocManaged(&arr, N * sizeof(float));

		for (int i = 0; i < N; i++)
			arr[i] = x[i];

		int blockSize = 1024;
		int blockCount = N / blockSize + (N % blockSize != 0);

		switch (activation)
		{
		case Activation::SIGMOID:
			sigmoid << < blockCount, blockSize >> > (arr, N);
			break;
		case Activation::ReLU:
			relu << < blockCount, blockSize >> > (arr, N);
			break;
		case Activation::SOFTMAX:
			break;
		default:
			throw "Unidentified Activation type";
		}

		cudaDeviceSynchronize();

		//cout << "\n";
		for (int i = 0; i < N; i++) {
			x[i] = arr[i];
			//cout << x[i] << " ";
		}
		//cout << "\n\n";

		cudaFree(arr);
	}
}
#endif // MathOps
