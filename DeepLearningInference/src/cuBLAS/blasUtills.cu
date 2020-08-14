#include "blasUtills.cuh"

namespace kernals
{

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
}

void BlasUtils::computeActivation(float*& x, int xSize, Activation activation)
{
	int blockSize = 1024;
	int blockCount = xSize / blockSize + (xSize % blockSize != 0);

	switch (activation)
	{
	case Activation::SIGMOID:
		kernals::sigmoid << < blockCount, blockSize >> > (x, xSize);
		break;
	case Activation::ReLU:
		kernals::relu << < blockCount, blockSize >> > (x, xSize);
		break;
	case Activation::SOFTMAX:
		break;
	default:
		throw "Unidentified Activation type";
	}

	cudaDeviceSynchronize();

	if (PRINT_TRACE)
	{
		std::cout << "\n";
		for (int i = 0; i < xSize; i++) {
			std::cout << x[i] << " ";
		}
		std::cout << "\n\n";
	}
}

void BlasUtils::computeActivation(std::vector<float>& x, Activation activation)
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
		kernals::sigmoid << < blockCount, blockSize >> > (arr, N);
		break;
	case Activation::ReLU:
		kernals::relu << < blockCount, blockSize >> > (arr, N);
		break;
	case Activation::SOFTMAX:
		break;
	default:
		throw "Unidentified Activation type";
	}

	cudaDeviceSynchronize();

	if (PRINT_TRACE)
		std::cout << "\n";
	for (int i = 0; i < N; i++) {
		x[i] = arr[i];
		if (PRINT_TRACE)
			std::cout << x[i] << " ";
	}
	if (PRINT_TRACE)
		std::cout << "\n\n";

	cudaFree(arr);
}

void BlasUtils::axpby_vector_matrix(ContextFactory contextFactory,
	float* x, int xSize, std::vector<std::vector<float>> wt, std::vector<float>& bias,
	float*& outputOfCurrentLayer, int& outputSizeForCurrentLayer)
{
	cublasHandle_t* handle = contextFactory.getContext().getCublasHandle();
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
	for (int i = 0, Ai = 0; i < m; i++)
	{
		C[i] = bias[i];
		for (int j = 0; j < n; j++)
		{
			W[i * n + j] = wt[i][j];
			Ai++;
		}
	}

	// Copy bias from host to device
	for (size_t i = 0; i < xSize; i++)
	{
		X[i] = x[i];
	}


	cublasSgemv(*handle, CUBLAS_OP_T, n, m, &alpha, W, n, X, 1, &beta, C, 1);
	cudaDeviceSynchronize();

	if (PRINT_TRACE)
	{
		std::cout << "\n";
		for (size_t i = 0; i < m; i++)
		{
			for (size_t j = 0; j < n; j++)
			{
				std::cout << wt[i][j] << " ";
			}
			std::cout << "\n";
		}
		std::cout << "\n";

		for (size_t i = 0; i < n; i++)
		{
			std::cout << x[i] << "\n";
		}
		std::cout << "\n";

		for (size_t i = 0; i < m; i++)
		{
			std::cout << C[i] << "\n";
		}
	}

	cudaFree(W);
	cudaFree(X);

	outputOfCurrentLayer = C;
	outputSizeForCurrentLayer = m;
}

void BlasUtils::axpby_vector_matrix(ContextFactory contextFactory, std::vector<std::vector<float>> wt, std::vector<float>& x, std::vector<float>& bias) 
{
	cublasHandle_t* handle = contextFactory.getContext().getCublasHandle();
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
		C[i] = bias[i];
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


	cublasSgemv(*handle, CUBLAS_OP_T, n, m, &alpha, W, n, X, 1, &beta, C, 1);
	cudaDeviceSynchronize();

	if (PRINT_TRACE)
	{
		std::cout << "\n";
		for (size_t i = 0; i < m; i++)
		{
			for (size_t j = 0; j < n; j++)
			{
				std::cout << wt[i][j] << " ";
			}
			std::cout << "\n";
		}
		std::cout << "\n";

		for (size_t i = 0; i < n; i++)
		{
			std::cout << x[i] << "\n";
		}
		std::cout << "\n";
	}

	x = std::vector<float>();
	for (size_t i = 0; i < m; i++)
	{
		if (PRINT_TRACE)
		{
			std::cout << C[i] << "\n";
		}
		x.push_back(C[i]);
	}

	cudaFree(W);
	cudaFree(X);
	cudaFree(C);
}

void BlasUtils::axpb_c(std::vector<float> a, std::vector<float>& x, float b)
{
	for (size_t i = 0; i < x.size(); i++)
	{
		x[i] = a[i] * x[i] + b;
	}
}