// This program shows off some basic cuBLAS examples
// By: Nick from CoffeeBeforeArch

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <algorithm> 
#include "iostream"
#include "iomanip"

using namespace std;

#pragma region Error handling code

inline cudaError_t checkCuda(cudaError_t result)
{
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
	return result;
}

#pragma endregion


#pragma region Auxilary functions for Matrix and vectors

int IDX2C(int i, int j, int ld)
{
	return j * ld + i;
}

void vector_init(float* a, int startValue, int n)
{
	for (int i = 0; i < n; i++)
	{
		a[i] = startValue++;
	}
}

void matrix_init(float* mat, int startValue, int m, int n, int ld)
{
	int val = startValue;
	for (int j = 0; j < n; j++)	// Columns
	{
		for (int i = 0; i < m; i++)  // Rows
		{
			mat[IDX2C(i, j, ld)] = val;
			val++;
		}
	}
}

void matrix_init_all(float* mat, int value, int m, int n, int ld)
{
	for (int j = 0; j < n; j++)	// Columns
	{
		for (int i = 0; i < m; i++)  // Rows
		{
			mat[IDX2C(i, j, ld)] = value;
		}
	}
}

void display_matrix(float* mat, int m, int n, int ld)
{
	for (int i = 0; i < m; i++) // Rows
	{
		for (int j = 0; j < n; j++) // Columns
		{
			cout << mat[IDX2C(i, j, ld)] << std::setw(5);
		}
		cout << "\n";
	}
	cout << "\n";
}

void display_vector(float* vec, int n) {
	for (int i = 0; i < n; i++)
		cout << vec[i] << std::setw(3);
}

#pragma endregion


#pragma region Kernals

__global__ void sigmoid(float* inp, float* res, int N) {
	int indx = blockDim.x * blockIdx.x + threadIdx.x;
	if (indx < N)
		res[indx] = 1 / (1 + exp(-inp[indx]));
}


#pragma endregion


#pragma region Linear algebra functions

void mat_mul() {
	cout << "Mat mul" << "\n" << "===============" << "\n";

	int p = 6, q = 4, r = 2;
	float* mat1; //  Human rep: p x q
	float* mat2; //  Human rep: q x r
	float* mat_prod;
	float* tmp;
	cudaError_t err;

	err = cudaMallocManaged((void**)&mat1, sizeof(float) * p * q);
	checkCuda(err);
	err = cudaMallocManaged((void**)&mat2, sizeof(float) * q * r);
	checkCuda(err);
	err = cudaMallocManaged((void**)&mat_prod, sizeof(float) * p * r);
	checkCuda(err);

	matrix_init(mat1, 0, p, q, p);
	matrix_init(mat2, 0, q, r, q);

	// Scalaing factors
	float alpha = 1.0f;
	float beta = 0.0f;

	// Create and initialize a new context
	cublasHandle_t handle;
	cublasStatus_t status;
	cublasCreate_v2(&handle);

	status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, p, r, q, &alpha, mat1, p, mat2, q, &beta, mat_prod, p);

	cudaDeviceSynchronize();

	display_matrix(mat1, p, q, p);
	display_matrix(mat2, q, r, q);
	display_matrix(mat_prod, r, p, r);

	// Clean up the created handle
	cublasDestroy(handle);

	// Release allocated memory
	cudaFree(mat1);
	cudaFree(mat2);
	cudaFree(mat_prod);

	cout << "\n\n";
}

void axpy() {

	cout << "a*x + y" << "\n";
	cout << "=========================" << "\n\n";

	int p = 6;
	float* x;
	float* y;
	float alpha = 2.0f;
	cudaError_t err;

	err = cudaMallocManaged((void**)&x, sizeof(float) * p);
	checkCuda(err);
	err = cudaMallocManaged((void**)&y, sizeof(float) * p);
	checkCuda(err);

	vector_init(x, 0, p);
	vector_init(y, 2, p);
	cout << "x : ";	display_vector(x, p); cout << "\n";
	cout << "y : ";	display_vector(y, p); cout << "\n";

	cublasHandle_t handle;
	cublasCreate_v2(&handle);
	cublasStatus_t status = cublasSaxpy(handle, p, &alpha, x, 1, y, 1);
	cudaDeviceSynchronize();

	cout << "y : ";	display_vector(y, p); cout << "\n";

	cublasDestroy(handle);
	cudaFree(x);
	cudaFree(y);

	cout << "=========================" << "\n\n";
}

void map() {
	float* arr;
	float* res;
	int N = 10;

	cudaMallocManaged(&arr, N * sizeof(float));
	cudaMallocManaged(&res, N * sizeof(float));

	for (int i = 0; i < N; i++)
		arr[i] = N / 2 - i;

	int blockSize = 8;
	int blockCount = N / blockSize + (N % blockSize != 0);

	sigmoid << < blockCount, blockSize >> > (arr, res, N);
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

void axpb_vector_matrix() {

	size_t m = 5, n = 3;
	float* A; // m x n
	float* X; // n x 1
	float b = 1; // scalar
	float* C; // m x 1
	float alpha = 1;

	cudaMallocManaged(&A, m * n * sizeof(float));
	cudaMallocManaged(&X, n * 1 * sizeof(float));
	cudaMallocManaged(&C, m * 1 * sizeof(float));

	matrix_init(A, 0, m, n, m);
	matrix_init(X, 0, n, 1, n);
	matrix_init_all(C, 1, m, 1, m);

	display_matrix(A, m, n, m);
	display_matrix(X, n, 1, n);
	display_matrix(C, m, 1, m);

	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, 1, n, &alpha, A, m, X, n, &b, C, m);
	cudaDeviceSynchronize();

	display_matrix(C, m, 1, m);

	cublasDestroy(handle);
	cudaFree(A);
	cudaFree(X);
	cudaFree(C);
}

void transpose_mat() {

	cout << "Transpose of a matrix\n";
	cout << "=========================" << "\n";
	float alpha = 1;
	float beta = 0;
	float* mat;
	float* res;
	size_t r = 3, c = 2;

	cudaMallocManaged(&mat, r * c * sizeof(float));
	cudaMallocManaged(&res, c * r * sizeof(float));

	matrix_init(mat, 0, r, c, r);
	display_matrix(mat, r, c, r);

	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, c, r, &alpha, mat, r, &beta, mat, r, res, c);
	cudaDeviceSynchronize();
	display_matrix(res, c, r, c);

	cublasDestroy(handle);
	cudaFree(mat);
	cudaFree(res);
}

#pragma endregion


/**
 *	https://www.adityaagrawal.net/blog/deep_learning/row_column_major
 */
void mat_mul_2() {
	cout << "Mat mul" << "\n" << "===============" << "\n";

	int p = 6, q = 4, r = 2;
	float* mat1; //  Human rep: p x q
	float* mat2; //  Human rep: q x r
	float* mat_prod;
	float* mat_prod_T;
	float* tmp;
	cudaError_t err;

	err = cudaMallocManaged((void**)&mat1, sizeof(float) * p * q);
	checkCuda(err);
	err = cudaMallocManaged((void**)&mat2, sizeof(float) * q * r);
	checkCuda(err);
	err = cudaMallocManaged((void**)&mat_prod, sizeof(float) * p * r);
	checkCuda(err);
	err = cudaMallocManaged((void**)&mat_prod_T, sizeof(float) * p * r);
	checkCuda(err);

	// Init mat1
	int val = 0;
	for (size_t i = 0; i < p; i++)
	{
		for (size_t j = 0; j < q; j++)
		{
			mat1[i * q + j] = val;
			val = (val + 1) % 5;
		}
	}

	for (size_t i = 0; i < p; i++)
	{
		for (size_t j = 0; j < q; j++)
		{
			cout << mat1[i * q + j] << std::setw(5);
		}
		cout << "\n";
	}
	cout << "\n";

	// Init mat2
	val = 0;
	for (size_t i = 0; i < q; i++)
	{
		for (size_t j = 0; j < r; j++)
		{
			mat2[i * r + j] = val;
			val = (val + 1) % 5;
		}
	}
	for (size_t i = 0; i < q; i++)
	{
		for (size_t j = 0; j < r; j++)
		{
			cout << mat2[i * r + j] << std::setw(5);
		}
		cout << "\n";
	}
	cout << "\n";

	// Scalaing factors
	float alpha = 1.0f;
	float beta = 0.0f;

	// Create and initialize a new context
	cublasHandle_t handle;
	cublasStatus_t status;
	cublasCreate_v2(&handle);

	//	CMO => mat_prod := alpha * mat2.T * mat1.T + beta * mat_prod.T
	//					:= mat2.T * mat1.T
	status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, r, p, q, &alpha, mat2, r, mat1, q, &beta, mat_prod, r);
	cudaDeviceSynchronize();
	for (size_t i = 0; i < r; i++)
	{
		for (size_t j = 0; j < p; j++)
		{
			cout << mat_prod[i * p + j] << std::setw(5);
		}
		cout << "\n";
	}
	cout << "\n";

	// CMO	=>	mat_prod_T	:= alpha * mat_prod.T  +  beta * mat_prod.T
	//						:= mat_prod.T
	status = cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, r, p, &alpha, mat_prod, p, &beta, mat_prod, p, mat_prod_T, r);
	cudaDeviceSynchronize();
	for (size_t i = 0; i < p; i++)
	{
		for (size_t j = 0; j < r; j++)
		{
			cout << mat_prod_T[i * r + j] << std::setw(5);
		}
		cout << "\n";
	}


	// Clean up the created handle
	cublasDestroy(handle);

	// Release allocated memory
	cudaFree(mat1);
	cudaFree(mat2);
	cudaFree(mat_prod);
	cudaFree(mat_prod_T);

	cout << "\n\n";
}


/**
 * Implemented in RMO
 * C = AX  ;   A(mxn) ; X(nx1)
 * https://www.adityaagrawal.net/blog/deep_learning/row_column_major
 * https://stackoverflow.com/questions/14595750/transpose-matrix-multiplication-in-cublas-howto
 */
void axpb_vector_matrix_() {

	size_t m = 5, n = 3;
	float* A; // m x n
	float* X; // n x 1
	float b = 1; // scalar
	float* C; // m x 1
	float alpha = 1, beta = 0;

	cudaMallocManaged(&A, m * n * sizeof(float));
	cudaMallocManaged(&X, n * 1 * sizeof(float));
	cudaMallocManaged(&C, m * 1 * sizeof(float));

	// Init A
	int val = 0;
	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			A[i * n + j] = val;
			val = (val + 1) % 8;
		}
	}
	cout << "\n";
	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			cout << A[i * n + j] << std::setw(5);
		}
		cout << "\n";
	}

	// Init X
	val = 1;
	for (size_t i = 0; i < n; i++)
	{
		X[i] = val;
		val = (val + 1);
	}
	cout << "\n";
	for (size_t i = 0; i < n; i++)
	{
		cout << X[i] << "\n";
	}

	// Init C
	for (size_t i = 0; i < m; i++)
	{
		C[i] = 0;
	}
	cout << "\n";
	for (size_t i = 0; i < m; i++)
	{
		cout << C[i] << "\n";
	}

	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgemv(handle, CUBLAS_OP_T, n, m, &alpha, A, n, X, 1, &beta, C, 1 );
	cudaDeviceSynchronize();

	cout << "\n";
	for (size_t i = 0; i < m; i++)
	{
		cout << C[i] << "\n";
	}

	cublasDestroy(handle);
	cudaFree(A);
	cudaFree(X);
	cudaFree(C);
}


int main_1() {
	//mat_mul();
	//mat_mul_2();
	//axpy();
	//axpb_vector_matrix();
	axpb_vector_matrix_();
	//map();
	//transpose_mat();
	return 0;
}