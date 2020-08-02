#include "./ContextObject.cuh"

ContextObject::ContextObject()
{
	cublas_handle = nullptr;
	cudnn_handle = nullptr;
}

cublasHandle_t* ContextObject::getCublasHandle()
{
	if (this->cublas_handle == nullptr)
	{
		this->cublas_handle = new cublasHandle_t();
		cublasStatus_t status;
		if ((status = cublasCreate_v2(this->cublas_handle)) != CUBLAS_STATUS_SUCCESS)
		{
			throw "cuBLAS initialization Failed. Status code: " + std::to_string(status) + ".";
		}
	}
	return this->cublas_handle;
}


cudnnHandle_t* ContextObject::getCudnnHandle()
{
	if (this->cudnn_handle == nullptr)
	{
		this->cudnn_handle = new cudnnHandle_t();
		cudnnStatus_t status;
		if ((status = cudnnCreate(this->cudnn_handle)) != CUDNN_STATUS_SUCCESS)
		{
			throw "cuDNN initialization Failed. Status code: " + std::to_string(status) + ".";
		}
	}
	return this->cudnn_handle;
}


void ContextObject::releaseCublasHandle()
{
	if (cublas_handle == nullptr)
		return;

	cublasStatus_t status = cublasDestroy(*cublas_handle);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		throw "Failed to release the cuBLAS handle. Status code: " + std::to_string(status) + ".";
	}
	free(this->cublas_handle);
}


void ContextObject::releaseCudnnHandle()
{
	if (cudnn_handle == nullptr)
		return;

	cudnnStatus_t status = cudnnDestroy(*cudnn_handle);
	if (status != CUDNN_STATUS_SUCCESS)
	{
		throw "Failed to release the cuDNN handle. Status code: " + std::to_string(status) + ".";
	}
	free(this->cudnn_handle);
}