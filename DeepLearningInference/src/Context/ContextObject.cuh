
#ifndef ContextObject_CUH
#define ContextObject_CUH


#include <cublas_v2.h>
#include <cudnn.h>
#include <string>
class ContextObject
{
private:
	cublasHandle_t* cublas_handle = nullptr;
	cudnnHandle_t* cudnn_handle = nullptr;
public:
	cublasHandle_t* getCublasHandle();
	cudnnHandle_t* getCudnnHandle();
	void releaseCublasHandle();
	void releaseCudnnHandle();
};


#endif // !ContextObject_CUH