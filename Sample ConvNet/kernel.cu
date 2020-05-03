
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda.h> // need CUDA_VERSION
#include <cudnn.h>

#include <stdio.h>


int main()
{
    int version = (int)cudnnGetVersion();
    printf("cudnnGetVersion() : %d , CUDNN_VERSION from cudnn.h : %d \n", version, CUDNN_VERSION);

}
