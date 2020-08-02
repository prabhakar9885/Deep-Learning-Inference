
#ifndef CuDnnLayer
#define CuDnnLayer

#include <cudnn.h>
#include<vector>
#include "../Layers/ConvLayer.cuh"
#include "../Layers/PoolingLayer.cuh"


enum CuDnnLayerType {
    CUDNN_INPUT,
    CUDNN_CONV,
    CUDNN_POOLING
};

struct DeviceDataAndDescriptor {
    size_t byteCount{ 0 };
    size_t elementCount{ 0 };
    float* data;
    cudnnTensorDescriptor_t* descriptor{ nullptr };
};


/*
    Holds the current-layer and the filter that operates on it to generate the output. This output will act as an input for the following cudnn/cublas-layer
*/
class CuDNNLayer {
public:
    CuDnnLayerType cuDnnLayerType;
    size_t workspace_bytes;

    cudnnFilterDescriptor_t* kernelDescriptor;
    cudnnPoolingDescriptor_t* poolingDescriptor;

    cudnnTensorDescriptor_t* getTensorDescriptor(MemoryLayout memoryLayout, Layer* layer);
    cudnnFilterDescriptor_t* getKernelDescriptor(MemoryLayout memoryLayout, int inputChannels, ConvLayer* convLayer, int outputChannels);
    cudnnConvolutionDescriptor_t* getConvolutionDescriptor(MemoryLayout memoryLayout, ConvLayer* convLayer);
    cudnnPoolingDescriptor_t* getPoolingDescriptor(PoolingLayer* poolingLayer);
    float* doConvOperation(DeviceDataAndDescriptor* input, Activation activationFunc, DeviceDataAndDescriptor* output);
    float* doPoolingOperation(DeviceDataAndDescriptor* input, DeviceDataAndDescriptor* output);
};

#endif //! CuDnnLayer