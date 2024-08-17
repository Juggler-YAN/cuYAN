#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>
#include "./slidingwindow.h"

#define N 6
#define IN_C 5
#define IN_H 8
#define IN_W 8
#define K_H 3
#define K_W 3
#define OUT_C 3
#define OUT_H 3
#define OUT_W 3
#define PAD_H 0
#define PAD_W 0
#define STRIDE_H 2
#define STRIDE_W 2
#define DILATION_H 1
#define DILATION_W 1

void rand_data(float *data, int num, float min, float max) {
    for (int i = 0; i < num; i++) {
        data[i] = (fabs(max - min) < 1e-5) ? min : ((max - min) * (rand() / (float)RAND_MAX) + min);
    }
}

int main() {
    // Initialize CUDA
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA initialization failed!" << std::endl;
        return 1;
    }

    // Initialize cuDNN
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    // Define the input tensor descriptor
    cudnnTensorDescriptor_t input_desc;
    cudnnCreateTensorDescriptor(&input_desc);
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, IN_C, IN_H, IN_W);

    // Define the convolution descriptor
    cudnnFilterDescriptor_t filter_desc;
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, OUT_C, IN_C, K_H, K_W);

    // Define the convolution descriptor
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc, PAD_H, PAD_W, STRIDE_H, STRIDE_W, DILATION_H, DILATION_W, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    // Get the output tensor descriptor
    // cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc, &n, &out_c, &out_h, &out_w);

    // Define the output tensor descriptor
    cudnnTensorDescriptor_t output_desc;
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, OUT_C, OUT_H, OUT_W);

    int size_input = N * IN_C * IN_H * IN_W;
    int size_filter = OUT_C * IN_C * K_H * K_W;
    int size_output = N * OUT_C * OUT_H * OUT_W;

    // Allocate host memory for input, filter, and output
    float *h_input, *h_output, *h_filter;
    h_input = (float*)malloc(size_input * sizeof(float));
    h_filter = (float*)malloc(size_filter * sizeof(float));
    h_output = (float*)malloc(size_output * sizeof(float));

    // Initialization
    rand_data(h_input, size_input, -1, 1);
    rand_data(h_filter, size_filter, -1, 1);
    rand_data(h_output, size_output, 0, 0);

    // Allocate device memory for input, filter, and output
    float *d_input, *d_output, *d_filter;
    cudaMalloc((void**)&d_input, size_input * sizeof(float));
    cudaMalloc((void**)&d_filter, size_filter * sizeof(float));
    cudaMalloc((void**)&d_output, size_output * sizeof(float));

    // Memcpy: host -> device
    cudaMemcpy(d_input, h_input, size_input * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, size_filter * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, size_output * sizeof(float), cudaMemcpyHostToDevice);

    // Perform forward convolution
    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(cudnn, &alpha, input_desc, d_input, filter_desc, d_filter, conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, nullptr, 0, &beta, output_desc, d_output);

    // Memcpy: device -> host
    cudaMemcpy(h_output, d_output, size_output * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare
    float *calc_output = (float*)malloc(size_output * sizeof(float));
    slidingwindow(h_input, h_filter, calc_output, N, IN_C, IN_H, IN_W, K_H, K_W, OUT_C, OUT_H, OUT_W, PAD_H, PAD_W, STRIDE_H, STRIDE_W, DILATION_H, DILATION_W);
    float diff = 0.0f;
    for (int i = 0; i < size_output; ++i) {
        diff += (h_output[i] - calc_output[i]);
    }
    printf("\n--------diff:%f------\n", diff);

    // Clean up
    free(h_input);
    free(h_filter);
    free(h_output);
    free(calc_output);
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn);

    return 0;
}
