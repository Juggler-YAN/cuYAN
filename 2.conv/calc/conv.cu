#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>
// #include "./slidingwindow.h"
// #include "./img2col.h"
#include "./fft.h"

// #define N 6
// #define IN_C 5
// #define IN_H 7
// #define IN_W 7
// #define K_H 3
// #define K_W 3
// #define OUT_C 3
// #define OUT_H 3
// #define OUT_W 3
// #define PAD_H 1
// #define PAD_W 1
// #define STRIDE_H 2
// #define STRIDE_W 2
// #define DILATION_H 2
// #define DILATION_W 2

// fft存在限制，pad=0，stride=1，dilation=1
#define N 6
#define IN_C 5
#define IN_H 7
#define IN_W 7
#define K_H 3
#define K_W 3
#define OUT_C 3
#define OUT_H 5
#define OUT_W 5
#define PAD_H 0
#define PAD_W 0
#define STRIDE_H 1
#define STRIDE_W 1
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

    // Define the x tensor descriptor
    cudnnTensorDescriptor_t x_desc;
    cudnnCreateTensorDescriptor(&x_desc);
    cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, IN_C, IN_H, IN_W);

    // Define the convolution descriptor
    cudnnFilterDescriptor_t w_desc;
    cudnnCreateFilterDescriptor(&w_desc);
    cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, OUT_C, IN_C, K_H, K_W);

    // Define the convolution descriptor
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc, PAD_H, PAD_W, STRIDE_H, STRIDE_W, DILATION_H, DILATION_W, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    // Define the y tensor descriptor
    cudnnTensorDescriptor_t y_desc;
    cudnnCreateTensorDescriptor(&y_desc);
    cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, OUT_C, OUT_H, OUT_W);

    int size_x = N * IN_C * IN_H * IN_W;
    int size_w = OUT_C * IN_C * K_H * K_W;
    int size_y = N * OUT_C * OUT_H * OUT_W;

    // Allocate host memory for x, w, and y
    float *h_x, *h_y, *h_w;
    h_x = (float*)malloc(size_x * sizeof(float));
    h_w = (float*)malloc(size_w * sizeof(float));
    h_y = (float*)malloc(size_y * sizeof(float));

    // Initialization
    rand_data(h_x, size_x, -1, 1);
    rand_data(h_w, size_w, -1, 1);
    rand_data(h_y, size_y, 0, 0);

    // Allocate device memory for x, w, and y
    float *d_x, *d_y, *d_w;
    cudaMalloc((void**)&d_x, size_x * sizeof(float));
    cudaMalloc((void**)&d_w, size_w * sizeof(float));
    cudaMalloc((void**)&d_y, size_y * sizeof(float));

    // Memcpy: host -> device
    cudaMemcpy(d_x, h_x, size_x * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, size_w * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, size_y * sizeof(float), cudaMemcpyHostToDevice);

    // Perform forward convolution
    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(cudnn, &alpha, x_desc, d_x, w_desc, d_w, conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, nullptr, 0, &beta, y_desc, d_y);

    // Memcpy: device -> host
    cudaMemcpy(h_y, d_y, size_y * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare
    float *calc_y = (float*)malloc(size_y * sizeof(float));
    memset(calc_y, 0, size_y * sizeof(float));
    // slidingwindow(h_x, h_w, calc_y, N, IN_C, IN_H, IN_W, K_H, K_W, OUT_C, OUT_H, OUT_W, PAD_H, PAD_W, STRIDE_H, STRIDE_W, DILATION_H, DILATION_W);
    // img2col(h_x, h_w, calc_y, N, IN_C, IN_H, IN_W, K_H, K_W, OUT_C, OUT_H, OUT_W, PAD_H, PAD_W, STRIDE_H, STRIDE_W, DILATION_H, DILATION_W);
    convfft(h_x, h_w, calc_y, N, IN_C, IN_H, IN_W, K_H, K_W, OUT_C, OUT_H, OUT_W, PAD_H, PAD_W, STRIDE_H, STRIDE_W, DILATION_H, DILATION_W);

    float diff = 0.0f;
    for (int i = 0; i < size_y; ++i) {
        diff += (h_y[i] - calc_y[i]);
    }
    printf("\n--------diff:%f------\n", diff);

    // Clean up
    free(h_x);
    free(h_w);
    free(h_y);
    free(calc_y);
    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_y);
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn);

    return 0;
}
