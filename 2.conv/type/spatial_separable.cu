/*
 * spatial separable conv
 * 要求 conv kernel 可以分解成成行向量和列向量的乘积, 例如
 *   -1 0 1     1
 *  (-2 0 2) = (2) * (-1 0 1)
 *   -1 0 1     1
 */

#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>

#define N 1
#define IN_C 1
#define IN_H 4
#define IN_W 4
#define K_H 3
#define K_W 3
#define OUT_C 1
#define OUT_H 2
#define OUT_W 2
#define PAD_H 0
#define PAD_W 0
#define STRIDE_H 1
#define STRIDE_W 1
#define DILATION_H 1
#define DILATION_W 1

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
    cudnnFilterDescriptor_t w1_desc;
    cudnnCreateFilterDescriptor(&w1_desc);
    cudnnSetFilter4dDescriptor(w1_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, OUT_C, IN_C, K_H, 1);
    cudnnFilterDescriptor_t w2_desc;
    cudnnCreateFilterDescriptor(&w2_desc);
    cudnnSetFilter4dDescriptor(w2_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, OUT_C, IN_C, 1, K_W);

    // Define the convolution descriptor
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc, PAD_H, PAD_W, STRIDE_H, STRIDE_W, DILATION_H, DILATION_W, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    // Define the y tensor descriptor
    cudnnTensorDescriptor_t y_desc;
    cudnnCreateTensorDescriptor(&y_desc);
    cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, OUT_C, OUT_H, OUT_W);
    cudnnTensorDescriptor_t y1_desc;
    cudnnCreateTensorDescriptor(&y1_desc);
    cudnnSetTensor4dDescriptor(y1_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, OUT_C, OUT_H, IN_W);
    cudnnTensorDescriptor_t y2_desc;
    cudnnCreateTensorDescriptor(&y2_desc);
    cudnnSetTensor4dDescriptor(y2_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, OUT_C, OUT_H, OUT_W);

    int size_x = N * IN_C * IN_H * IN_W;
    int size_w = OUT_C * IN_C * K_H * K_W;
    int size_w1 = OUT_C * IN_C * K_H * 1;
    int size_w2 = OUT_C * IN_C * 1 * K_W;
    int size_y = N * OUT_C * OUT_H * OUT_W;
    int size_y1 = N * OUT_C * OUT_H * IN_W;
    int size_y2 = N * OUT_C * OUT_H * OUT_W;

    // Allocate host memory for x, w, and y
    float h_x[size_x] = {
        0, 1, 2, 3,
        1, 2, 3, 4,
        2, 3, 4, 5,
        3, 4, 5, 6
    };
    float h_w[size_w] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };
    float h_w1[size_w1] = {
        1,
        2,
        1
    };
    float h_w2[size_w2] = {
        -1, 0, 1
    };
    float *h_y = (float*)malloc(size_y * sizeof(float));
    float *h_y1 = (float*)malloc(size_y1 * sizeof(float));
    float *h_y2 = (float*)malloc(size_y2 * sizeof(float));

    // Allocate device memory for x, w, and y
    float *d_x, *d_w, *d_w1, *d_w2, *d_y, *d_y1, *d_y2;
    cudaMalloc((void**)&d_x, size_x * sizeof(float));
    cudaMalloc((void**)&d_w, size_w * sizeof(float));
    cudaMalloc((void**)&d_w1, size_w1 * sizeof(float));
    cudaMalloc((void**)&d_w2, size_w2 * sizeof(float));
    cudaMalloc((void**)&d_y, size_y * sizeof(float));
    cudaMalloc((void**)&d_y1, size_y1 * sizeof(float));
    cudaMalloc((void**)&d_y2, size_y2 * sizeof(float));

    // Memcpy: host -> device
    cudaMemcpy(d_x, h_x, size_x * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, size_w * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w1, h_w1, size_w1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2, h_w2, size_w2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, size_y * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y1, h_y1, size_y1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y2, h_y2, size_y2 * sizeof(float), cudaMemcpyHostToDevice);

    // Perform forward convolution
    float alpha = 1, beta = 0;
    cudnnConvolutionForward(cudnn, &alpha, x_desc, d_x, w_desc, d_w, conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, nullptr, 0, &beta, y_desc, d_y);

    // Perform forward convolution1
    cudnnConvolutionForward(cudnn, &alpha, x_desc, d_x, w1_desc, d_w1, conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, nullptr, 0, &beta, y1_desc, d_y1);
    // Perform forward convolution2
    cudnnConvolutionForward(cudnn, &alpha, y1_desc, d_y1, w2_desc, d_w2, conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, nullptr, 0, &beta, y2_desc, d_y2);

    // Memcpy: device -> host
    cudaMemcpy(h_y, d_y, size_y * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y2, d_y2, size_y2 * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare
    float diff = 0;
    for (int i = 0; i < size_y; ++i) {
        diff += (h_y[i] - h_y2[i]);
    }
    printf("\n--------diff:%f------\n", diff);

    // Clean up
    // free(h_x);
    // free(h_w);
    // free(h_w1);
    // free(h_w2);
    free(h_y);
    free(h_y1);
    free(h_y2);
    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_w1);
    cudaFree(d_w2);
    cudaFree(d_y);
    cudaFree(d_y1);
    cudaFree(d_y2);
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyFilterDescriptor(w1_desc);
    cudnnDestroyFilterDescriptor(w2_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn);
    
    return 0;
}