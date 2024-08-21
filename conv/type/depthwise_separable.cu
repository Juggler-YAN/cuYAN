/*
 * depthwise separable conv
 * 分为深度 conv 和 逐点 conv 两部分
 */

#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>
#include "../calc/slidingwindow.h"

#define N 6
#define IN_C 8
#define IN_H 7
#define IN_W 7
#define K_H 3
#define K_W 3
#define OUT_C 16
#define OUT_H 3
#define OUT_W 3
#define PAD_H 1
#define PAD_W 1
#define STRIDE_H 2
#define STRIDE_W 2
#define DILATION_H 2
#define DILATION_W 2
#define GROUP IN_C

void rand_data(float *data, int num, float min, float max) {
    for (int i = 0; i < num; i++) {
        data[i] = (fabs(max - min) < 1e-5) ? min : ((max - min) * (rand() / (float)RAND_MAX) + min);
    }
}

void groupconv(const float* x, const float* w, float* y) {
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < GROUP; ++g) {
            int xg_size = 1 * IN_C / GROUP * IN_H * IN_W;
            int wg_size = OUT_C / GROUP * IN_C / GROUP * K_H * K_W;
            int yg_size = 1 * OUT_C / GROUP * OUT_H * OUT_W;
            float* xg = (float *)malloc(xg_size * sizeof(float));
            float* wg = (float *)malloc(wg_size * sizeof(float));
            float* yg = (float *)malloc(yg_size * sizeof(float));
            // 第 g 组对应的 x
            memcpy(xg, x + IX(n, g * IN_C / GROUP, 0, 0), xg_size * sizeof(float));
            // 第 g 组对应的 w
            memcpy(wg, w + g * OUT_C / GROUP * IN_C / GROUP * K_H * K_W, wg_size * sizeof(float));
            // conv (1,C_in/group,H_in,W_in) * (C_out/group,C_in,H_k,W_k) = (1,C_out/group,H_out,W_out)
            slidingwindow(xg, wg, yg, 1, IN_C / GROUP, IN_H, IN_W, K_H, K_W, OUT_C / GROUP, OUT_H, OUT_W, PAD_H,
                PAD_W, STRIDE_H, STRIDE_W, DILATION_H, DILATION_W);
            // 第 g 组对应的 y
            memcpy(y + IY(n, g * OUT_C / GROUP, 0, 0), yg, yg_size * sizeof(float));
            free(xg);
            free(wg);
            free(yg);
        }
    }
}

void depthwiseSeparableConv(const float *x, const float *depthwise_w, const float *pointwise_w, float *y1, float *y2) {
    // depthwise conv
    groupconv(x, depthwise_w, y1);
    // pointwise conv
    slidingwindow(y1, pointwise_w, y2, N, OUT_C, OUT_H, OUT_W, 1, 1, OUT_C / GROUP, OUT_H, OUT_W, PAD_H,
        PAD_W, STRIDE_H, STRIDE_W, DILATION_H, DILATION_W);
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
    cudnnFilterDescriptor_t depthwise_w_desc, pointwise_w_desc;
    cudnnCreateFilterDescriptor(&depthwise_w_desc);
    cudnnCreateFilterDescriptor(&pointwise_w_desc);
    cudnnSetFilter4dDescriptor(depthwise_w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, OUT_C, IN_C / GROUP, K_H, K_W);
    cudnnSetFilter4dDescriptor(pointwise_w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, OUT_C / GROUP, OUT_C, 1, 1);

    // Define the convolution descriptor
    cudnnConvolutionDescriptor_t depthwise_conv_desc, pointwise_conv_desc;
    cudnnCreateConvolutionDescriptor(&depthwise_conv_desc);
    cudnnCreateConvolutionDescriptor(&pointwise_conv_desc);
    cudnnSetConvolution2dDescriptor(depthwise_conv_desc, PAD_H, PAD_W, STRIDE_H, STRIDE_W, DILATION_H, DILATION_W, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    cudnnSetConvolution2dDescriptor(pointwise_conv_desc, PAD_H, PAD_W, STRIDE_H, STRIDE_W, DILATION_H, DILATION_W, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    // Define the y tensor descriptor
    cudnnTensorDescriptor_t y1_desc, y2_desc;
    cudnnCreateTensorDescriptor(&y1_desc);
    cudnnCreateTensorDescriptor(&y2_desc);
    cudnnSetTensor4dDescriptor(y1_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, OUT_C, OUT_H, OUT_W);
    cudnnSetTensor4dDescriptor(y2_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, OUT_C / GROUP, OUT_H, OUT_W);

    int size_x = N * IN_C * IN_H * IN_W;
    int size_depthwise_w = OUT_C / GROUP * IN_C * K_H * K_W;
    int size_pointwise_w = OUT_C / GROUP * OUT_C * 1 * 1;
    int size_y1 = N * OUT_C * OUT_H * OUT_W;
    int size_y2 = N * OUT_C / GROUP * OUT_H * OUT_W;

    // Allocate host memory for x, w, and y
    float *h_x, *h_depthwise_w, *h_pointwise_w, *h_y1, *h_y2;
    h_x = (float*)malloc(size_x * sizeof(float));
    h_depthwise_w = (float*)malloc(size_depthwise_w * sizeof(float));
    h_pointwise_w = (float*)malloc(size_pointwise_w * sizeof(float));
    h_y1 = (float*)malloc(size_y1 * sizeof(float));
    h_y2 = (float*)malloc(size_y2 * sizeof(float));

    // Initialization
    rand_data(h_x, size_x, -1, 1);
    rand_data(h_depthwise_w, size_depthwise_w, -1, 1);
    rand_data(h_pointwise_w, size_pointwise_w, 1, 1);
    rand_data(h_y1, size_y1, 0, 0);
    rand_data(h_y2, size_y2, 0, 0);

    // Allocate device memory for x, w, and y
    float *d_x, *d_depthwise_w, *d_pointwise_w, *d_y1, *d_y2;
    cudaMalloc((void**)&d_x, size_x * sizeof(float));
    cudaMalloc((void**)&d_depthwise_w, size_depthwise_w * sizeof(float));
    cudaMalloc((void**)&d_pointwise_w, size_pointwise_w * sizeof(float));
    cudaMalloc((void**)&d_y1, size_y1 * sizeof(float));
    cudaMalloc((void**)&d_y2, size_y2 * sizeof(float));

    // Memcpy: host -> device
    cudaMemcpy(d_x, h_x, size_x * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_depthwise_w, h_depthwise_w, size_depthwise_w * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pointwise_w, h_pointwise_w, size_pointwise_w * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y1, h_y1, size_y1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y2, h_y2, size_y2 * sizeof(float), cudaMemcpyHostToDevice);

    // Perform depthwise forward convolution
    float alpha = 1.0f, beta = 0.0f;
    // Set the group count
    cudnnSetConvolutionGroupCount(depthwise_conv_desc, GROUP);
    cudnnConvolutionForward(cudnn, &alpha, x_desc, d_x, depthwise_w_desc, d_depthwise_w, depthwise_conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, nullptr, 0, &beta, y1_desc, d_y1);
    // Perform POINTwise forward convolution
    cudnnConvolutionForward(cudnn, &alpha, y1_desc, d_y1, pointwise_w_desc, d_pointwise_w, pointwise_conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, nullptr, 0, &beta, y2_desc, d_y2);

    // Memcpy: device -> host
    cudaMemcpy(h_y1, d_y1, size_y1 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y2, d_y2, size_y2 * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare
    float *calc_y1 = (float*)malloc(size_y1 * sizeof(float));
    float *calc_y2 = (float*)malloc(size_y2 * sizeof(float));
    depthwiseSeparableConv(h_x, h_depthwise_w, h_pointwise_w, calc_y1, calc_y2);
    float diff = 0.0f;
    for (int i = 0; i < size_y2; ++i) {
        diff += (h_y2[i] - calc_y2[i]);
    }
    printf("\n--------diff:%f------\n", diff);

    // Clean up
    free(h_x);
    free(h_depthwise_w);
    free(h_pointwise_w);
    free(h_y1);
    free(h_y2);
    free(calc_y1);
    free(calc_y2);
    cudaFree(d_x);
    cudaFree(d_depthwise_w);
    cudaFree(d_pointwise_w);
    cudaFree(d_y1);
    cudaFree(d_y2);
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyFilterDescriptor(depthwise_w_desc);
    cudnnDestroyFilterDescriptor(pointwise_w_desc);
    cudnnDestroyTensorDescriptor(y1_desc);
    cudnnDestroyTensorDescriptor(y2_desc);
    cudnnDestroyConvolutionDescriptor(depthwise_conv_desc);
    cudnnDestroyConvolutionDescriptor(pointwise_conv_desc);
    cudnnDestroy(cudnn);

    return 0;
}
