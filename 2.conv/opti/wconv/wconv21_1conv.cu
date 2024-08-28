/*
 * wgrad conv 转换成 1*1 conv
 */

#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>

#define N 1
#define IN_C 1
#define IN_H 3
#define IN_W 3
#define K_H 1
#define K_W 1
#define OUT_C 1
#define OUT_H 3
#define OUT_W 3
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

void NCHW2NHWC(const float *old_arr, float *new_arr, const int NUM, const int C, const int H, const int W) {
    for (int n = 0; n < NUM; ++n) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                for (int c = 0; c < C; ++c) {
                    new_arr[((n * H + h) * W + w) * C + c] = old_arr[((n * C + c) * H + h) * W + w];
                }
            }
        }
    }
}

void NHWC2NCHW(const float *old_arr, float *new_arr, const int NUM, const int H, const int W, const int C) {
    for (int n = 0; n < NUM; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    new_arr[((n * C + c) * H + h) * W + w] = old_arr[((n * H + h) * W + w) * C + c];
                }
            }
        }
    }
}

void transpose(const float *old_arr, float *new_arr, const int ROW, const int COL) {
    for (int row = 0; row < ROW; ++row) {
        for (int col = 0; col < COL; ++col) {
            new_arr[col * ROW + row] = old_arr[row * COL + col];
        }
    }
}

void gemm(const float *A, const float *B, float *C, const int ROW, const int COL, const int K) {
    for (int m = 0; m < ROW; ++m) {
        for (int n = 0; n < COL; ++n) {
            float temp = 0.0f;
            for (int k = 0; k < K; ++k) {
                temp += A[m * K + k] * B[k * COL + n];
            }
            C[m * COL + n] = temp;
        }
    }
}

void dwgrad(const float *dy, const float *x, float *dw) {

    int size_dy = N * OUT_C * IN_H * IN_W;
    int size_x = N * IN_C * IN_H * IN_W;
    int size_dw = OUT_C * IN_C * 1 * 1;
    float *A = (float *)malloc(size_dy * sizeof(float));
    float *Atran = (float *)malloc(size_dy * sizeof(float));
    float *B = (float *)malloc(size_x * sizeof(float));
    float *C = (float *)malloc(size_dw * sizeof(float));

    // 1.第2维放到第4维 (N,OUTC,INH,INW) -> (OUTC,N*INH*INW)
    NCHW2NHWC(dy, A, N, OUT_C, IN_H, IN_W);
    transpose(A, Atran, N * IN_H * IN_W, OUT_C);
    // 2.第4维放到第2维 (N,IN_C,INH,INW) -> (N*INH*INW,INC)
    NCHW2NHWC(x, B, N, IN_C, IN_H, IN_W);
    // 3.矩阵乘 (OUTC,N*INH*INW) * (N*INH*INW,INC) = (OUTC,INC)
    gemm(Atran, B, dw, OUT_C, IN_C, N * IN_H * IN_W);

    free(A);
    free(Atran);
    free(B);
    free(C);

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

    // Define the dy tensor descriptor
    cudnnTensorDescriptor_t dy_desc;
    cudnnCreateTensorDescriptor(&dy_desc);
    cudnnSetTensor4dDescriptor(dy_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, OUT_C, OUT_H, OUT_W);

    // Define the x tensor descriptor
    cudnnTensorDescriptor_t x_desc;
    cudnnCreateTensorDescriptor(&x_desc);
    cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, IN_C, IN_H, IN_W);

    // Define the dw descriptor
    cudnnFilterDescriptor_t dw_desc;
    cudnnCreateFilterDescriptor(&dw_desc);
    cudnnSetFilter4dDescriptor(dw_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, OUT_C, IN_C, K_H, K_W);

    // Define the convolution descriptor
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc, PAD_H, PAD_W, STRIDE_H, STRIDE_W, DILATION_H, DILATION_W, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    int size_dy = N * OUT_C * OUT_H * OUT_W;
    int size_x = N * IN_C * IN_H * IN_W;
    int size_dw = OUT_C * IN_C * K_H * K_W;

    // Allocate host memory
    float *h_dy, *h_x, *h_dw;
    h_dy = (float*)malloc(size_dy * sizeof(float));
    h_x = (float*)malloc(size_x * sizeof(float));
    h_dw = (float*)malloc(size_dw * sizeof(float));

    // Initialization
    rand_data(h_dy, size_dy, -1, 1);
    rand_data(h_x, size_x, -1, 1);
    rand_data(h_dw, size_dw, 0, 0);

    // Allocate device memory
    float *d_dy, *d_x, *d_dw;
    cudaMalloc((void**)&d_dy, size_dy * sizeof(float));
    cudaMalloc((void**)&d_x, size_x * sizeof(float));
    cudaMalloc((void**)&d_dw, size_dw * sizeof(float));

    // Memcpy: host -> device
    cudaMemcpy(d_dy, h_dy, size_dy * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, size_x * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dw, h_dw, size_dw * sizeof(float), cudaMemcpyHostToDevice);

    // Perform bgrad convolution
    float alpha = 1.0f, beta = 0.0f;
    cudnnStatus_t yan = cudnnConvolutionBackwardFilter(cudnn, &alpha, x_desc, d_x, dy_desc, d_dy, conv_desc, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0, nullptr, 0, &beta, dw_desc, d_dw);

    // Memcpy: device -> host
    cudaMemcpy(h_dw, d_dw, size_dw * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare
    float *calc_dw = (float*)malloc(size_dw * sizeof(float));
    dwgrad(h_dy, h_x, calc_dw);
    float diff = 0.0f;
    for (int i = 0; i < size_dw; ++i) {
        diff += (h_dw[i] - calc_dw[i]);
    }
    printf("\n--------diff:%f------\n", diff);

    // Clean up
    free(h_dy);
    free(h_x);
    free(h_dw);
    free(calc_dw);
    cudaFree(d_dy);
    cudaFree(d_x);
    cudaFree(d_dw);
    cudnnDestroyTensorDescriptor(dy_desc);
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyFilterDescriptor(dw_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn);

    return 0;
}