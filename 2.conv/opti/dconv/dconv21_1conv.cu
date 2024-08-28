/*
 * dgrad conv 转换成 1*1 conv
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

void dbgrad(const float *dy, const float *w, float *dx) {

    int size_dy = N * OUT_C * IN_H * IN_W;
    int size_w = OUT_C * IN_C * 1 * 1;
    int size_dx = N * IN_C * IN_H * IN_W;
    float *A = (float *)malloc(size_dy * sizeof(float));
    float *B = (float *)malloc(size_w * sizeof(float));
    float *C = (float *)malloc(size_dx * sizeof(float));

    // 1.第2维放到第4维 (N,OUTC,INH,INW) -> (N*INH*INW,OUTC)
    NCHW2NHWC(dy, A, N, OUT_C, IN_H, IN_W);
    // 2.转置 (OUT_C,IN_C,1,1) -> (OUT_C,IN_C)
    // transpose(w, B, OUT_C, IN_C);
    // 3.矩阵乘 (N*INH*INW,OUTC) * (OUT_C,IN_C) = (N*INH*INW,IN_C)
    gemm(A, w, C, N * IN_H * IN_W, IN_C, OUT_C);
    // 4.第4维放到第2维 (N*INH*INW,IN_C) -> (N,INC,INH,INW)
    NHWC2NCHW(C, dx, N, IN_H, IN_W, IN_C);

    free(A);
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

    // Define the w descriptor
    cudnnFilterDescriptor_t w_desc;
    cudnnCreateFilterDescriptor(&w_desc);
    cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, OUT_C, IN_C, K_H, K_W);

    // Define the dx tensor descriptor
    cudnnTensorDescriptor_t dx_desc;
    cudnnCreateTensorDescriptor(&dx_desc);
    cudnnSetTensor4dDescriptor(dx_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, IN_C, IN_H, IN_W);

    // Define the convolution descriptor
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc, PAD_H, PAD_W, STRIDE_H, STRIDE_W, DILATION_H, DILATION_W, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    int size_dy = N * OUT_C * OUT_H * OUT_W;
    int size_w = OUT_C * IN_C * K_H * K_W;
    int size_dx = N * IN_C * IN_H * IN_W;

    // Allocate host memory
    float *h_dy, *h_w, *h_dx;
    h_dy = (float*)malloc(size_dy * sizeof(float));
    h_w = (float*)malloc(size_w * sizeof(float));
    h_dx = (float*)malloc(size_dx * sizeof(float));

    // Initialization
    rand_data(h_dy, size_dy, -1, 1);
    rand_data(h_w, size_w, -1, 1);
    rand_data(h_dx, size_dx, 0, 0);

    // Allocate device memory
    float *d_dy, *d_w, *d_dx;
    cudaMalloc((void**)&d_dy, size_dy * sizeof(float));
    cudaMalloc((void**)&d_w, size_w * sizeof(float));
    cudaMalloc((void**)&d_dx, size_dx * sizeof(float));

    // Memcpy: host -> device
    cudaMemcpy(d_dy, h_dy, size_dy * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, size_w * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dx, h_dx, size_dx * sizeof(float), cudaMemcpyHostToDevice);

    // Perform bgrad convolution
    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionBackwardData(cudnn, &alpha, w_desc, d_w, dy_desc, d_dy, conv_desc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, nullptr, 0, &beta, dx_desc, d_dx);

    // Memcpy: device -> host
    cudaMemcpy(h_dx, d_dx, size_dx * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compare
    float *calc_dx = (float*)malloc(size_dx * sizeof(float));
    rand_data(calc_dx, size_dx, 0, 0);
    dbgrad(h_dy, h_w, calc_dx);
    float diff = 0.0f;
    for (int i = 0; i < size_dx; ++i) {
        diff += (h_dx[i] - calc_dx[i]);
    }
    printf("\n--------diff:%f------\n", diff);

    // Clean up
    free(h_dy);
    free(h_w);
    free(h_dx);
    free(calc_dx);
    cudaFree(d_dy);
    cudaFree(d_w);
    cudaFree(d_dx);
    cudnnDestroyTensorDescriptor(dy_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyTensorDescriptor(dx_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn);

    return 0;
}
