/*
 * conv抽取H,W维补充C维 (限制p=0,d=1)
 * 举例如下，抽W维补充Cin维
 */

#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>

#define GROUP 2
#define N 6
#define IN_C 5
#define IN_H 7
#define IN_W 7
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

void conv(const float* x, const float* w, float* y) {
    
    float *xh = (float *)malloc(N * IN_C * GROUP * IN_H * (IN_W + GROUP - 1) / GROUP * sizeof(float));
    float *wh = (float *)malloc(OUT_C * IN_C * GROUP * K_H * (K_W + GROUP - 1) / GROUP * sizeof(float));
    memset(xh, 0, N * IN_C * GROUP * IN_H * (IN_W + GROUP - 1) / GROUP * sizeof(float));
    memset(wh, 0, OUT_C * IN_C * GROUP * K_H * (K_W + GROUP - 1) / GROUP * sizeof(float));
    // 1. 转换x (N,Cin,Hin,Win) -> (N,Cin*GROUP,Hin,Win/GROUP)
    for (int n = 0; n < N; ++n) {
        for (int in_c = 0; in_c < IN_C; ++in_c) {
            for (int in_h = 0; in_h < IN_H; ++in_h) {
                for (int in_w = 0; in_w < IN_W; ++in_w) {
                    int real_in_c = (in_w % GROUP) * IN_C + in_c;
                    int real_in_w = in_w / GROUP;
                    int pos1 = (((n) * IN_C * GROUP + real_in_c) * IN_H + in_h) * (IN_W + GROUP - 1) / GROUP + real_in_w;
                    int pos2 = (((n) * IN_C + in_c) * IN_H + in_h) * IN_W + in_w;
                    xh[pos1] = x[pos2];
                }
            }
        }
    }
    // 2. 转换w (Cout,Cin,Hk,Wk) -> (Cout,Cin \times 2,Hk,Wk/2)
    for (int out_c = 0; out_c < OUT_C; ++out_c) {
        for (int in_c = 0; in_c < IN_C; ++in_c) {
            for (int k_h = 0; k_h < K_H; ++k_h) {
                for (int k_w = 0; k_w < K_W; ++k_w) {
                    int real_in_c = (k_w % GROUP) * IN_C + in_c;
                    int real_k_w = k_w / GROUP;
                    int pos1 = (((out_c) * IN_C * GROUP + real_in_c) * K_H + k_h) * (K_W + GROUP - 1) / GROUP + real_k_w;
                    int pos2 = (((out_c) * IN_C + in_c) * K_H + k_h) * K_W + k_w;
                    wh[pos1] = w[pos2];
                }
            }
        }
    }
    // 3. conv
    for (int n = 0; n < N; ++n) {
        for (int out_c = 0; out_c < OUT_C; ++out_c) {
            for (int out_h = 0; out_h < OUT_H; ++out_h) {
                for (int out_w = 0; out_w < OUT_W; ++out_w) {
                    float temp = 0.0f;
                    for (int k_h = 0; k_h < K_H; ++k_h) {
                        for (int k_w = 0; k_w < (K_W + GROUP - 1) / GROUP; ++k_w) {
                            int real_in_h = out_h * STRIDE_H + k_h;
                            int real_in_w = out_w + k_w;
                            if (real_in_h >= 0 && real_in_h < IN_H && real_in_w >= 0 && real_in_w < (IN_W + GROUP - 1) / GROUP) {
                                int real_k_h = k_h;
                                int real_k_w = k_w;
                                for (int in_c = 0; in_c < IN_C * GROUP; ++in_c) {
                                    int xpos = (((n) * IN_C * GROUP + in_c) * IN_H + real_in_h) * (IN_W + GROUP - 1) / GROUP + real_in_w;
                                    int wpos = (((out_c) * IN_C * GROUP + in_c) * K_H + real_k_h) * (K_W + GROUP - 1) / GROUP + real_k_w;
                                    temp += (float)xh[xpos] * (float)wh[wpos];
                                }
                            }
                        }
                    }
                    // 4.累加
                    int ypos = (((n) * OUT_C + out_c) * OUT_H + out_h) * OUT_W + out_w;
                    y[ypos] += temp;
                }
            }
        }
    }

    free(xh);
    free(wh);

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
    rand_data(calc_y, size_y, 0, 0);
    conv(h_x, h_w, calc_y);
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
