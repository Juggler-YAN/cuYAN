/*
 * conv转换成1*1 conv
 */

#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>

#define N 6
#define IN_C 5
#define IN_H 7
#define IN_W 7
#define K_H 2
#define K_W 2
#define OUT_C 3
#define OUT_H 4
#define OUT_W 4
#define PAD_H 1
#define PAD_W 1
#define STRIDE_H 2
#define STRIDE_W 2
#define DILATION_H 2
#define DILATION_W 2

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

void conv(const float* x, const float* w, float* y) {

#define NEW_K_H ((K_H - 1) * (DILATION_H - 1) + K_H)
#define NEW_K_W ((K_W - 1) * (DILATION_W - 1) + K_W)

    for (int k_h = 0; k_h < NEW_K_H; k_h += DILATION_H) {
        for (int k_w = 0; k_w < NEW_K_W; k_w += DILATION_W) {
            int real_k_h = k_h / DILATION_H;
            int real_k_w = k_w / DILATION_W;
            int size_x = N * IN_C * OUT_H * OUT_W;
            int size_w = OUT_C * IN_C * 1 * 1;
            int size_y = N * OUT_C * OUT_H * OUT_W;
            float *xh = (float *)malloc(size_x * sizeof(float));
            float *wh = (float *)malloc(size_w * sizeof(float));
            float *yh = (float *)malloc(size_y * sizeof(float));
            memset(xh, 0, size_x * sizeof(float));
            memset(wh, 0, size_w * sizeof(float));
            memset(yh, 0, size_y * sizeof(float));
            // 1. 提取x中的对应数据 (N,INC,INH,INW)->(N,INC,OUTH,OUTW)
            for (int n = 0; n < N; ++n) {
                for (int in_c = 0; in_c < IN_C; ++in_c) {
                    for (int in_h = 0; in_h < OUT_H; ++in_h) {
                        for (int in_w = 0; in_w < OUT_W; ++in_w) {
                            int real_in_h = in_h * STRIDE_H + k_h - PAD_H;
                            int real_in_w = in_w * STRIDE_W + k_w - PAD_W;
                            if (real_in_h >= 0 && real_in_h < IN_H && real_in_w >= 0 && real_in_w < IN_W) {
                                int pos1 = ((n * IN_C + in_c) * OUT_H + in_h) * OUT_W + in_w;
                                int pos2 = ((n * IN_C + in_c) * IN_H + real_in_h) * IN_W + real_in_w;
                                xh[pos1] = x[pos2];
                            }
                        }
                    }
                }
            }
            // 2. 提取w中的对应数据 (OUTC,INC,KH,KW)->第(kh,kw)个(OUTC,INC,1，1)
            for (int out_c = 0; out_c < OUT_C; ++out_c) {
                for (int in_c = 0; in_c < IN_C; ++in_c) {
                    int pos1 = out_c * IN_C + in_c;
                    int pos2 = ((out_c * IN_C + in_c) * K_H + real_k_h) * K_W + real_k_w;
                    wh[pos1] = w[pos2];
                }
            }
            // 3. 1*1 conv
            float *A = (float *)malloc(size_x * sizeof(float));
            float *B = (float *)malloc(size_w * sizeof(float));
            float *C = (float *)malloc(size_y * sizeof(float));
            // (1) 第2维放到第4维 (N,INC,OUTH,OUTW) -> (N*OUTH*OUTW,INC)
            NCHW2NHWC(xh, A, N, IN_C, OUT_H, OUT_W);
            // (2) 转置 (OUT_C,IN_C,1,1) -> (IN_C,OUT_C,1,1)
            transpose(wh, B, OUT_C, IN_C);
            // (3) 矩阵乘 (N*OUTH*OUTW,INC) * (IN_C,OUT_C) = (N*INH*INW,OUT_C)
            gemm(A, B, C, N * OUT_H * OUT_W, OUT_C, IN_C);
            // (4) 第4维放到第2维 (N*INH*INW,OUT_C) -> (N,OUTC,OUTH,OUTW)
            NHWC2NCHW(C, yh, N, OUT_H, OUT_W, OUT_C);
            // 4. 写回+累加
            for (int i = 0; i < size_y; ++i) {
                y[i] += yh[i];
            }
            free(A);
            free(B);
            free(C);
            free(xh);
            free(wh);
            free(yh);
        }
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
