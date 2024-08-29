/*
 * wgrad conv 转换成 1*1 conv
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

void dwgrad(const float *dy, const float *x, float *dw) {

#define NEW_OUT_H ((OUT_H - 1) * (STRIDE_H - 1) + OUT_H)
#define NEW_OUT_W ((OUT_W - 1) * (STRIDE_W - 1) + OUT_W)

    for (int k_h = 0; k_h < K_H; ++k_h) {
        for (int k_w = 0; k_w < K_W; ++k_w) {
            int size_x = N * IN_C * NEW_OUT_H * NEW_OUT_W;
            int size_dy = N * OUT_C * NEW_OUT_H * NEW_OUT_W;
            int size_dw = OUT_C * IN_C * 1 * 1;
            float *xh = (float *)malloc(size_x * sizeof(float));
            float *dyh = (float *)malloc(size_dy * sizeof(float));
            float *dwh = (float *)malloc(size_dw * sizeof(float));
            memset(xh, 0, size_x * sizeof(float));
            memset(dyh, 0, size_dy * sizeof(float));
            memset(dwh, 0, size_dw * sizeof(float));
            // 1. 提取x中的对应数据 (N,INC,INH,INW)->(N,INC,OUTH,OUTW)
            for (int n = 0; n < N; ++n) {
                for (int in_c = 0; in_c < IN_C; ++in_c) {
                    for (int in_h = 0; in_h < NEW_OUT_H; ++in_h) {
                        for (int in_w = 0; in_w < NEW_OUT_W; ++in_w) {
                            int real_in_h = in_h + k_h * DILATION_H - PAD_H;
                            int real_in_w = in_w + k_w * DILATION_W - PAD_W;
                            if (real_in_h >= 0 && real_in_h < IN_H && real_in_w >= 0 && real_in_w < IN_W) {
                                int pos1 = ((n * IN_C + in_c) * NEW_OUT_H + in_h) * NEW_OUT_W + in_w;
                                int pos2 = ((n * IN_C + in_c) * IN_H + real_in_h) * IN_W + real_in_w;
                                xh[pos1] = x[pos2];
                            }
                        }
                    }
                }
            }
            // 2. 提取dy中的对应数据 (N,OUTC,OUTH,OUTW)->(N,OUTC,OUTH,OUTW)
            for (int n = 0; n < N; ++n) {
                for (int out_c = 0; out_c < OUT_C; ++out_c) {
                    for (int out_h = 0; out_h < NEW_OUT_H; out_h += STRIDE_H) {
                        for (int out_w = 0; out_w < NEW_OUT_W; out_w += STRIDE_W) {
                            if (out_h % STRIDE_H == 0 && out_w % STRIDE_W == 0) {
                                int real_out_h = out_h / STRIDE_H;
                                int real_out_w = out_w / STRIDE_W;
                                int pos1 = ((n * OUT_C + out_c) * NEW_OUT_H + out_h) * NEW_OUT_W + out_w;
                                int pos2 = ((n * OUT_C + out_c) * OUT_H + real_out_h) * OUT_W + real_out_w;
                                dyh[pos1] = dy[pos2];
                            }
                        }
                    }
                }
            }
            // 3. 1*1 conv
            float *A = (float *)malloc(size_x * sizeof(float));
            float *Atran = (float *)malloc(size_x * sizeof(float));
            float *B = (float *)malloc(size_dy * sizeof(float));
            float *C = (float *)malloc(size_dw * sizeof(float));
            // (1) 第4维放到第2维 (N,INC,OUTH,OUTW) -> (INC,N*OUTH*OUTW)
            NCHW2NHWC(xh, A, N, IN_C, NEW_OUT_H, NEW_OUT_W);
            transpose(A, Atran, N * NEW_OUT_H * NEW_OUT_W, IN_C);
            // (2) 第2维放到第4维 (N,OUTC,OUTH,OUTW) -> (N*OUTH*OUTW,OUTC)
            NCHW2NHWC(dyh, B, N, OUT_C, NEW_OUT_H, NEW_OUT_W);
            // (3) 矩阵乘 (INC,N*OUTH*OUTW) * (N*OUTH*OUTW,OUTC) = (INC,OUTC)
            gemm(Atran, B, C, IN_C, OUT_C, N * NEW_OUT_H * NEW_OUT_W);
            // (4) 转置
            transpose(C, dwh, IN_C, OUT_C);
            // 4. 累加
            for (int i = 0; i < size_dw; ++i) {
                dw[(i * K_H + k_h) * K_W + k_w] = dwh[i];
            }
            free(A);
            free(Atran);
            free(B);
            free(C);
            free(xh);
            free(dyh);
            free(dwh);

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
    rand_data(calc_dw, size_dw, 0, 0);
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