/*
 * winograd
 * 存在限制，pad=0，stride=1，dilation=1
 * 以 F(2,3) 为例，即 IN = 4, K = 3
 */

#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>

using namespace std;

#define N 6
#define IN_C 5
#define IN_H 4
#define IN_W 4
#define K_H 3
#define K_W 3
#define OUT_C 4
#define OUT_H 2
#define OUT_W 2
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

void winograd(const float* x, const float* w, float* y) {

#define IX(n, in_c, in_h, in_w) ((((n) * IN_C + in_c) * IN_H + in_h) * IN_W + in_w)
#define IW(out_c, in_c, k_h, k_w) ((((out_c) * IN_C + in_c) * K_H + k_h) * K_W + k_w)
#define IY(n, out_c, out_h, out_w) ((((n) * OUT_C + out_c) * OUT_H + out_h) * OUT_W + out_w)

    for (int n = 0; n < N; ++n) {
        for (int out_c = 0; out_c < OUT_C; ++out_c) {
            for (int in_c = 0; in_c < IN_C; ++in_c) {

                float *X[4], *W[3], *Ytemp[4], *Y[4];
                
                // X
                for (int num = 0; num < 4; ++num) {
                    X[num] = (float *)malloc(2 * 3 * sizeof(float));
                }
                X[0][0] = x[IX(n, in_c, 0, 0)];
                X[0][1] = x[IX(n, in_c, 0, 1)];
                X[0][2] = x[IX(n, in_c, 0, 2)];
                X[0][3] = x[IX(n, in_c, 0, 1)];
                X[0][4] = x[IX(n, in_c, 0, 2)];
                X[0][5] = x[IX(n, in_c, 0, 3)];
                X[1][0] = x[IX(n, in_c, 1, 0)];
                X[1][1] = x[IX(n, in_c, 1, 1)];
                X[1][2] = x[IX(n, in_c, 1, 2)];
                X[1][3] = x[IX(n, in_c, 1, 1)];
                X[1][4] = x[IX(n, in_c, 1, 2)];
                X[1][5] = x[IX(n, in_c, 1, 3)];
                X[2][0] = x[IX(n, in_c, 2, 0)];
                X[2][1] = x[IX(n, in_c, 2, 1)];
                X[2][2] = x[IX(n, in_c, 2, 2)];
                X[2][3] = x[IX(n, in_c, 2, 1)];
                X[2][4] = x[IX(n, in_c, 2, 2)];
                X[2][5] = x[IX(n, in_c, 2, 3)];
                X[3][0] = x[IX(n, in_c, 3, 0)];
                X[3][1] = x[IX(n, in_c, 3, 1)];
                X[3][2] = x[IX(n, in_c, 3, 2)];
                X[3][3] = x[IX(n, in_c, 3, 1)];
                X[3][4] = x[IX(n, in_c, 3, 2)];
                X[3][5] = x[IX(n, in_c, 3, 3)];

                // W
                for (int num = 0; num < 3; ++num) {
                    W[num] = (float *)malloc(3 * 1 * sizeof(float));
                }
                W[0][0] = w[IW(out_c, in_c, 0, 0)];
                W[0][1] = w[IW(out_c, in_c, 0, 1)];
                W[0][2] = w[IW(out_c, in_c, 0, 2)];
                W[1][0] = w[IW(out_c, in_c, 1, 0)];
                W[1][1] = w[IW(out_c, in_c, 1, 1)];
                W[1][2] = w[IW(out_c, in_c, 1, 2)];
                W[2][0] = w[IW(out_c, in_c, 2, 0)];
                W[2][1] = w[IW(out_c, in_c, 2, 1)];
                W[2][2] = w[IW(out_c, in_c, 2, 2)];

                // Ytemp
                for (int num = 0; num < 4; ++num) {
                    Ytemp[num] = (float *)malloc(2 * 1 * sizeof(float));
                }
                // Ytemp1 = (X1-X3)W1
                for (int row = 0; row < 2; ++row) {
                    for (int col = 0; col < 1; ++col) {
                        float temp = 0.0f;
                        for (int k = 0; k < 3; ++k) {
                            temp += (X[0][row * 3 + k] - X[2][row * 3 + k]) * W[0][k * 1 + col];
                        }
                        Ytemp[0][row * 1 + col] = temp;
                    }
                }
                // Ytemp2 = (X2+X3)(W1+W2+W3)/2
                for (int row = 0; row < 2; ++row) {
                    for (int col = 0; col < 1; ++col) {
                        float temp = 0.0f;
                        for (int k = 0; k < 3; ++k) {
                            temp += (X[1][row * 3 + k] + X[2][row * 3 + k]) * (W[0][k * 1 + col] +
                                W[1][k * 1 + col] + W[2][k * 1 + col]) / 2.0f;
                        }
                        Ytemp[1][row * 1 + col] = temp;
                    }
                }
                // Ytemp3 = (X3-X2)(W1-W2+W3)/2
                for (int row = 0; row < 2; ++row) {
                    for (int col = 0; col < 1; ++col) {
                        float temp = 0.0f;
                        for (int k = 0; k < 3; ++k) {
                            temp += (X[2][row * 3 + k] - X[1][row * 3 + k]) * (W[0][k * 1 + col] -
                                W[1][k * 1 + col] + W[2][k * 1 + col]) / 2.0f;
                        }
                        Ytemp[2][row * 1 + col] = temp;
                    }
                }
                // Ytemp4 = (X2-X4)W3
                for (int row = 0; row < 2; ++row) {
                    for (int col = 0; col < 1; ++col) {
                        float temp = 0.0f;
                        for (int k = 0; k < 3; ++k) {
                            temp += (X[1][row * 3 + k] - X[3][row * 3 + k]) * W[2][k * 1 + col];
                        }
                        Ytemp[3][row * 1 + col] = temp;
                    }
                }
                // Y
                for (int num = 0; num < 2; ++num) {
                    Y[num] = (float *)malloc(2 * 1 * sizeof(float));
                }
                // Y1 = Ytemp1+Ytemp2+Ytemp3
                for (int i = 0; i < 2; ++i) {
                    for (int j = 0; j < 1; ++j) {
                        Y[0][i*1+j] = Ytemp[0][i*1+j] + Ytemp[1][i*1+j] + Ytemp[2][i*1+j];
                    }
                }
                // Y1 = Ytemp2-Ytemp3-Ytemp4
                for (int i = 0; i < 2; ++i) {
                    for (int j = 0; j < 1; ++j) {
                        Y[1][i*1+j] = Ytemp[1][i*1+j] - Ytemp[2][i*1+j] - Ytemp[3][i*1+j];
                    }
                }

                // write back
                y[IY(n, out_c, 0, 0)] += Y[0][0];
                y[IY(n, out_c, 0, 1)] += Y[0][1];
                y[IY(n, out_c, 1, 0)] += Y[1][0];
                y[IY(n, out_c, 1, 1)] += Y[1][1];

                // free
                for (int num = 0; num < 4; ++num) {
                    free(X[num]);
                }
                for (int num = 0; num < 3; ++num) {
                    free(W[num]);
                }
                for (int num = 0; num < 4; ++num) {
                    free(Ytemp[num]);
                }
                for (int num = 0; num < 2; ++num) {
                    free(Y[num]);
                }

            }
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
    memset(calc_y, 0, size_y * sizeof(float));
    winograd(h_x, h_w, calc_y);
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
