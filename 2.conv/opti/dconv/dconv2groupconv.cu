/*
 * dgrad conv 转换成 stride_h * stride_w 组 stride = 1 的 dgrad conv
 */

#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>

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

#define N 1
#define IN_C 1
#define IN_H 3
#define IN_W 3
#define K_H 2
#define K_W 2
#define OUT_C 1
#define OUT_H 2
#define OUT_W 2
#define PAD_H 0
#define PAD_W 0
#define STRIDE_H 1
#define STRIDE_W 1
#define DILATION_H 1
#define DILATION_W 1

// void rand_data(float *data, int num, float min, float max) {
//     for (int i = 0; i < num; i++) {
//         data[i] = (fabs(max - min) < 1e-5) ? min : ((max - min) * (rand() / (float)RAND_MAX) + min);
//     }
// }
void rand_data(float *data, int num, float min, float max) {
    for (int i = 0; i < num; i++) {
        data[i] = (fabs(max - min) < 1e-5) ? min : ((max - min) * (rand() % 5) + min);
    }
}

void dgrad(const float *dy, const float *w, float *dx) {

#define NEW_K_H ((K_H - 1) * (DILATION_H - 1) + K_H)
#define NEW_K_W ((K_W - 1) * (DILATION_W - 1) + K_W)
#define STRIDE_K_H (NEW_K_H + STRIDE_H - 1) / STRIDE_H
#define STRIDE_K_W (NEW_K_W + STRIDE_W - 1) / STRIDE_W
#define STRIDE_IN_H (IN_H + STRIDE_H - 1) / STRIDE_H
#define STRIDE_IN_W (IN_W + STRIDE_W - 1) / STRIDE_W

    for (int s_h = 0; s_h < STRIDE_H; ++s_h) {
        for (int s_w = 0; s_w < STRIDE_W; ++s_w) {

            float *wh = (float *)malloc(OUT_C * IN_C * STRIDE_K_H * STRIDE_K_W * sizeof(float));
            float *dxh = (float *)malloc(N * IN_C * STRIDE_IN_H * STRIDE_IN_W * sizeof(float));
            memset(wh, 0, OUT_C * IN_C * STRIDE_K_H * STRIDE_K_W * sizeof(float));
            memset(dxh, 0, N * IN_C * STRIDE_IN_H * STRIDE_IN_W * sizeof(float));
            // 1. dy
            // 2. (sh,sw)对应的w
            for (int out_c = 0; out_c < OUT_C; ++out_c) {
                for (int in_c = 0; in_c < IN_C; ++in_c) {
                    for (int k_h = 0; k_h < STRIDE_K_H; ++k_h) {
                        for (int k_w = 0; k_w < STRIDE_K_W; ++k_w) {
                            if ((k_h * STRIDE_H + s_h) % DILATION_H == 0 && (k_w * STRIDE_W + s_w) % DILATION_W == 0) {
                                int real_k_h = (k_h * STRIDE_H + s_h) / DILATION_H;
                                int real_k_w = (k_w * STRIDE_W + s_w) / DILATION_W;
                                int pos1 = ((out_c * IN_C + in_c) * STRIDE_K_H + k_h) * STRIDE_K_W + k_w;
                                int pos2 = ((out_c * IN_C + in_c) * K_H + real_k_h) * K_W + real_k_w;
                                wh[pos1] = w[pos2];
                            } 
                        }
                    }
                }
            }
    printf("\n");
    for (int i = 0; i < STRIDE_K_H; ++i) {
        for (int j = 0; j < STRIDE_K_W; ++j) {
            printf("%3.0f ", wh[i*STRIDE_K_W+j]);
        }
        printf("\n");
    }
    printf("\n");
            // 3. conv -> (sh,sw)对应的dx
            for (int n = 0; n < N; ++n) {
                for (int in_c = 0; in_c < IN_C; ++in_c) {
                    for (int in_h = 0; in_h < STRIDE_IN_H; ++in_h) {
                        for (int in_w = 0; in_w < STRIDE_IN_W; ++in_w) {
                            float temp = 0.0f;
                            for (int k_h = 0; k_h < STRIDE_K_H; ++k_h) {
                                for (int k_w = 0; k_w < STRIDE_K_W; ++k_w) {
                                    int out_h = in_h + k_h;
                                    int out_w = in_w + k_w;
                                    for (int out_c = 0; out_c < OUT_C; ++out_c) {
                                        int dypos = (((n * OUT_C) + out_c) * OUT_H + out_h) *  OUT_W + out_w;
                                        int wpos = (((out_c * IN_C) + in_c) * STRIDE_K_H + k_h) *  STRIDE_K_W + k_w;
                                        printf("%3.0f %3.0f ", dy[dypos], w[wpos]);
                                        temp += dy[dypos] * w[wpos];
                                    }
                                }
                            }
                            printf("\n");
                            int real_in_h = in_h * STRIDE_H + s_h - PAD_H;
                            int real_in_w = in_w * STRIDE_W + s_w - PAD_W;
                            if (real_in_h >= 0 && real_in_h < IN_H && real_in_w >= 0 && real_in_w < IN_W) {
                                int dxpos = (((n * IN_C) + in_c) * IN_H + real_in_h) * IN_W + real_in_w;
                                dx[dxpos] = temp;
                            }
                        }
                    }
                }
            }
            // 4. write back

            free(wh);
            free(dxh);

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
    for (int i = 0; i < OUT_H; ++i) {
        for (int j = 0; j < OUT_W; ++j) {
            printf("%3.0f ", h_dy[i*OUT_W+j]);
        }
        printf("\n");
    }
    printf("\n");
    for (int i = 0; i < K_H; ++i) {
        for (int j = 0; j < K_W; ++j) {
            printf("%3.0f ", h_w[(K_H-i-1)*K_W+(K_W-j-1)]);
        }
        printf("\n");
    }
    printf("\n");
    dgrad(h_dy, h_w, calc_dx);
    float diff = 0.0f;
    for (int i = 0; i < size_dx; ++i) {
        printf("\n--------diff:%3.0f %3.0f------\n", h_dx[i], calc_dx[i]);
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
