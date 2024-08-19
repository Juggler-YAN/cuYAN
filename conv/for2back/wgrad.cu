#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>

#define N 6
#define IN_C 5
#define IN_H 7
#define IN_W 7
#define K_H 3
#define K_W 3
#define OUT_C 3
#define OUT_H 3
#define OUT_W 3
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

void dwgrad(const float *dy, const float *x, float *dw) {

#define IDY(n, out_c, out_h, out_w) ((((n) * OUT_C + out_c) * OUT_H + out_h) * OUT_W + out_w)
#define IX(n, in_c, in_h, in_w) ((((n) * IN_C + in_c) * IN_H + in_h) * IN_W + in_w)
#define IDW(out_c, in_c, k_h, k_w) ((((out_c) * IN_C + in_c) * K_H + k_h) * K_W + k_w)

    const int NEW_OUT_H = (STRIDE_H - 1) * (OUT_H - 1) + OUT_H;
    const int NEW_OUT_W = (STRIDE_W - 1) * (OUT_W - 1) + OUT_W;

    for (int out_c = 0; out_c < OUT_C; ++out_c) {
        for (int in_c = 0; in_c < IN_C; ++in_c) {
            for (int k_h = 0; k_h < K_H; ++k_h) {
                for (int k_w = 0; k_w < K_W; ++k_w) {
                    float temp = 0.0f;
                    for (int out_h = 0; out_h < NEW_OUT_H; out_h += STRIDE_H) {
                        for (int out_w = 0; out_w < NEW_OUT_W; out_w += STRIDE_W) {
                            int real_in_h = k_h * DILATION_H + out_h - PAD_H;
                            int real_in_w = k_w * DILATION_W + out_w - PAD_W;
                            if (real_in_h >= 0 && real_in_h < IN_H && real_in_w >= 0 && real_in_w < IN_W) {
                                int real_out_h = out_h / STRIDE_H;
                                int real_out_w = out_w / STRIDE_W;
                                for (int n = 0; n < N; ++n) {
                                    temp += (float)dy[IDY(n, out_c, real_out_h, real_out_w)] * (float)x[IX(n, in_c, real_in_h, real_in_w)];
                                }
                            }
                        }
                    }
                    dw[IDW(out_c, in_c, k_h, k_w)] = temp;
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