#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>

#define N 6
#define IN_C 5
#define IN_L 7
#define K_L 3
#define OUT_C 3
#define OUT_L 3
#define PAD_L 1
#define STRIDE_L 2
#define DILATION_L 2

void rand_data(float *data, int num, float min, float max) {
    for (int i = 0; i < num; i++) {
        data[i] = (fabs(max - min) < 1e-5) ? min : ((max - min) * (rand() / (float)RAND_MAX) + min);
    }
}

void conv1d(const float* x, const float* w, float* y) {

#define IX(n, in_c, in_l) (((n) * IN_C + in_c) * IN_L + in_l)
#define IW(out_c, in_c, k_l) (((out_c) * IN_C + in_c) * K_L + k_l)
#define IY(n, out_c, out_l) (((n) * OUT_C + out_c) * OUT_L + out_l)

    for (int n = 0; n < N; ++n) {
        for (int out_c = 0; out_c < OUT_C; ++out_c) {
            for (int out_l = 0; out_l < OUT_L; ++out_l) {
                float temp = 0.0f;
                for (int k_l = 0; k_l < (DILATION_L - 1) * (K_L - 1) + K_L; k_l += DILATION_L) {
                    int real_in_l = out_l * STRIDE_L + k_l - PAD_L;
                    if (real_in_l >= 0 && real_in_l < IN_L) {
                        int real_k_l = k_l / DILATION_L;
                        for (int in_c = 0; in_c < IN_C; ++in_c) {
                            temp += (float)x[IX(n, in_c, real_in_l)] * (float)w[IW(out_c, in_c, real_k_l)];
                        }
                    }
                }
                y[IY(n, out_c, out_l)] = temp;
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
    cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, IN_C, IN_L, 1);

    // Define the convolution descriptor
    cudnnFilterDescriptor_t w_desc;
    cudnnCreateFilterDescriptor(&w_desc);
    cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, OUT_C, IN_C, K_L, 1);

    // Define the convolution descriptor
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc, PAD_L, 0, STRIDE_L, 1, DILATION_L, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    // Define the y tensor descriptor
    cudnnTensorDescriptor_t y_desc;
    cudnnCreateTensorDescriptor(&y_desc);
    cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, OUT_C, OUT_L, 1);

    int size_x = N * IN_C * IN_L;
    int size_w = OUT_C * IN_C * K_L;
    int size_y = N * OUT_C * OUT_L;

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
    cudnnConvolutionForward(cudnn, &alpha, x_desc, d_x, w_desc, d_w, conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, nullptr, 0, &beta, y_desc, d_y);

    // Memcpy: device -> host
    cudaMemcpy(h_y, d_y, size_y * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare
    float *calc_y = (float*)malloc(size_y * sizeof(float));
    conv1d(h_x, h_w, calc_y);
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
