#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>

#define N 6
#define IN_C 5
#define IN_D 7
#define IN_H 7
#define IN_W 7
#define K_D 3
#define K_H 3
#define K_W 3
#define OUT_C 3
#define OUT_D 3
#define OUT_H 3
#define OUT_W 3
#define PAD_D 2
#define PAD_H 1
#define PAD_W 1
#define STRIDE_D 2
#define STRIDE_H 2
#define STRIDE_W 2
#define DILATION_D 3
#define DILATION_H 2
#define DILATION_W 2

void rand_data(float *data, int num, float min, float max) {
    for (int i = 0; i < num; i++) {
        data[i] = (fabs(max - min) < 1e-5) ? min : ((max - min) * (rand() / (float)RAND_MAX) + min);
    }
}

void conv3d(const float* x, const float* w, float* y) {

#define IX(n, in_c, in_d, in_h, in_w) (((((n) * IN_C + in_c) * IN_D + in_d) * IN_H + in_h) * IN_W + in_w)
#define IW(out_c, in_c, k_d, k_h, k_w) (((((out_c) * IN_C + in_c) * K_D + k_d) * K_H + k_h) * K_W + k_w)
#define IY(n, out_c, out_d, out_h, out_w) (((((n) * OUT_C + out_c) * OUT_D + out_d) * OUT_H + out_h) * OUT_W + out_w)

    for (int n = 0; n < N; ++n) {
        for (int out_c = 0; out_c < OUT_C; ++out_c) {
            for (int out_d = 0; out_d < OUT_D; ++out_d) {
                for (int out_h = 0; out_h < OUT_H; ++out_h) {
                    for (int out_w = 0; out_w < OUT_W; ++out_w) {
                        float temp = 0.0f;
                        for (int k_d = 0; k_d < (DILATION_D - 1) * (K_D - 1) + K_D; k_d += DILATION_D) {
                            for (int k_h = 0; k_h < (DILATION_H - 1) * (K_H - 1) + K_H; k_h += DILATION_H) {
                                for (int k_w = 0; k_w < (DILATION_W - 1) * (K_W - 1) + K_W; k_w += DILATION_W) {
                                    int real_in_d = out_d * STRIDE_D + k_d - PAD_D;
                                    int real_in_h = out_h * STRIDE_H + k_h - PAD_H;
                                    int real_in_w = out_w * STRIDE_W + k_w - PAD_W;
                                    if (real_in_d >= 0 && real_in_d < IN_D && real_in_h >= 0 && real_in_h < IN_H && real_in_w >= 0 && real_in_w < IN_W) {
                                        int real_k_d = k_d / DILATION_D;
                                        int real_k_h = k_h / DILATION_H;
                                        int real_k_w = k_w / DILATION_W;
                                        for (int in_c = 0; in_c < IN_C; ++in_c) {
                                            temp += (float)x[IX(n, in_c, real_in_d, real_in_h, real_in_w)] * (float)w[IW(out_c, in_c, real_k_d, real_k_h, real_k_w)];
                                        }
                                    }
                                }
                            }
                        }
                        y[IY(n, out_c, out_d, out_h, out_w)] = temp;
                    }
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
    const int x_dim[] = {N, IN_C, IN_D, IN_H, IN_W};
    const int x_stride[] = {x_dim[1] * x_dim[2] * x_dim[3] * x_dim[4], x_dim[2] * x_dim[3] * x_dim[4], x_dim[3] * x_dim[4], x_dim[4], 1};
    cudnnSetTensorNdDescriptor(x_desc, CUDNN_DATA_FLOAT, 5, x_dim, x_stride);

    // Define the convolution descriptor
    cudnnFilterDescriptor_t w_desc;
    cudnnCreateFilterDescriptor(&w_desc);
    const int w_dim[] = {OUT_C, IN_C, K_D, K_H, K_W};
    cudnnSetFilterNdDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 5, w_dim);

    // Define the convolution descriptor
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    const int PAD[] = {PAD_D, PAD_H, PAD_W};
    const int STRIDE[] = {STRIDE_D, STRIDE_H, STRIDE_W};
    const int DILATION[] = {DILATION_D, DILATION_H, DILATION_W};
    cudnnSetConvolutionNdDescriptor(conv_desc, 3, PAD, STRIDE, DILATION, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    // Define the y tensor descriptor
    cudnnTensorDescriptor_t y_desc;
    cudnnCreateTensorDescriptor(&y_desc);
    const int y_dim[] = {N, OUT_C, OUT_D, OUT_H, OUT_W};
    const int y_stride[] = {y_dim[1] * y_dim[2] * y_dim[3] * y_dim[4], y_dim[2] * y_dim[3] * y_dim[4], y_dim[3] * y_dim[4], y_dim[4], 1};
    cudnnSetTensorNdDescriptor(y_desc, CUDNN_DATA_FLOAT, 5, y_dim, y_stride);

    int size_x = N * IN_C * IN_D * IN_H * IN_W;
    int size_w = OUT_C * IN_C * K_D * K_H * K_W;
    int size_y = N * OUT_C * OUT_D * OUT_H * OUT_W;

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
    cudnnStatus_t res = cudnnConvolutionForward(cudnn, &alpha, x_desc, d_x, w_desc, d_w, conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, nullptr, 0, &beta, y_desc, d_y);

    // Memcpy: device -> host
    cudaMemcpy(h_y, d_y, size_y * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare
    float *calc_y = (float*)malloc(size_y * sizeof(float));
    conv3d(h_x, h_w, calc_y);
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
