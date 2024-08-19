#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>

#define N 6
#define OUT_C 3
#define OUT_H 3
#define OUT_W 3

void rand_data(float *data, int num, float min, float max) {
    for (int i = 0; i < num; i++) {
        data[i] = (fabs(max - min) < 1e-5) ? min : ((max - min) * (rand() / (float)RAND_MAX) + min);
    }
}

void dbgrad(const float *dy, float *db) {
    for (int i = 0; i < OUT_C; ++i) {
        float temp = 0.0f;
        for (int j = 0; j < N * OUT_H * OUT_W; ++j) {
            temp += dy[j * OUT_C + i];
        }
        db[i] = temp;
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

    // Define the db tensor descriptor
    cudnnTensorDescriptor_t db_desc;
    cudnnCreateTensorDescriptor(&db_desc);
    cudnnSetTensor4dDescriptor(db_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, OUT_C, 1, 1);

    int size_dy = N * OUT_C * OUT_H * OUT_W;
    int size_db = OUT_C;

    // Allocate host memory
    float *h_dy, *h_db;
    h_dy = (float*)malloc(size_dy * sizeof(float));
    h_db = (float*)malloc(size_db * sizeof(float));

    // Initialization
    rand_data(h_dy, size_dy, -1, 1);
    rand_data(h_db, size_db, 0, 0);

    // Allocate device memory
    float *d_dy, *d_db;
    cudaMalloc((void**)&d_dy, size_dy * sizeof(float));
    cudaMalloc((void**)&d_db, size_db * sizeof(float));

    // Memcpy: host -> device
    cudaMemcpy(d_dy, h_dy, size_dy * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_db, h_db, size_db * sizeof(float), cudaMemcpyHostToDevice);

    // Perform bgrad convolution
    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionBackwardBias(cudnn, &alpha, dy_desc, d_dy, &beta, db_desc, d_db);

    // Memcpy: device -> host
    cudaMemcpy(h_db, d_db, size_db * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compare
    float *calc_db = (float*)malloc(size_db * sizeof(float));
    dbgrad(h_dy, calc_db);
    float diff = 0.0f;
    for (int i = 0; i < size_db; ++i) {
        diff += (h_db[i] - calc_db[i]);
    }
    printf("\n--------diff:%f------\n", diff);

    // Clean up
    free(h_dy);
    free(h_db);
    free(calc_db);
    cudaFree(d_dy);
    cudaFree(d_db);
    cudnnDestroyTensorDescriptor(dy_desc);
    cudnnDestroyTensorDescriptor(db_desc);
    cudnnDestroy(cudnn);

    return 0;
}
