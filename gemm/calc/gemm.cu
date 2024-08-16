#include <iostream>
#include <cublas_v2.h>
#include "def.h"
#include "cannon.h"
#include "summa.h"

using namespace std;

#define M 8
#define K 16
#define N 12

void rand_data(float *data, int num, float min, float max) {
    for (int i = 0; i < num; i++) {
        data[i] = (fabs(max - min) < 1e-5) ? min : ((max - min) * (rand() / (float)RAND_MAX) + min);
    }
}

int main() {

    // Initialize CUDA
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        cerr << "CUDA initialization failed!" << endl;
        return 1;
    }

    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    int A_num = M * K;
    int B_num = K * N;
    int C_num = M * N;

    // Allocate host memory for A, B, and C
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(A_num * sizeof(float));
    h_B = (float*)malloc(B_num * sizeof(float));
    h_C = (float*)malloc(C_num * sizeof(float));

    // Initialization
    rand_data(h_A, A_num, -1, 1);
    rand_data(h_B, B_num, -1, 1);
    rand_data(h_C, C_num, 0, 0);

    // Allocate device memory for A, B, and C
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, A_num * sizeof(float));
    cudaMalloc((void**)&d_B, B_num * sizeof(float));
    cudaMalloc((void**)&d_C, C_num * sizeof(float));

    // Memcpy: host -> device
    cudaMemcpy(d_A, h_A, A_num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, B_num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, C_num * sizeof(float), cudaMemcpyHostToDevice);

    // Perform GEMM
    float a = 1, b = 0;
    cublasSgemm(
        handle,
        CUBLAS_OP_T,    // 矩阵A的属性参数，转置，按行优先
        CUBLAS_OP_T,    // 矩阵B的属性参数，转置，按行优先
        M,              // 矩阵A、C的行数
        N,              // 矩阵B、C的列数
        K,              // A的列数，B的行数
        &a,             // alpha
        d_A,            // 左矩阵A
        K,              // A的leading dimension，此时选择转置，按行优先，则leading dimension为A的列数
        d_B,            // 右矩阵B
        N,              // B的leading dimension，此时选择转置，按行优先，则leading dimension为B的列数
        &b,             // beta
        d_C,            // 结果矩阵C
        M               // C的leading dimension，C矩阵一定按列优先，则leading dimension为C的行数
    );

    // Memcpy: device -> host
    cudaMemcpy(h_C, d_C, C_num * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare
    float *calc_C = (float*)malloc(C_num * sizeof(float));
    // def(h_A, h_B, calc_C, M, N, K);
    cannon(h_A, h_B, calc_C, M, N, K, 16);
    // summa(h_A, h_B, calc_C, M, N, K, 4, 3);
    // summa(h_A, h_B, calc_C, M, N, K, 2, 4);
    float diff = 0.0f;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            diff += h_C[j*M+i] - calc_C[i*N+j];
        }
    }
    printf("\n--------diff:%f------\n", diff);


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}