/*
 * cannon
 */

void matrixLeft(float *A, const int M, const int K, const int P, const int row, const int shift) {
    if (shift == 0) return;
    const int beg = (M / P) * row;
    const int end = (M / P) * (row + 1);
    const int num = (K / P) * shift;
    for (int i = beg; i < end; ++i) {
        // shift
        float *tmp = (float*)malloc(K * sizeof(float));
        int j = 0;
        for (int k = num; k < K; ++k) {
            tmp[j++] = A[i * K + k];
        }
        for (int k = 0; k < num; ++k) {
            tmp[j++] = A[i * K + k];
        }
        // write_back
        for (int k = 0; k < K; ++k) {
            A[i * K + k] = tmp[k];
        }
    }
}

void matrixUp(float *B, const int K, const int N, const int P, const int col, const int shift) {
    if (shift == 0) return;
    const int beg = (N / P) * col;
    const int end = (N / P) * (col + 1);
    const int num = (K / P) * shift;
    for (int j = beg; j < end; ++j) {
        // shift
        float *tmp = (float*)malloc(K * sizeof(float));
        int i = 0;
        for (int k = num; k < K; ++k) {
            tmp[i++] = B[k * N + j];
        }
        for (int k = 0; k < num; ++k) {
            tmp[i++] = B[k * N + j];
        }
        // write_back
        for (int k = 0; k < K; ++k) {
            B[k * N + j] = tmp[k];
        }
    }
}

void mma(const float *A, const float *B, float *C, const int M, const int N, const int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float tmp = 0.0f;
            for (int k = 0; k < K; ++k) {
                tmp += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] += tmp;
        }
    }
}

void cannon(const float *Ainit, const float *Binit, float *C, const int M, const int N, const int K, const int Pnum) {
    float *A = (float*)malloc(M * K * sizeof(float));
    float *B = (float*)malloc(K * N * sizeof(float));
    memcpy(A, Ainit, M * K * sizeof(float));
    memcpy(B, Binit, K * N * sizeof(float));
    const int P = sqrt(Pnum);
    // 1. Aij循环左移i位，Bij循环上移j位
    for (int i = 0; i < P; ++i) {
        matrixLeft(A, M, K, P, i, i);
    }
    for (int j = 0; j < P; ++j) {
        matrixUp(B, K, N, P, j, j);
    }
    // 2.每个进程中的子矩阵相乘并累加
    float **allSubC = (float**)malloc(P * P * sizeof(float*));
    for (int i = 0; i < P * P; ++i) {
        allSubC[i] = (float*)malloc((M / P) * (N / P) * sizeof(float));
        memset(allSubC[i], 0, (M / P) * (N / P) * sizeof(float));
    }
    for (int i = 0; i < P; ++i) {
        for (int j = 0; j < P; ++j) {
            float *subA = (float*)malloc((M / P) * (K / P) * sizeof(float));
            float *subB = (float*)malloc((K / P) * (N / P) * sizeof(float));
            for (int m = 0; m < M / P; ++m) {
                for (int k = 0; k < K / P; ++k) {
                    subA[m * K / P + k] = A[(i * M / P + m) * K + j * K / P + k];
                }
            }
            for (int k = 0; k < K / P; ++k) {
                for (int n = 0; n < N / P; ++n) {
                    subB[k * N / P + n] = B[(i * K / P + k) * N + j * N / P + n];
                }
            }
            mma(subA, subB, allSubC[i * P + j], M / P, N / P, K / P);
            free(subA);
            free(subB);
        }
    }
    // 3.循环P-1次
    for (int p = 0; p < P - 1; ++p) {
        // 3.1. Aij循环左移1位，Bij循环上移1位
        for (int i = 0; i < P; ++i) {
            matrixLeft(A, M, K, P, i, 1);
        }
        for (int j = 0; j < P; ++j) {
            matrixUp(B, K, N, P, j, 1);
        }
        // 3.2.每个进程中的子矩阵相乘并累加
        for (int i = 0; i < P; ++i) {
            for (int j = 0; j < P; ++j) {
                float *subA = (float*)malloc((M / P) * (K / P) * sizeof(float));
                float *subB = (float*)malloc((K / P) * (N / P) * sizeof(float));
                for (int m = 0; m < M / P; ++m) {
                    for (int k = 0; k < K / P; ++k) {
                        subA[m * K / P + k] = A[(i * M / P + m) * K + j * K / P + k];
                    }
                }
                for (int k = 0; k < K / P; ++k) {
                    for (int n = 0; n < N / P; ++n) {
                        subB[k * N / P + n] = B[(i * K / P + k) * N + j * N / P + n];
                    }
                }
                mma(subA, subB, allSubC[i * P + j], M / P, N / P, K / P);
                free(subA);
                free(subB);
            }
        }
    }
    // 4.写回子矩阵
    for (int i = 0; i < P; ++i) {
        for (int j = 0; j < P; ++j) {
            for (int m = 0; m < M / P; ++m) {
                for (int n = 0; n < N / P; ++n) {
                    C[(i * M / P + m) * N + j * N / P + n] = allSubC[i * P + j][m * N / P + n];
                }
            }
        }
    }
    free(A);
    free(B);
    for (int i = 0; i < P * P; ++i) {
        free(allSubC[i]);
    }
    free(allSubC);
}

