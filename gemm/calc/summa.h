/*
 * summa
 */

void summa(const float *A, const float *B, float *C, const int M, const int N, const int K, const int P1, const int P2) {
    // 1. 广播A的子矩阵Ai到第i行
    // 2. 广播B的子矩阵Bj到第j列
    // 3. 对应进程中的子矩阵相乘
    for (int p1 = 0; p1 < P1; ++p1) {
        for (int p2 = 0; p2 < P2; ++p2) {
            const float *Ai = A + (p1 * M / P1) * K;
            const float *Bj = B + p2 * N / P2;
            float *Cij = (float*)malloc(M / P1 * N / P2 * sizeof(float));
            for (int m = 0; m < M / P1; ++m) {
                for (int n = 0; n < N / P2; ++n) {
                    float tmp = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        tmp += Ai[m * K + k] * Bj[k * N + n];
                    }
                    Cij[m * N / P2 + n] = tmp;
                }
            }
            // 4.将Cij写回C
            for (int m = 0; m < M / P1; ++m) {
                for (int n = 0; n < N / P2; ++n) {
                    C[(p1 * M / P1 + m) * N + p2 * N / P2 + n] = Cij[m * N / P2 + n];
                }
            }
            free(Cij);
        }
    }
}