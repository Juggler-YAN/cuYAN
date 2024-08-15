/*
 * 定义
 */

void def(const float *A, const float *B, float *C, const int M, const int N, const int K) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float tmp = 0.0f;
      for (int k = 0; k < K; ++k) {
        tmp += A[m * K + k] * B[k * N + n];
      }
      C[m * N + n] = tmp;
    }
  }
}