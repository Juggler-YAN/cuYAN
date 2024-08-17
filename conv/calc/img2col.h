/*
 * img2col
 */

void img2col(const float* input, const float* filter, float* output, const int N, const int IN_C, const int IN_H,
    const int IN_W, const int K_H, const int K_W, const int OUT_C, const int OUT_H, const int OUT_W, const int PAD_H,
    const int PAD_W, const int STRIDE_H, const int STRIDE_W, const int DILATION_H, const int DILATION_W) {

#define IX(n, in_c, in_h, in_w) ((((n) * IN_C + in_c) * IN_H + in_h) * IN_W + in_w)
#define IW(out_c, in_c, k_h, k_w) ((((out_c) * IN_C + in_c) * K_H + k_h) * K_W + k_w)
#define IY(n, out_c, out_h, out_w) ((((n) * OUT_C + out_c) * OUT_H + out_h) * OUT_W + out_w)

    const int DILA_K_H = (K_H - 1) * (DILATION_H - 1) + K_H;
    const int DILA_K_W = (K_W - 1) * (DILATION_W - 1) + K_W;

    for (int n = 0; n < N; ++n) {
        // (1,Cin,Hin,Win) * (Cout,Cin,Hk,Wk) = (1,Cout,Hout,Wout)
        float *A = (float *)malloc((OUT_H * OUT_W) * (IN_C * K_H * K_W) * sizeof(float));
        float *B = (float *)malloc((IN_C * K_H * K_W) * OUT_C * sizeof(float));
        float *C = (float *)malloc(OUT_C * (OUT_H * OUT_W) * sizeof(float));
        // 1.transform (1,Cin,Hin,Win) to matrix A(Hout*Wout,Cin*Hk*Wk), according to pad, stride and dilation
        for (int out_h = 0; out_h < OUT_H; ++out_h) {
            for (int out_w = 0; out_w < OUT_W; ++out_w) {
                for (int in_c = 0; in_c < IN_C; ++in_c) {
                    for (int k_h = 0; k_h < DILA_K_H; k_h += DILATION_H) {
                        for (int k_w = 0; k_w < DILA_K_W; k_w += DILATION_W) {
                            int real_in_h = out_h * STRIDE_H + k_h - PAD_H;
                            int real_in_w = out_w * STRIDE_W + k_w - PAD_W;
                            if (real_in_h >= 0 && real_in_h < IN_H && real_in_w >= 0 && real_in_w < IN_W) {
                                int pos = (out_h * OUT_W + out_w) * (IN_C * DILA_K_H * DILA_K_W) + 
                                    in_c * DILA_K_H * DILA_K_W + k_h * DILA_K_W + k_w;
                                A[pos] = input[IX(n, in_c, real_in_h, real_in_w)];
                            }
                        }
                    }
                }
            }
        }
        // 2.transform (Cout,Cin,Hk,Wk) to matrix B(Cin*Hk*Wk,Cout), according to dilation
        for (int out_c = 0; out_c < OUT_C; ++out_c) {
            for (int in_c = 0; in_c < IN_C; ++in_c) {
                for (int k_h = 0; k_h < DILA_K_H; k_h += DILATION_H) {
                    for (int k_w = 0; k_w < DILA_K_W; k_w += DILATION_W) {
                        int real_k_h = k_h / DILATION_H;
                        int real_k_w = k_w / DILATION_W;
                        int pos = (in_c * DILA_K_H * DILA_K_W + k_h * DILA_K_W + k_w) * OUT_C + out_c;
                        B[pos] = filter[IW(out_c, in_c, real_k_h, real_k_w)];
                    }
                }
            }
        }
        // 3.gemm A(Hout*Wout,Cin*Hk*Wk) * B(Cin*Hk*Wk,Cout) = C(Hout*Wout,Cout)
        for (int out_c = 0; out_c < OUT_C; ++out_c) {
            for (int out_h = 0; out_h < OUT_H; ++out_h) {
                for (int out_w = 0; out_w < OUT_W; ++out_w) {
                    float temp = 0.0f;
                    for (int k = 0; k < IN_C * DILA_K_H * DILA_K_W; ++k) {
                        temp += (float)A[(out_h * OUT_W + out_w) * (IN_C * DILA_K_H * DILA_K_W) + k] * 
                            (float)B[k * OUT_C + out_c];
                    }
                    C[IY(0, out_c, out_h, out_w)] = temp;
                }
            }
        }
        // 4.write back
        for (int i = 0; i < OUT_C * OUT_H * OUT_W; ++i) {
            output[n * OUT_C * OUT_H * OUT_W + i] = C[i];
        }
        free(A);
        free(B);
        free(C);
    }

}