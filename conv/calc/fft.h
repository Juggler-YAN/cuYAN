/*
 * fft
 */

#include <complex>
#include <cmath>

using namespace std;

#define PI acos(-1.0)

// inv == 1 为 FFT， inv == -1 为 IFFT
void fft(complex<float> *A, const int L, const int inv) {
    // 位反转置换
    for (int i = 1, j = 0; i < L; ++i) {
        for (int k = L >> 1; (j ^= k) < k; k >>= 1);
        if (i < j) swap(A[i], A[j]);
    }
    // Cooley-Tukey算法
    for (int len = 2; len <= L; len <<= 1) {
        float ang = inv * 2 * PI / len;
        complex<float> wlen(cos(ang), sin(ang));
        for (int i = 0; i < L; i += len) {
            complex<float> w(1, 0);
            for (int j = 0, mid = len >> 1; j < mid; ++j) {
                complex<float> u = A[i + j], v = w * A[i + j + mid];
                A[i + j] = u + v;
                A[i + j + mid] = u - v;
                w *= wlen;
            }
        }
    }
    // 不加这一步和 python 以及 matlab 库 算出来的结果不一样
    for (int i = 1; i < L - i; ++i) {
        swap(A[i], A[L-i]);
    }
    if (inv == -1) {
        for (int i = 0; i < L; ++i) {
            A[i].real(A[i].real() / L);
            A[i].imag(A[i].imag() / L);
        }
    }
}

// 2D fft 相当于先每行再每列或者先每列再每行进行 fft
void fft2(complex<float> *A, const int L1, const int L2, const int inv) {
    complex<float> *ATran = (complex<float> *)malloc(L1 * L2 * sizeof(complex<float>));
    for (int i = 0; i < L1; ++i) {
        fft(A+i*L2, L2, inv);
    }
    for (int i = 0; i < L1; ++i) {
        for (int j = 0; j < L2; ++j) {
            ATran[j*L1+i] = A[i*L2+j];
        }
    }
    for (int j = 0; j < L2; ++j) {
        fft(ATran+j*L1, L1, inv);
    }
    for (int i = 0; i < L1; ++i) {
        for (int j = 0; j < L2; ++j) {
            A[i*L2+j] = ATran[j*L1+i];
        }
    }
    free(ATran);
}

void convfft(const float* x, const float* w, float* y, const int N, const int IN_C, const int IN_H,
    const int IN_W, const int K_H, const int K_W, const int OUT_C, const int OUT_H, const int OUT_W, const int PAD_H,
    const int PAD_W, const int STRIDE_H, const int STRIDE_W, const int DILATION_H, const int DILATION_W) {

#define IX(n, in_c, in_h, in_w) ((((n) * IN_C + in_c) * IN_H + in_h) * IN_W + in_w)
#define IW(out_c, in_c, k_h, k_w) ((((out_c) * IN_C + in_c) * K_H + k_h) * K_W + k_w)
#define IY(n, out_c, out_h, out_w) ((((n) * OUT_C + out_c) * OUT_H + out_h) * OUT_W + out_w)

    const int L_H = pow(2, ceil(log2(IN_H + K_H - 1)));
    const int L_W = pow(2, ceil(log2(IN_W + K_W - 1)));

    for (int n = 0; n < N; ++n) {
        for (int out_c = 0; out_c < OUT_C; ++out_c) {
            for (int in_c = 0; in_c < IN_C; ++in_c) {
                complex<float> *xfft = (complex<float> *)malloc(L_H * L_W * sizeof(complex<float>));
                complex<float> *wfft = (complex<float> *)malloc(L_H * L_W * sizeof(complex<float>));
                complex<float> *yfft = (complex<float> *)malloc(L_H * L_W * sizeof(complex<float>));
                // 1. init
                for (int in_h = 0; in_h < L_H; ++in_h) {
                    for (int in_w = 0; in_w < L_W; ++in_w) {
                        if (in_h < IN_H && in_w < IN_W) {
                            xfft[in_h*L_W+in_w].real(x[IX(n, in_c, in_h, in_w)]);
                        } else {
                            xfft[in_h*L_W+in_w].real(0.0f);
                        }
                        xfft[in_h*L_W+in_w].imag(0.0f);
                    }
                }
                for (int k_h = 0; k_h < L_H; ++k_h) {
                    for (int k_w = 0; k_w < L_W; ++k_w) {
                        if (k_h < K_H && k_w < K_W) {
                            // rot 180°
                            wfft[k_h*L_W+k_w].real(w[IW(out_c, in_c, K_H - k_h - 1, K_W - k_w - 1)]);
                        } else {
                            wfft[k_h*L_W+k_w].real(0.0f);
                        }
                        wfft[k_h*L_W+k_w].imag(0.0f);
                    }
                }

                // 2.FFT
                fft2(xfft, L_H, L_W, 1);
                fft2(wfft, L_H, L_W, 1);

                // 3.相乘
                for (int out_h = 0; out_h < L_H; ++out_h) {
                    for (int out_w = 0; out_w < L_W; ++out_w) {
                        yfft[out_h * L_W + out_w] = xfft[out_h * L_W + out_w] * wfft[out_h * L_W + out_w];
                    }
                }

                // 4.IFFT
                fft2(yfft, L_H, L_W, -1);

                // 5.write_back
                for (int out_h = 0; out_h < OUT_H; ++out_h) {
                    for (int out_w = 0; out_w < OUT_W; ++out_w) {
                        // 去pad
                        y[IY(n, out_c, out_h, out_w)] += yfft[(out_h + (K_H - 1)) * L_W + out_w + (K_W - 1)].real();
                    }
                }

                free(xfft);
                free(wfft);
                free(yfft);
            }
        }
    }
}