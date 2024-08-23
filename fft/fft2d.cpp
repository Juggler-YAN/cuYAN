#include <iostream>
#include <complex>
#include <cmath>

using namespace std;

#define PI acos(-1.0)

void print(const complex<float> *A, const int M, const int N) {
    printf("\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            cout << A[i*N+j].real() << "    ";
            // cout << A[i*N+j].real() << (A[i*N+j].imag() >= 0 ? "+" : "") << A[i*N+j].imag() << "    ";
        }
        cout << endl;
    }
    printf("\n");
}

// inv == 1 为 FFT， inv == -1 为 IFFT
void fft(complex<float> *A, int L, int inv) {
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
void fft2(complex<float> *A, int L1, int L2, int inv) {
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

int main() {
    
    // input
    const int M = 3, N = 3;
    float A[M*N] = {
        0, 1, 2,
        1, 2, 3,
        2, 3, 4
    };

    int L1 = pow(2, ceil(log2(M)));
    int L2 = pow(2, ceil(log2(N)));
    complex<float> *Afft = (complex<float> *)malloc(L1 * L2 * sizeof(complex<float>));
    complex<float> *AfftTran = (complex<float> *)malloc(L1 * L2 * sizeof(complex<float>));
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            Afft[i*L2+j].real(A[i*N+j]);
        }
    }

    fft2(Afft, L1, L2, 1);
    print(Afft, L1, L2);
    fft2(Afft, L1, L2, -1);
    print(Afft, L1, L2);
    
    free(Afft);
    free(AfftTran);
    
    return 0;
}