#include <iostream>
#include <complex>
#include <cmath>

using namespace std;

#define PI acos(-1.0)

void print(complex<float> *A, int L) {
    printf("\n");
    for (int i = 0; i < L; ++i) {
        cout << A[i].real() << " " << A[i].imag() << endl;
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

int main() {
    
    // input
    const int M = 6;
    float A[M] = {0, 1, 2, 3, 4, 5};

    int L = pow(2, ceil(log2(M)));
    complex<float> *Afft = (complex<float> *)malloc(L * sizeof(complex<float>));
    for (int i = 0; i < M; ++i) {
        Afft[i].real(A[i]);
    }
    fft(Afft, L, 1);
    print(Afft, L);
    fft(Afft, L, -1);
    print(Afft, L);
    
    free(Afft);
    return 0;
}