#include <iostream>
#include <complex>
#include <cmath>

using namespace std;

void printcomplex(const complex<float>* Acomplex, const int M, const int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            cout << Acomplex[i*N+j].real() << (Acomplex[i*N+j].imag() >= 0 ? "+" : "") <<
                Acomplex[i*N+j].imag() << "j" << "    ";
        }
        cout << endl;
    }
}

void real2complex(const float* Areal, complex<float>* Acomplex, const int M, const int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            Acomplex[i*N+j].real(Areal[i*N+j]);
            Acomplex[i*N+j].imag(0.0f);
        }
    }
}

int main() {

    // input
    const int M = 3, N = 3;
    float Areal[] = {
        0, 1, 2,
        1, 2, 3,
        2, 3, 4
    };

    complex<float>* Acomplex = (complex<float>*)malloc(M * N * sizeof(complex<float>));
    real2complex(Areal, Acomplex, M, N);
    // printcomplex(Acomplex, M, N);

    int Mfft = pow(2, ceil(log2((float)M)));
    int Nfft = pow(2, ceil(log2((float)N)));
    // cout << Mfft << endl;
    // cout << Nfft << endl;
    complex<float>* Afftcomplex = (complex<float>*)malloc(Mfft * Nfft * sizeof(complex<float>));


    // free
    free(Acomplex);
    free(Afftcomplex);

    return 0;
}