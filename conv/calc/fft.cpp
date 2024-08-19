#include <iostream>
#include <cmath>

using namespace std;

// void convfft(float* x, float* w, )

int main() {
    float x[] = {0,1,2,1,2,3,3,4,5};
    float w[] = {0,1,1,2};
    int M1 = 3, M2 = 3, N1 = 2, N2 = 2;
    int MN1 = M1 + N1 - 1;
    int MN2 = M2 + N2 - 1;
    int L1 = pow(2, ceil(log2(MN1)));
    int L2 = pow(2, ceil(log2(MN2)));
    float w_trans[] = {0,0,0,0};
    for (int i = 0; i < N1; ++i) {
        for (int j = 0; j < N2; ++j) {
            w_trans[i * N2 + j] = w[(N1 - 1 - i) * N2 + (N2 - 1 - j)];
        }
    }
    // for (int i = 0; i < N1 * N2; ++i) {
    //     printf("%f ", w_trans[i]);
    // }
    printf("\n--%d:%d---\n", L1, L2);
    cout << "helloworld" << endl;
    return 0;
}