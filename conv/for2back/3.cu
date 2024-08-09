/*
 * wgrad conv
 */

#define IX(n, h, w, c) ((((n)*H + h) * W + w) * C + c)
#define IDW(c, r, s, m) ((((c)*R + r) * S + s) * M + m)
#define IDY(n, e, f, m) ((((n)*E + e) * F + f) * M + m)
		
// forward
const int PH = pad_h;                                           // x, pad
const int PW = pad_w;
const int SH = stride_h;                                        // x, stride
const int SW = stride_w;
const int DH = dilation_h;                                      // w, dilation
const int DW = dilation_w;
// reverse
const int DE = (SH - 1) * (E - 1) + E;                          // dy, stride->dilation
const int DF = (SW - 1) * (F - 1) + F;

for (int c = 0; c < C; ++c) {
    for (int r = 0; r < R; ++r) {
        for (int s = 0; s < S; ++s) {
            for (int m = 0; m < M; ++m) {
                float tmp = 0.0f;
                for (int e = 0; e < DE; e += SH) {
                    for (int f = 0; f < DF; f += SW) {
                        int real_h = r * DH + e - PH;                   // dx, index
                        int real_w = s * DW + f - PW;
                        if (real_h >= 0 && real_h < H && real_w >= 0 && real_w < W) {
                            int real_e = e / SH;                        // dy, index
                            int real_f = f / SW;
                            for (int n = 0; n < N; ++n) {
                                tmp += (float)x[IX(n, real_h, real_w, c)] * (float)dy[IDY(n, real_e, real_f, m)];
                            }
                        }
                    }
                }
                dw[IDW(c, r, s, m)] = tmp;
            }
        }
    }
}