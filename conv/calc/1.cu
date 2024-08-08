/*
 * 方法一：定义
 */

#define IX(n, h, w, c) ((((n)*H + h) * W + w) * C + c)
#define IW(c, r, s, m) ((((c)*R + r) * S + s) * M + m)
#define IY(n, e, f, m) ((((n)*E + e) * F + f) * M + m)

const int PH = pad_h;
const int PW = pad_w;
const int SH = stride_h;
const int SW = stride_w;
const int DH = dilation_h;
const int DW = dilation_w;
const int DR = (DH - 1) * (R - 1) + R;
const int DS = (DW - 1) * (S - 1) + S;

for (int n = 0; n < N; ++n) {
    for (int e = 0; e < E; ++e) {
        for (int f = 0; f < F; ++f) {
            for (int m = 0; m < M; ++m) {
                float tmp = 0.0f;
                for (int r = 0; r < DR; r += DH) {
                    for (int s = 0; s < DS; s += DW) {
                        int real_h = e * SH + r - PH;
                        int real_w = f * SW + s - PW;
                        if (real_h >= 0 && real_h < H && real_w >= 0 && real_w < W) {
                            int real_r = r / DH;
                            int real_s = s / DW;
                            for (int c = 0; c < C; ++c) {
                                tmp += (float)x[IX(n, real_h, real_w, c)] * (float)w[IW(c, real_r, real_s, m)];
                            }
                        }
                    }
                }
                y[IY(n, e, f, m)] = tmp;
            }
        }
    }
}