/*
 * 方法一：定义
 */

#define IX(n, h, w, c) ((((n)*H + h) * W + w) * C + c)
#define IW(c, r, s, m) ((((c)*R + r) * S + s) * M + m)
#define IY(n, e, f, m) ((((n)*E + e) * F + f) * M + m)

const int ph = pad_h;
const int pw = pad_w;
const int sh = stride_h;
const int sw = stride_w;
const int dh = dilation_h;
const int dw = dilation_w;
const int dr = (dh - 1) * (R - 1) + R;
const int ds = (dw - 1) * (S - 1) + S;

for (int n = 0; n < N; ++n) {
    for (int e = 0; e < E; ++e) {
        for (int f = 0; f < F; ++f) {
            for (int m = 0; m < M; ++m) {
                float tmp = 0.0f;
                for (int r = 0; r < dr; r += dh) {
                    for (int s = 0; s < ds; s += dw) {
                        int real_h = e * sh + r - ph;
                        int real_w = f * sw + s - pw;
                        if (real_h >= 0 && real_h < H && real_w >= 0 && real_w < W) {
                            int real_r = r / dh;
                            int real_s = s / dw;
                            for (int c = 0; c < C; ++c) {
                                tmp += x[IX(n, real_h, real_w, c)] * w[IW(c, real_r, real_s, m)];
                            }
                        }
                    }
                }
                y[IY(n, e, f, m)] = tmp;
            }
        }
    }
}