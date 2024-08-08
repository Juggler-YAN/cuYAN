/*
 * 方法二：Img2col+gemm
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
    // require additional HBM to save matrix A and B
    memset(ws, 0, workSpaceSize);
    half *ws1 = ws;
    half *ws2 = ws + E * F * C * DR * DS;
    // 1HWC * CRSM = 1EFM
    // transform 1HWC to matrix A(EF,CDRDS), according to pad, stride and dilation
    for (int e = 0; e < E; ++e) {
        for (int f = 0; f < F; ++f) {
            for (int c = 0; c < C; ++c) {
                for (int r = 0; r < DR; r += DH) {
                    for (int s = 0; s < DS; s += DW) {
                        int real_h = e * SH + r - PH;
                        int real_w = f * SW + s - PW;
                        if (real_h >= 0 && real_h < H && real_w >= 0 && real_w < W) {
                            int pos = (e * F + f) * (C * DR * DS) + c * DR * DS + r * DS + s;
                            ws1[pos] = x[IX(n, real_h, real_w, c)];
                        }
                    }
                }
            }
        }
    }
    // transform CRSM to matrix B(CDRDS,M), according to dilation
    for (int c = 0; c < C; ++c) {
        for (int r = 0; r < DR; r += DH) {
            for (int s = 0; s < DS; s += DW) {
                for (int m = 0; m < M; ++m) {
                    int real_r = r / DH;
                    int real_s = s / DW;
                    int pos = (c * DR * DS + r * DS + s) * M + m;
                    ws2[pos] = w[IW(c, real_r, real_s, m)];
                }
            }
        }
    }
    // gemm A(EF,CDRDS) * B(CDRDS,M) = C(EF,M)
    for (int e = 0; e < E; ++e) {
        for (int f = 0; f < F; ++f) {
            for (int m = 0; m < M; ++m) {
                float temp = 0.0f;
                for (int k = 0; k < C * DR * DS; ++k) {
                    temp += (float)ws1[(e * F + f) * C * DR * DS + k] * (float)ws2[k * M + m];
                }
                // write back
                y[IY(n, e, f, m)] = temp;
            }
        }
    }
}