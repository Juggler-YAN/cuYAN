/*
 * dgrad conv
 */

#define IDX(n, h, w, c) ((((n)*H + h) * W + w) * C + c)
#define IW(c, r, s, m) ((((c)*R + r) * S + s) * M + m)
#define IDY(n, e, f, m) ((((n)*E + e) * F + f) * M + m)

// forward
const int PH = pad_h; 							// x, pad
const int PW = pad_w;
const int SH = stride_h;						// x, stride
const int SW = stride_w;
const int DH = dilation_h; 						// w, dilation
const int DW = dilation_w;
// reverse
const int DR = (DH - 1) * (R - 1) + R; 			// w, dilation
const int DS = (DW - 1) * (S - 1) + S;
const int PE = DR - PH - 1; 					// dy, pad, k-p-1
const int PF = DS - PW - 1;
const int SE = (SH - 1) * (E - 1) + E; 			// dy, stride
const int SF = (SW - 1) * (F - 1) + F;

const int OPH = H - ((E - 1) * SH + DR - 2 * PH); // when stride is 2, out_padding may be needed
const int OPW = W - ((F - 1) * SW + DS - 2 * PW);

for (int n = 0; n < N; ++n) {
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            for (int c = 0; c < C; ++c) {
                _Float16 tmp = 0.0f;
                for (int r = 0; r < DR; r += DH) { // after dilation, value exists for w only in multiples of DH and DW
                    for (int s = 0; s < DS; s += DW) {
                        int real_r = (DR - 1 - r) / DH; // w, rot + index
                        int real_s = (DS - 1 - s) / DW;
                        if ((h + r - PE) % SH == 0 && (w + s - PF) % SW == 0) {  // after stride, value exists for dy only in multiples of SH and SW
                            int real_e = (h + r - PE) / SH; // dy, index
                            int real_f = (w + s - PF) / SW;
                            if (real_e >= 0 && real_e < E && real_f >= 0 && real_f < F) {
                                for (int m = 0; m < M; ++m) {
                                    tmp += dy[IDY(n, real_e, real_f, m)] * dw[IW(c, real_r, real_s, m)];
                                }
                            }
                        }
                    }
                }
                dx[IDX(n, h, w, c)] = tmp;
            }
        }
    }
}