/*
 * slidingwindow
 */

void slidingwindow(const float* input, const float* filter, float* output, const int N, const int IN_C, const int IN_H,
    const int IN_W, const int K_H, const int K_W, const int OUT_C, const int OUT_H, const int OUT_W, const int PAD_H,
    const int PAD_W, const int STRIDE_H, const int STRIDE_W, const int DILATION_H, const int DILATION_W) {

#define IX(n, in_c, in_h, in_w) ((((n) * IN_C + in_c) * IN_H + in_h) * IN_W + in_w)
#define IW(out_c, in_c, k_h, k_w) ((((out_c) * IN_C + in_c) * K_H + k_h) * K_W + k_w)
#define IY(n, out_c, out_h, out_w) ((((n) * OUT_C + out_c) * OUT_H + out_h) * OUT_W + out_w)

    for (int n = 0; n < N; ++n) {
        for (int out_c = 0; out_c < OUT_C; ++out_c) {
            for (int out_h = 0; out_h < OUT_H; ++out_h) {
                for (int out_w = 0; out_w < OUT_W; ++out_w) {
                    float tmp = 0.0f;
                    for (int k_h = 0; k_h < (DILATION_H - 1) * (K_H - 1) + K_H; k_h += DILATION_H) {
                        for (int k_w = 0; k_w < (DILATION_W - 1) * (K_W - 1) + K_W; k_w += DILATION_W) {
                            int real_in_h = out_h * STRIDE_H + k_h - PAD_H;
                            int real_in_w = out_w * STRIDE_W + k_w - PAD_W;
                            if (real_in_h >= 0 && real_in_h < IN_H && real_in_w >= 0 && real_in_w < IN_W) {
                                int real_k_h = k_h / DILATION_H;
                                int real_k_w = k_w / DILATION_W;
                                for (int in_c = 0; in_c < IN_C; ++in_c) {
                                    tmp += (float)input[IX(n, in_c, real_in_h, real_in_w)] * (float)filter[IW(out_c, in_c, real_k_h, real_k_w)];
                                }
                            }
                        }
                    }
                    output[IY(n, out_c, out_h, out_w)] = tmp;
                }
            }
        }
    }

}