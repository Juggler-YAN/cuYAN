/*
 * 方法三：FFT
 */

import numpy as np

def convfft(x, w, p, s, d):
    # 直接由fft实现时, p=N-1, s=1, d=1
    # p<N-1, y前后截断；p>N-1, y前后补0
    # s>1, 对x进行补0, x隔一个增加s-1个0
    # d>1, 对w进行补0, w每隔一个增加d-1个0
    # 1.stride
    xlen = len(x)
    M = xlen + (xlen - 1) * (s - 1)
    newx = [0] * M
    for i in range(0,M,s):
        newx[i] = x[(int)(i/s)]
    print(newx)
    # 2.reverse + dilation
    wlen = len(w)
    N = wlen + (wlen - 1) * (d - 1)
    neww = [0] * N
    for i in range(0,N,d):
        neww[i] = w[(int)(i/d)]
    print(neww)
    neww.reverse()
    # 3.FFT
    MN = M + N - 1
    L = 2 ** (int(np.log2(MN)) + 1)
    xfft = np.fft.fft(newx, L)
    wfft = np.fft.fft(neww, L)
    yfft = xfft * wfft
    # 4.padding
    if p <= N - 1:
        ybeg = N-1-p
        yend = MN-(N-1-p)
        y = np.fft.ifft(yfft).real[ybeg:yend]
    else:
        y = np.fft.ifft(yfft).real[:MN]
        y = np.pad(y, p-(N-1))
    return y

if __name__ == '__main__':
    x = [0, 1, 2, 3, 4, 5]
    w = [0, 1, 2]
    padding = 5
    stride = 1
    dilation = 2
    print(convfft(x,w,padding,stride,dilation))
