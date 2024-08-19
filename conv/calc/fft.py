import numpy as np

# def convfft(x, w):
#     w.reverse()
#     M = len(x)
#     N = len(w)
#     MN = M + N - 1
#     L = 2 ** (int(np.log2(MN)) + 1)
#     xfft = np.fft.fft(x, L)
#     wfft = np.fft.fft(w, L)
#     yfft = xfft * wfft
#     y = np.fft.ifft(yfft).real[:MN]
#     return y

# if __name__ == '__main__':
#     x = [0, 1, 2, 3, 4, 5]
#     w = [0, 1, 2]
#     print(convfft(x,w))

import numpy as np

def convfft(x, w):
    shape = np.array(x.shape) + np.array(w.shape) - 1
    fsize = 2 ** (np.ceil(np.log2(shape)).astype(int))
    # print(np.rot90(w, 2))
    B = np.fft.fft2(np.rot90(w, 2), fsize)
    A = np.fft.fft2(x, fsize)
    AB = np.fft.ifft2(A * B).real[:shape[0], :shape[1]]
    return AB

if __name__ == '__main__':
    x = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
    w = np.array([[0, 1], [1, 2]])
    print(convfft(x,w))