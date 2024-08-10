### conv计算方法

### TODO

1. 定义
2. Img2col+gemm

conv运算

```math
\begin{pmatrix}
	x_{11} & x_{12} & x_{13} \\
	x_{21} & x_{22} & x_{23} \\
	x_{31} & x_{32} & x_{33}
\end{pmatrix}
\ast
\begin{pmatrix}
	w_{11} & w_{12} \\
	w_{21} & w_{22}
\end{pmatrix}
+
b
=
\begin{pmatrix}
	y_{11} & y_{12} \\
	y_{21} & y_{22}
\end{pmatrix}
```

可以转换成gemm运算

```math
\begin{pmatrix}
	x_{11} & x_{12} & x_{21} & x_{22} \\
	x_{12} & x_{13} & x_{22} & x_{23} \\
	x_{21} & x_{22} & x_{31} & x_{32} \\
	x_{22} & x_{23} & x_{32} & x_{33}
\end{pmatrix}
\begin{pmatrix}
	w_{11} \\
	w_{12} \\
	w_{21} \\
	w_{22}
\end{pmatrix}
+
b
=
\begin{pmatrix}
	y_{11} \\
	y_{12} \\
	y_{21} \\
	y_{22}
\end{pmatrix}
```

即conv运算

$$ NHWC * CRSM = NEFM $$

可以转换为 $N$ 个gemm运算

$$ (EF, CDRDS) (CDRDS, M) = (EF, M)$$


3. FFT

时域中的卷积等效于频域中的乘法，所以可以对输入数据和卷积核做FFT后进行乘法运算再做IFFT来替代卷积运算。注意这里的卷积其实指的是互相关运算，卷积需要把卷积核旋转180°而互相关运算不用。


4. Winograd Algorithm

### TODO

- [ ] FFT：python 1D -> CUDA 2D
- [ ] Winograd Algorithm