### conv计算方法

### TODO

1. 滑动窗口
2. Img2col

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

- 优点
- 缺点：要对卷积核旋转180°并且填充到和输入一样大，只有在输入和卷积核大小相近时效果会好一些

4. Winograd Algorithm

$F(m,n)$最小乘法次数是$m+n-1$次，$F(m \times m,n \times n)$是$(m+n-1)(m+n-1)$次

$F(m,n)$以$F(2, 3)$为例，

```math
\begin{pmatrix}
	x_{1} & x_{2} & x_{3} & x_{4}
\end{pmatrix}
\ast
\begin{pmatrix}
	w_{1} & w_{2} & w_{3}
\end{pmatrix}
=
\begin{pmatrix}
	y_{1} & y_{2}
\end{pmatrix}
```

img2col算法如下

```math
\begin{pmatrix}
	x_{1} & x_{2} & x_{3} \\
	x_{2} & x_{3} & x_{4}
\end{pmatrix}
\begin{pmatrix}
	w_{1} \\
	w_{2} \\
	w_{3}
\end{pmatrix}
=
\begin{pmatrix}
	y_{1} \\
	y_{2}
\end{pmatrix}
```

可以看到，需要6次乘法4次加法

而Winograd算法如下

```math
\begin{pmatrix}
	x_{1} & x_{2} & x_{3} \\
	x_{2} & x_{3} & x_{4}
\end{pmatrix}
\begin{pmatrix}
	w_{1} \\
	w_{2} \\
	w_{3}
\end{pmatrix}
=
\begin{pmatrix}
	y_{1} + y_{2} + y_{3} \\
	y_{2} - y_{3} - y_{4}
\end{pmatrix}
```

其中，

```math
\left\{
\begin{aligned}
	y_{1} &= (x_1 - x_3)w_1 \\
	y_{2} &= (x_2 + x_3)\frac{w_1 + w_2 + w_3}{2} \\
	y_{3} &= (x_3 - x_2)\frac{w_1 - w_2 + w_3}{2} \\
	y_{4} &= (x_2 - x_4)w_3
\end{aligned}
\right.
```

可以看到，
- 输入：4次加法
- 卷积核：3次加法2次乘法，但一般卷积核是固定的，预先算一次就行，可以忽略
- 输出：4次加法4次乘法
  
总计4次乘法8次加法

相比于img2col，乘法次数减少，而计算机计算乘法要远比加法慢，因此起到了加速效果

$F(2, 3)$用矩阵可表示为

```math
Y=C^T[(Bw)\odot(A^Tx)]
```

其中,

```math
A^T
=
\begin{pmatrix}
	1 &  0 & -1 &  0 \\
	0 &  1 &  1 &  0 \\
	0 & -1 &  1 &  0 \\
	0 &  1 &  0 & -1
\end{pmatrix}
```

```math
B
=
\begin{pmatrix}
	1 & 0 & 0 \\
	\frac{1}{2} & \frac{1}{2} & \frac{1}{2} \\
	\frac{1}{2} & -\frac{1}{2} & \frac{1}{2} \\
	0 & 0 & 1
\end{pmatrix}
```

```math
C^T
=
\begin{pmatrix}
	1 & 1 & 1 &  0 \\
	0 & 1 & -1 & -1 \\
\end{pmatrix}
```

```math
x
=
\begin{pmatrix}
	x_1 & x_2 & x_3 & x_4
\end{pmatrix}^T
```

```math
w
=
\begin{pmatrix}
	w_1 & w_2 & w_3
\end{pmatrix}^T
```

```math
y
=
\begin{pmatrix}
	y_1 & y_2
\end{pmatrix}^T
```

$x$为输入，$w$为卷积核，$y$为输出，$A^T$为输入变形矩阵，$B$为卷积核变形矩阵，$C^T$为输出变形矩阵

根据shape的不同，$A$，$B$，$C$也会不同

$F(m \times m,n \times n)$以$F(2 \times 2,3 \times 3)$为例，

```math
\begin{pmatrix}
	x_{11} & x_{12} & x_{13} & x_{14} \\
	x_{21} & x_{22} & x_{23} & x_{24} \\
	x_{31} & x_{32} & x_{33} & x_{34} \\
	x_{41} & x_{42} & x_{43} & x_{44}
\end{pmatrix}
\ast
\begin{pmatrix}
	w_{11} & w_{12} & w_{13} \\
	w_{21} & w_{22} & w_{23} \\
	w_{31} & w_{32} & w_{33}
\end{pmatrix}
```

转换成

```math
\begin{pmatrix}
	x_{11} & x_{12} & x_{13} & x_{21} & x_{22} & x_{23} & x_{31} & x_{32} & x_{33} \\
	x_{12} & x_{13} & x_{14} & x_{22} & x_{23} & x_{24} & x_{32} & x_{33} & x_{34} \\
	x_{21} & x_{22} & x_{23} & x_{31} & x_{32} & x_{33} & x_{41} & x_{42} & x_{43} \\
	x_{22} & x_{23} & x_{24} & x_{32} & x_{33} & x_{34} & x_{42} & x_{43} & x_{44}
\end{pmatrix}
\begin{pmatrix}
	w_{11} \\
	w_{12} \\
	w_{13} \\
	w_{21} \\
	w_{22} \\
	w_{23} \\
	w_{31} \\
	w_{32} \\
	w_{33}
\end{pmatrix}
```

刚好可以均分成几个大块


```math
\begin{pmatrix}
	X_{1} & X_{2} & X_{3} \\
	X_{2} & X_{3} & X_{4}
\end{pmatrix}
\begin{pmatrix}
	W_{1} \\
	W_{2} \\
	W_{3}
\end{pmatrix}
```

这样就又变成了1D Winograd的问题

- 优点：减少乘法次数
- 缺点：根据不同shape要计算不同的$A$，$B$，$C$，而且shape过大$A$，$B$，$C$也会过大，需要存储空间会变大，加法次数也会变大，导致累加后精度变差

参考论文 [Winograd Algorithm](https://arxiv.org/pdf/1509.09308 "Winograd Algorithm")
