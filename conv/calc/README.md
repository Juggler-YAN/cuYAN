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


- [ ] FFT method
- [ ] Winograd Algorithm