# conv 优化方案

1. 转换成 $1 \times 1$ conv

$H \times W$ 大小的卷积核可以转换成 $H \times W$ 个 $1 \times 1$ 卷积核

举个例子，

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
=
\begin{pmatrix}
	y_{11} & y_{12} \\
	y_{21} & y_{22}
\end{pmatrix}
```

是不是可以拆成

```math
\begin{pmatrix}
	x_{11} & x_{12} \\
	x_{21} & x_{22}
\end{pmatrix}
\ast
\begin{pmatrix}
	w_{11}
\end{pmatrix}
=
\begin{pmatrix}
	a_{11} & a_{12} \\
	a_{21} & a_{22}
\end{pmatrix}
```

```math
\begin{pmatrix}
	x_{12} & x_{13} \\
	x_{22} & x_{23}
\end{pmatrix}
\ast
\begin{pmatrix}
	w_{12}
\end{pmatrix}
=
\begin{pmatrix}
	b_{11} & b_{12} \\
	b_{21} & b_{22}
\end{pmatrix}
```

```math
\begin{pmatrix}
	x_{21} & x_{22} \\
	x_{31} & x_{32}
\end{pmatrix}
\ast
\begin{pmatrix}
	w_{21}
\end{pmatrix}
=
\begin{pmatrix}
	c_{11} & c_{12} \\
	c_{21} & c_{22}
\end{pmatrix}
```

```math
\begin{pmatrix}
	x_{22} & x_{23} \\
	x_{32} & x_{33}
\end{pmatrix}
\ast
\begin{pmatrix}
	w_{22}
\end{pmatrix}
=
\begin{pmatrix}
	d_{11} & d_{12} \\
	d_{21} & d_{22}
\end{pmatrix}
```

这几个结果累加起来

```math
\begin{pmatrix}
	a_{11} & a_{12} \\
	a_{21} & a_{22}
\end{pmatrix}
+
\begin{pmatrix}
	b_{11} & b_{12} \\
	b_{21} & b_{22}
\end{pmatrix}
+
\begin{pmatrix}
	c_{11} & c_{12} \\
	c_{21} & c_{22}
\end{pmatrix}
+
\begin{pmatrix}
	d_{11} & d_{12} \\
	d_{21} & d_{22}
\end{pmatrix}
=
\begin{pmatrix}
	y_{11} & y_{12} \\
	y_{21} & y_{22}
\end{pmatrix}
```

- 优点：避免因为pad和dilation产生的补0操作
- 缺点：本来能够进行数据复用的现在不行了（stride不够大）

2. 转换成 stride_h * stride_w 组 stride = 1 的 conv

H和W维分别按stride_h和stride_w循环切分，举个例子，stride_h = stride_w = 2

```math
\begin{pmatrix}
	x_{11} & x_{12} & x_{13} & x_{14} & x_{15} & x_{16} \\
	x_{21} & x_{22} & x_{23} & x_{24} & x_{25} & x_{26} \\
	x_{31} & x_{32} & x_{33} & x_{34} & x_{35} & x_{36} \\
	x_{41} & x_{42} & x_{43} & x_{44} & x_{45} & x_{46} \\
	x_{51} & x_{52} & x_{53} & x_{54} & x_{55} & x_{56} \\
	x_{61} & x_{62} & x_{63} & x_{64} & x_{65} & x_{66} 
\end{pmatrix}
\ast
\begin{pmatrix}
	w_{11} & w_{12} & w_{13} & w_{14} \\
	w_{21} & w_{22} & w_{23} & w_{24} \\
	w_{31} & w_{32} & w_{33} & w_{34} \\
	w_{41} & w_{42} & w_{43} & w_{44}
\end{pmatrix}
=
\begin{pmatrix}
	y_{11} & y_{12} \\
	y_{21} & y_{22}
\end{pmatrix}
```

是不是可以刚好拆成 4 组

```math
\begin{pmatrix}
	x_{11} & x_{13} & x_{15} \\
	x_{31} & x_{33} & x_{35} \\
	x_{51} & x_{53} & x_{55}
\end{pmatrix}
\ast
\begin{pmatrix}
	w_{11} & w_{13} \\
	w_{31} & w_{33}
\end{pmatrix}
=
\begin{pmatrix}
	a_{11} & a_{12} \\
	a_{21} & a_{22}
\end{pmatrix}
```

```math
\begin{pmatrix}
	x_{12} & x_{14} & x_{16} \\
	x_{32} & x_{34} & x_{36} \\
	x_{52} & x_{54} & x_{56}
\end{pmatrix}
\ast
\begin{pmatrix}
	w_{12} & w_{14} \\
	w_{32} & w_{34}
\end{pmatrix}
=
\begin{pmatrix}
	b_{11} & b_{12} \\
	b_{21} & b_{22}
\end{pmatrix}
```

```math
\begin{pmatrix}
	x_{21} & x_{23} & x_{25} \\
	x_{41} & x_{43} & x_{45} \\
	x_{61} & x_{63} & x_{65}
\end{pmatrix}
\ast
\begin{pmatrix}
	w_{21} & w_{23} \\
	w_{41} & w_{43}
\end{pmatrix}
=
\begin{pmatrix}
	c_{11} & c_{12} \\
	c_{21} & c_{22}
\end{pmatrix}
```

```math
\begin{pmatrix}
	x_{22} & x_{24} & x_{26} \\
	x_{42} & x_{44} & x_{46} \\
	x_{62} & x_{64} & x_{66} 
\end{pmatrix}
\ast
\begin{pmatrix}
	w_{22} & w_{24} \\
	w_{42} & w_{44}
\end{pmatrix}
=
\begin{pmatrix}
	d_{11} & d_{12} \\
	d_{21} & d_{22}
\end{pmatrix}
```

这几个结果累加起来

```math
\begin{pmatrix}
	a_{11} & a_{12} \\
	a_{21} & a_{22}
\end{pmatrix}
+
\begin{pmatrix}
	b_{11} & b_{12} \\
	b_{21} & b_{22}
\end{pmatrix}
+
\begin{pmatrix}
	c_{11} & c_{12} \\
	c_{21} & c_{22}
\end{pmatrix}
+
\begin{pmatrix}
	d_{11} & d_{12} \\
	d_{21} & d_{22}
\end{pmatrix}
=
\begin{pmatrix}
	y_{11} & y_{12} \\
	y_{21} & y_{22}
\end{pmatrix}
```

- 优点：和上一个方法刚好优缺点相反
- 缺点：和上一个方法刚好优缺点相反

3. 折叠

抽取 $H$ 或 $W$ 维数据补充到 $C_{in}$ 和 $C_{out}$ 维

举个例子，stride_h = stride_w = 2

```math
\begin{pmatrix}
	x_{11} & x_{12} & x_{13} & x_{14} \\
	x_{21} & x_{22} & x_{23} & x_{24} \\
	x_{31} & x_{32} & x_{33} & x_{34} \\
	x_{41} & x_{42} & x_{43} & x_{44}
\end{pmatrix}
\ast
\begin{pmatrix}
	w_{11} & w_{12} \\
	w_{21} & w_{22}
\end{pmatrix}
=
\begin{pmatrix}
	y_{11} & y_{12} \\
	y_{21} & y_{22}
\end{pmatrix}
```

- 抽取 $H$ 维补充到 $C$ 维

$C_0$

```math
\begin{pmatrix}
	x_{11} & x_{12} & x_{13} & x_{14} \\
	x_{31} & x_{32} & x_{33} & x_{34}
\end{pmatrix}
\ast
\begin{pmatrix}
	w_{11} & w_{12}
\end{pmatrix}
```

$C_1$

```math
\begin{pmatrix}
	x_{21} & x_{22} & x_{23} & x_{24} \\
	x_{41} & x_{42} & x_{43} & x_{44}
\end{pmatrix}
\ast
\begin{pmatrix}
	w_{21} & w_{22}
\end{pmatrix}
```

相当于原来的 $(N,Cin,Hin,Win) * (Cout,Cin,Hk,Wk) = (N,Cout,Hout,Wout)$ 变成了 $(N,Cin \times 2,Hin/2,Win) * (Cout,Cin \times 2,Hk/2,Wk) = (N,Cout,Hout,Wout)$

- 抽取 $W$ 维补充到 $C$ 维

$C_0$

```math
\begin{pmatrix}
	x_{11} & x_{13} \\
	x_{21} & x_{23} \\
	x_{31} & x_{33} \\
	x_{41} & x_{43}
\end{pmatrix}
\ast
\begin{pmatrix}
	w_{11} \\
	w_{21}
\end{pmatrix}
```

$C_1$

```math
\begin{pmatrix}
	x_{12} & x_{14} \\
	x_{22} & x_{24} \\
	x_{32} & x_{34} \\
	x_{42} & x_{44}
\end{pmatrix}
\ast
\begin{pmatrix}
	w_{12} \\
	w_{22}
\end{pmatrix}
```

相当于原来的 $(N,Cin,Hin,Win) * (Cout,Cin,Hk,Wk) = (N,Cout,Hout,Wout)$ 变成了 $(N,Cin \times 2,Hin,Win/2) * (Cout,Cin \times 2,Hk,Wk/2) = (N,Cout,Hout,Wout)$

- 抽取 $H,W$ 维补充到 $C$ 维

$C_0$

```math
\begin{pmatrix}
	x_{11} & x_{13} \\
	x_{31} & x_{33}
\end{pmatrix}
\ast
\begin{pmatrix}
	w_{11}
\end{pmatrix}
```

$C_1$

```math
\begin{pmatrix}
	x_{12} & x_{14} \\
	x_{32} & x_{34}
\end{pmatrix}
\ast
\begin{pmatrix}
	w_{12}
\end{pmatrix}
```

$C_2$

```math
\begin{pmatrix}
	x_{21} & x_{23} \\
	x_{41} & x_{43}
\end{pmatrix}
\ast
\begin{pmatrix}
	w_{21}
\end{pmatrix}
```

$C_3$

```math
\begin{pmatrix}
	x_{22} & x_{24} \\
	x_{42} & x_{44}
\end{pmatrix}
\ast
\begin{pmatrix}
	w_{22}
\end{pmatrix}
```

相当于原来的 $(N,Cin,Hin,Win) * (Cout,Cin,Hk,Wk) = (N,Cout,Hout,Wout)$ 变成了 $(N,Cin \times 4,Hin/2,Win/2) * (Cout,Cin \times 4,Hk/2,Wk/2) = (N,Cout,Hout,Wout)$

- 优点：解决了C_in和C_out比较小的问题
- 缺点：限制p=0，d=1；stride不够大的话重复访存的问题很难解决