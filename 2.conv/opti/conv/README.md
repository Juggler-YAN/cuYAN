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

2. 转换成 stride_h * stride_w 组 group conv

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

是不是可以刚好拆成 4 组

```math
\begin{pmatrix}
	x_{11} & x_{13} \\
	x_{31} & x_{33}
\end{pmatrix}
\ast
\begin{pmatrix}
	w_{11} & w_{12} \\
	w_{21} & w_{22}
\end{pmatrix}
=
\begin{pmatrix}
	y_{11}
\end{pmatrix}
```

```math
\begin{pmatrix}
	x_{12} & x_{14} \\
	x_{32} & x_{34}
\end{pmatrix}
\ast
\begin{pmatrix}
	w_{11} & w_{12} \\
	w_{21} & w_{22}
\end{pmatrix}
=
\begin{pmatrix}
	y_{12}
\end{pmatrix}
```

```math
\begin{pmatrix}
	x_{21} & x_{23} \\
	x_{41} & x_{43}
\end{pmatrix}
\ast
\begin{pmatrix}
	w_{11} & w_{12} \\
	w_{21} & w_{22}
\end{pmatrix}
=
\begin{pmatrix}
	y_{21}
\end{pmatrix}
```

```math
\begin{pmatrix}
	x_{22} & x_{24} \\
	x_{42} & x_{44}
\end{pmatrix}
\ast
\begin{pmatrix}
	w_{11} & w_{12} \\
	w_{21} & w_{22}
\end{pmatrix}
=
\begin{pmatrix}
	y_{22}
\end{pmatrix}
```



3. 折叠

抽取某其他维数据补充到 $C_{in}$ or $C_{out}$ 维

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

是不是可以把 W 维切断然后补充到 C 维上呢

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
=
\begin{pmatrix}
	y_{11} \\
	y_{21}
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
=
\begin{pmatrix}
	y_{12} \\
	y_{22}
\end{pmatrix}
```


- [ ] 折叠优化
    - [ ] $C_{in}$ 折叠 抽 H,W 维补充 $C_{in}$ 维
    - [ ] $C_{out}$ 折叠
    - [ ] $C_{in}$, $C_{out}$ 折叠

