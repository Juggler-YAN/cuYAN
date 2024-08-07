### 正反向conv的推导

举个例子，convF 计算为

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

即

```math
\left\{
\begin{aligned}
	y_{11} &= w_{11}x_{11} + w_{12}x_{12} + w_{21}x_{21} + w_{22}x_{22} + b \\
	y_{12} &= w_{11}x_{12} + w_{12}x_{13} + w_{21}x_{22} + w_{22}x_{23} + b \\
	y_{21} &= w_{11}x_{21} + w_{12}x_{22} + w_{21}x_{31} + w_{22}x_{32} + b \\
	y_{22} &= w_{11}x_{22} + w_{12}x_{23} + w_{21}x_{32} + w_{22}x_{33} + b
\end{aligned}
\right.
```

根据链式求导法则，可得

1. bgrad conv


```math
\partial{b} = \partial{y_{11}} + \partial{y_{12}} + \partial{y_{21}} + \partial{y_{22}}
```

即

```math
\partial{b} = \sum_{u,v}\partial{y}
```

2. dgrad conv

```math
\left\{
\begin{aligned}
	\partial{x_{11}} &= \partial{y_{11}}w_{11} \\
	\partial{x_{12}} &= \partial{y_{11}}w_{12} + \partial{y_{12}}w_{11} \\
	\partial{x_{13}} &= \partial{y_{12}}w_{12} \\
	\partial{x_{21}} &= \partial{y_{11}}w_{21} + \partial{y_{21}}w_{11} \\
	\partial{x_{22}} &= \partial{y_{11}}w_{22} + \partial{y_{12}}w_{21} + \partial{y_{21}}w_{12} + 		\partial{y_{22}}w_{11} \\
	\partial{x_{23}} &= \partial{y_{12}}w_{22} + \partial{y_{22}}w_{12} \\
	\partial{x_{31}} &= \partial{y_{21}}w_{21} \\
	\partial{x_{32}} &= \partial{y_{21}}w_{22} + \partial{y_{22}}w_{21} \\
	\partial{x_{33}} &= \partial{y_{22}}w_{22}
\end{aligned}
\right.
```

即

```math
\partial{X}
= 
\begin{pmatrix}
	\partial{x_{11}} & \partial{x_{12}} & \partial{x_{13}} \\
	\partial{x_{21}} & \partial{x_{22}} & \partial{x_{23}} \\
	\partial{x_{31}} & \partial{x_{32}} & \partial{x_{33}}
\end{pmatrix}
=
\begin{pmatrix}
	0 & 0 & 0 & 0 \\
	0 & \partial{y_{11}} & \partial{y_{12}} & 0 \\
	0 & \partial{y_{21}} & \partial{y_{22}} & 0
	0 & 0 & 0 & 0 \\
\end{pmatrix}
\ast
\begin{pmatrix}
	w_{22} & w_{21} \\
	w_{12} & w_{11}
\end{pmatrix}
```

与之类似，分析可得，dgrad conv 转换为 conv 需要：

对于 $\partial{Y}$，
- 填充 $p$ 即填充 $k-p-1$行或列，注意指的是卷积核膨胀后的 $k=(k-1)*(d-1)+k=(k-1)*d+1$
- 跨步 $s$ 即在行或列之间插入 $s-1$ 行或列

对于 $W$，
- 需要旋转 $180^{\circ}$
- 膨胀 $d$ 即行或列之间插入 $d-1$ 行或列

对于 $\partial{X}$，
- 需要再填充 $in-((out-1)*s+((k-1)(d-1)+k)-2*p)$

可以简单记为

$$\partial{X} = padding(\partial{Y}) \ast rot(W)$$

3. wgrad conv

```math
\left\{
\begin{aligned}
	\partial{w_{11}} &= \partial{y_{11}}x_{11} + \partial{y_{12}}x_{12}  + \partial{y_{21}}x_{21}  + \partial{y_{22}}x_{22} \\
	\partial{w_{12}} &= \partial{y_{11}}x_{12} + \partial{y_{12}}x_{13}  + \partial{y_{21}}x_{22}  + \partial{y_{22}}x_{23} \\
	\partial{w_{21}} &= \partial{y_{11}}x_{21} + \partial{y_{12}}x_{22}  + \partial{y_{21}}x_{31}  + \partial{y_{22}}x_{32} \\
	\partial{w_{22}} &= \partial{y_{11}}x_{22} + \partial{y_{12}}x_{23}  + \partial{y_{21}}x_{32}  + \partial{y_{22}}x_{33}
\end{aligned}
\right.
```

即

```math
\begin{pmatrix}
	x_{11} & x_{12} & x_{13} \\
	x_{21} & x_{22} & x_{23} \\
	x_{31} & x_{32} & x_{33}
\end{pmatrix}
\ast
\begin{pmatrix}
	\partial{y_{11}} & \partial{y_{12}} \\
	\partial{y_{21}} & \partial{y_{22}}
\end{pmatrix}
=
\begin{pmatrix}
	\partial{w_{11}} & \partial{w_{12}} \\
	\partial{w_{21}} & \partial{w_{22}}
\end{pmatrix}
```

与之类似，分析可得，wgrad conv 转换为 conv 需要：

对于 $X$，

- 填充 $p$ 即填充 $p$ 行或列
- 膨胀 $d$ 即跨步为 $s=d$

对于 $\partial{Y}$，
- 跨步 $s$ 即膨胀为 $d=s$

即

$$\partial{W} = X \ast \partial{Y} $$

### TODO

- [x] bgrad conv
- [x] dgrad conv
- [x] wgrad conv
- [ ] dgrad conv -> group conv
- [ ] wgrad conv -> group conv