# dgrad conv 优化方案

1. 转换成 $1 \times 1$ conv

参考 conv 即可

2. dgrad conv 和 wgrad conv 转换成 stride_h * stride_w 组 stride = 1 的 conv

可参考 conv 实现，但因为 dgrad conv 和 wgrad conv 会有很多 0，所以这个方法在实际计算是反而可以更好

举个例子，conv (p=0，s=2，d=1) 计算为

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
	y_{12} &= w_{11}x_{13} + w_{12}x_{14} + w_{21}x_{23} + w_{22}x_{24} + b \\
	y_{21} &= w_{11}x_{31} + w_{12}x_{32} + w_{21}x_{41} + w_{22}x_{42} + b \\
	y_{22} &= w_{11}x_{33} + w_{12}x_{34} + w_{21}x_{43} + w_{22}x_{44} + b
\end{aligned}
\right.
```

根据链式法则得，grad conv为

```math
\left\{
\begin{aligned}
	x_{11} &= w_{11}y_{11} \\
	x_{12} &= w_{12}y_{11} \\
	x_{13} &= w_{11}y_{12} \\
	x_{14} &= w_{12}y_{12} \\
	x_{21} &= w_{21}y_{11} \\
	x_{22} &= w_{22}y_{11} \\
	x_{23} &= w_{21}y_{12} \\
	x_{24} &= w_{22}y_{12} \\
	x_{31} &= w_{11}y_{21} \\
	x_{32} &= w_{12}y_{21} \\
	x_{33} &= w_{11}y_{22} \\
	x_{34} &= w_{12}y_{22} \\
	x_{41} &= w_{21}y_{21} \\
	x_{42} &= w_{22}y_{21} \\
	x_{43} &= w_{21}y_{22} \\
	x_{44} &= w_{22}y_{22}
\end{aligned}
\right.
```

可以拆成 $stride_h*stride_w$ 组

```math
\left\{
\begin{aligned}
	x_{11} &= w_{11}y_{11} \\
	x_{13} &= w_{11}y_{12} \\
	x_{31} &= w_{11}y_{21} \\
	x_{33} &= w_{11}y_{22}
\end{aligned}
\right.
```

```math
\left\{
\begin{aligned}
	x_{12} &= w_{12}y_{11} \\
	x_{14} &= w_{12}y_{12} \\
	x_{32} &= w_{12}y_{21} \\
	x_{34} &= w_{12}y_{22}
\end{aligned}
\right.
```

```math
\left\{
\begin{aligned}
	x_{21} &= w_{21}y_{11} \\
	x_{23} &= w_{21}y_{12} \\
	x_{41} &= w_{21}y_{21} \\
	x_{43} &= w_{21}y_{22}
\end{aligned}
\right.
```

```math
\left\{
\begin{aligned}
	x_{22} &= w_{22}y_{11} \\
	x_{24} &= w_{22}y_{12} \\
	x_{42} &= w_{22}y_{21} \\
	x_{44} &= w_{22}y_{22}
\end{aligned}
\right.
```

即

```math
\begin{align}
dgrad conv
&=
\begin{pmatrix}
        0 & 0 & 0 & 0 & 0 \\
	0 & y_{11} & 0 & y_{12} & 0 \\
        0 & 0 & 0 & 0 & 0 \\
	0 & y_{21} & 0 & y_{22} & 0 \\
        0 & 0 & 0 & 0 & 0
\end{pmatrix}
\ast
\begin{pmatrix}
	w_{22} & w_{21} \\
	w_{12} & w_{11}
\end{pmatrix} \notag \\
&=
\begin{pmatrix}
	y_{11} & y_{12} \\
	y_{21} & y_{22}
\end{pmatrix}
(w_{22} + w_{21} + w_{12} + w_{11}) \notag \\
&= 
\begin{pmatrix}
	11 & 12 & 11 & 12 \\
	21 & 22 & 21 & 22 \\
	11 & 12 & 11 & 12 \\
	21 & 22 & 21 & 22
\end{pmatrix} \notag \\
&=
\begin{pmatrix}
	x_{11} & x_{12} & x_{13} & x_{14} \\
	x_{21} & x_{22} & x_{23} & x_{24} \\
	x_{31} & x_{32} & x_{33} & x_{34} \\
	x_{41} & x_{42} & x_{43} & x_{44}
\end{pmatrix} \notag
\end{align}
```

convBF 同理

```math
\begin{align}
wgrad conv
&=
\begin{pmatrix}
	x_{11} & x_{12} & x_{13} & x_{14} \\
	x_{21} & x_{22} & x_{23} & x_{24} \\
	x_{31} & x_{32} & x_{33} & x_{34} \\
	x_{41} & x_{42} & x_{43} & x_{44}
\end{pmatrix}
\ast
\begin{pmatrix}
	y_{11} & 0 & y_{12} \\
	0 & 0 & 0 \\
	y_{21} & 0 & y_{22}
\end{pmatrix} & \notag \\
&=
\begin{pmatrix}
	x_{11} & x_{12} & x_{13} & x_{14} \\
	x_{21} & x_{22} & x_{23} & x_{24} \\
	x_{31} & x_{32} & x_{33} & x_{34} \\
	x_{41} & x_{42} & x_{43} & x_{44}
\end{pmatrix}
(
\begin{pmatrix}
	y_{11} & y_{12} \\
	y_{21} & y_{22}
\end{pmatrix}
+
\begin{pmatrix}
	0 \\
	0
\end{pmatrix}
+
\begin{pmatrix}
	0 & 0
\end{pmatrix}
+
0
) \notag \\
&=
\begin{pmatrix}
	w_{11} & w_{12} \\
	w_{21} & w_{22}
\end{pmatrix}  & \notag
\end{align}
```

可以看到，这样计算的好处是输入基本上不变，只需要对卷积核进行处理（不用再考虑填充0之类的操作），大大减小了计算量
