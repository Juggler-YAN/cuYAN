
### TODO

1. 定义
2. cannon 算法

矩阵 $A$， $B$， $C$ 大小分别为 $(M,N)$， $(N,K)$， $(M,N)$ ，假设有 $P$ 个进程，可以分割成 $(\frac{M}{\sqrt{p}},\frac{K}{\sqrt{p}})$， $(\frac{K}{\sqrt{p}},\frac{N}{\sqrt{p}})$， $(\frac{M}{\sqrt{p}},\frac{N}{\sqrt{p}})$ 大小的子矩阵，每个进程对应一个子矩阵

以 $P = 16$ 为例，

```math
\begin{pmatrix}
	A_{00} & A_{01} & A_{02} & A_{03} \\
	A_{10} & A_{11} & A_{12} & A_{13} \\
	A_{20} & A_{21} & A_{22} & A_{23} \\
	A_{30} & A_{31} & A_{32} & A_{33}
\end{pmatrix}
```

```math
\begin{pmatrix}
	B_{00} & B_{01} & B_{02} & B_{03} \\
	B_{10} & B_{11} & B_{12} & B_{13} \\
	B_{20} & B_{21} & B_{22} & B_{23} \\
	B_{30} & B_{31} & B_{32} & B_{33}
\end{pmatrix}
```

将 $A_{ij}$ 循环左移 $i$ 位， $B_{ij}$ 循环上移 $j$ 位，

```math
\begin{pmatrix}
	A_{00} & A_{01} & A_{02} & A_{03} \\
	A_{11} & A_{12} & A_{13} & A_{10} \\
	A_{22} & A_{23} & A_{20} & A_{21} \\
	A_{33} & A_{30} & A_{31} & A_{32}
\end{pmatrix}
```

```math
\begin{pmatrix}
	B_{00} & B_{11} & B_{22} & B_{33} \\
	B_{10} & B_{21} & B_{32} & B_{03} \\
	B_{20} & B_{31} & B_{02} & B_{13} \\
	B_{30} & B_{01} & B_{12} & B_{23}
\end{pmatrix}
```

将每个进程上的子矩阵相乘并累加，有

```math
\begin{pmatrix}
	A_{00}B_{00} & A_{01}B_{11} & A_{02}B_{22} & A_{03}B_{33} \\
	A_{11}B_{10} & A_{12}B_{21} & A_{13}B_{32} & A_{10}B_{03} \\
	A_{22}B_{20} & A_{23}B_{31} & A_{20}B_{02} & A_{21}B_{13} \\
	A_{33}B_{30} & A_{30}B_{01} & A_{31}B_{12} & A_{32}B_{23}
\end{pmatrix}
```

将 $A_{ij}$ 循环左移 $1$ 位， $B_{ij}$ 循环上移 $1$ 位，然后将每个进程上的子矩阵相乘并累加，重复这一操作 $\sqrt{p}-1$ 次，可得

```math
\begin{pmatrix}
	A_{00}B_{00} + A_{01}B_{10} + A_{02}B_{20} + A_{03}B_{30} &
    A_{01}B_{11} + A_{02}B_{21} + A_{03}B_{31} + A_{00}B_{01} &
    A_{02}B_{22} + A_{03}B_{32} + A_{00}B_{02} + A_{01}B_{12} &
    A_{03}B_{33} + A_{00}B_{03} + A_{01}B_{13} + A_{02}B_{23} \\ 
	A_{11}B_{10} + A_{12}B_{20} + A_{13}B_{30} + A_{10}B_{00} &
    A_{12}B_{21} + A_{13}B_{31} + A_{10}B_{01} + A_{11}B_{11} &
    A_{13}B_{32} + A_{10}B_{02} + A_{11}B_{12} + A_{12}B_{22} &
    A_{10}B_{03} + A_{11}B_{13} + A_{12}B_{23} + A_{13}B_{33} \\ 
	A_{22}B_{20} + A_{23}B_{30} + A_{20}B_{00} + A_{21}B_{10} &
    A_{23}B_{31} + A_{20}B_{01} + A_{21}B_{11} + A_{22}B_{21} &
    A_{20}B_{02} + A_{21}B_{12} + A_{22}B_{22} + A_{23}B_{32} &
    A_{21}B_{13} + A_{22}B_{23} + A_{23}B_{33} + A_{20}B_{03} \\ 
	A_{33}B_{30} + A_{30}B_{00} + A_{31}B_{10} + A_{32}B_{20} &
    A_{30}B_{01} + A_{31}B_{11} + A_{32}B_{21} + A_{33}B_{31} &
    A_{31}B_{12} + A_{32}B_{22} + A_{33}B_{32} + A_{30}B_{02} &
    A_{32}B_{23} + A_{33}B_{33} + A_{30}B_{03} + A_{31}B_{13}
\end{pmatrix}
```

即

```math
\begin{pmatrix}
	C_{00} & C_{01} & C_{02} & C_{03} \\
	C_{11} & C_{12} & C_{13} & C_{10} \\
	C_{22} & C_{23} & C_{20} & C_{21} \\
	C_{33} & C_{30} & C_{31} & C_{32}
\end{pmatrix}
```

3. fox 算法
4. summa 算法

矩阵 $A$， $B$ 大小分别为 $(M,N)$， $(N,K)$，假设有 $P_1 \times P_2$ 个进程，可以分割成 $(\frac{M}{p_1},K)$， $(K,\frac{N}{p_2})$ 大小的子矩阵，将第 $i$ 个子矩阵 $A_i$ 广播到第 $i$ 行，将第 $j$ 个子矩阵 $B_j$ 广播到第 $j$ 列，每一个进程对应的子矩阵相乘可得 $C_{ij}$

以 $P_1 = 2$， $P_2 = 3$ 为例，

```math
\begin{pmatrix}
	A_{0} \\
	A_{1}
\end{pmatrix}
```

广播后变成

```math
\begin{pmatrix}
	A_{0} & A_{0} & A_{0} \\
	A_{1} & A_{1} & A_{1}
\end{pmatrix}
```

```math
\begin{pmatrix}
	B_{0} & B_{1} & B_{2}
\end{pmatrix}
```

广播后变成

```math
\begin{pmatrix}
	B_{0} & B_{1} & B_{2} \\
	B_{0} & B_{1} & B_{2}
\end{pmatrix}
```

将每个进程所拥有的子矩阵相乘，可得

```math
\begin{pmatrix}
	A_{0}B_{0} & A_{0}B_{1} & A_{0}B_{2} \\
	A_{1}B_{0} & A_{1}B_{1} & A_{1}B_{2}
\end{pmatrix}
```

即

```math
\begin{pmatrix}
	C_{00} & C_{01} & C_{02} \\
	C_{10} & C_{11} & C_{12}
\end{pmatrix}
```


5. strassen 算法