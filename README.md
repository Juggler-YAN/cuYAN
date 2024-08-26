# cuYAN

记录关于自己对算子优化的研究

### Contents

以下实现均与cublas和cudnn对比过结果无误

0. Introduce：总结算子优化的通用思路
1. gemm
    - calc

      实现了定义、cannon 算法、summa 算法计算 gemm

2. conv
    - calc

      实现了滑窗法、img2col 算法、FFT 算法、winograd 算法计算 conv

    - type

      实现了1D/2D/3D conv、1*1 conv、转置conv、空洞conv、分组conv、可分离conv（空间可分离和深度可分离）等多种conv类型

    - backward2forward

      实现了bgrad conv、dgrad conv 和 wgrad conv 转换成conv进行计算

    - 常用的优化思路

      conv转换成1*1 conv，conv转换成group conv，conv抽取H或W维补充到C_in或Cout维

3. fft


### TODO

1. gemm

- [ ] fox 算法
- [ ] strassen算法

2. conv

- [ ] 支持 NHWC 和 NCHW 两种格式
- [ ] winograd：F(2,3) -> F(n,3)
- [ ] 代码：conv转换成1*1 conv
- [ ] 代码：conv转换成group conv
- [ ] 代码：conv抽取H或W维补充到C_in或Cout维

3. fft

- [ ] 几种算法的实现
  - [ ] 现在用C++实现的 和 matlab python 实现的 fft 顺序不一样，原因待分析
