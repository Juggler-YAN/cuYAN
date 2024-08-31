# wgrad conv 优化方案

1. 转换成 $1 \times 1$ conv

参考 conv 即可

2. 思路一：$x * dy = w$
   思路二：$dy^T * x^T = w^T$

3. - [ ] 切分 C,M 到 32 核/切分 NHW/NEF 到 32 个从核
