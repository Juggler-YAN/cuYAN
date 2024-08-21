# wgrad conv 优化方案

- [ ] 转换成 $1 \times 1$ conv
- [ ] 思路一：dx 转置后和 y 做 conv： CNHW\*NEFM = CM；思路二：y 转置后和 dx 做 conv，然后再把结果转置：MNEF\*NHWC=MC->CM
- [ ] 切分 C,M 到 32 核/切分 NHW/NEF 到 32 个从核
