### conv优化思路

### conv

- [ ] conv
  - [ ] 折叠优化 C,M,C&&M

C 折叠 抽 H,W 维补充 C 维
M 折叠
C,M 折叠

convbf
   1. 其他转换成-> 1\*1convbf
   2. 思路一：dx 转置后和 y 做 conv： CNHW\*NEFM = CM
   3. 思路二：y 转置后和 dx 做 conv，然后再把结果转置：MNEF\*NHWC=MC->CM
   4. 切分 C,M 到 32 核/切分 NHW/NEF 到 32 个从核