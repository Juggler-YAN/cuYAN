# conv

### 常见类型

1. 1D/2D/3D conv
2. $1 \times 1$ conv，计算时可等效于 gemm
3. transpose conv，和 dgrad conv 的计算一致，恢复因为 conv 丢失的信息
4. dilated conv， $dilation > 1$ 的 conv，增大 conv kernel 捕获范围
5. group conv，split C to n group，g 组 $NHWC_i*C_iRSM=NEFM$，即 $NEFgM$，可以减少计算量
6. separable conv
   1. spatial separable convolution，for example，conv kernel 可以分解为 alpha，beta 两个向量的乘积，可以先 * alpha，再 * beta，可以减少计算量
   2. depthwise separable conv，split C to C group，C 组 $NHWC_i*C_iRSM=NEFM$，再经过 $1 \times 1$ conv ，可以减少计算量
7. 可变形 conv
   conv kernel 不固定，适当增大或减小 conv kernel 捕获范围，根据重点来处理信息，减少计算量，能使结果更精准

### 参数

![conv](../img/conv.png)

1. N、H、W、C 图像的数量，长度，宽度和通道
2. C、R、S、M 卷积核的输入通道，长度，宽度和输出通道
3. N、E、F、M 输出的数量，长度，宽度和通道
4. padding 填充
5. stride 步幅
6. dilation 膨胀
   
### 常用的计算公式

1. 卷积输入输出计算公式

$$out = \lfloor \frac{in + 2 * p - k'}{s} \rfloor + 1$$

其中，

$$k' = (d - 1) * (k - 1) + k = d * (k - 1) + 1$$

2. IO量

$$(N * H * W * C)*sizeof(input) + (C * R * S * M)*sizeof(conv) + (N * E * F * M)*sizeof(output)$$

3. 计算量

$$N * E * F * M * C * R * S * 2$$