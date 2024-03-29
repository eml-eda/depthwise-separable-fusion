# Depthwise separable convolution fusion

Kernels for GAP8 implementing the layer fusion of depthwise convolution and pointwise convolutions.
If you intend to use or reference this library for an academic publication, please consider citing it:
```
TBD
```

## Library overview
The source file can be found under src/, divided in three different directories.

### Convolution
Kernels of base depthwise and pointwise with different layouts (CHW/HWC) than the ones available in pulp-nn.
### Fused
Fused depthwise separable kernels.

### Matmul
Base matmul kernel with CHW layout.

