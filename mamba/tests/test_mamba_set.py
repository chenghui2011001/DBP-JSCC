from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
import torch

# 定义张量维度
B, C, L = 1, 4, 16

# 构造输入张量（CUDA设备）
u = torch.randn(B, C, L, device='cuda')
delta = torch.ones(B, C, L, device='cuda')
A = torch.zeros(C, 1, device='cuda')
Bp = torch.ones(C, 1, device='cuda')
Cp = torch.ones(C, 1, device='cuda')
Dp = torch.zeros(C, device='cuda')

# 调用selective_scan_fn并输出结果形状
y = selective_scan_fn(
    u, delta, A, Bp, Cp, Dp,
    delta_bias=None,
    delta_softplus=False,
    nrows=1
)
print(y.shape)

