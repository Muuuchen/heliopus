import torch

x =torch.randn(2,2,3)
print("orgin x ",x)
print(x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5))
print(torch.nn.functional.rms_norm(x,[3]))
