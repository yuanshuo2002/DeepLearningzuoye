import torch
from torch import nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, w1, w2, w3) -> None:
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def forward(self, x):
        x1 = F.linear(x, self.w1.weight)
        x2 = F.linear(x, self.w2.weight)
        hidden = F.silu(x1) * x2
        return F.linear(hidden, self.w3.weight)


# 定义输入维度和隐藏维度
input_dim, hidden_dim, output_dim = 8, 16, 4

# 创建线性层
w1 = nn.Linear(input_dim, hidden_dim)
w2 = nn.Linear(input_dim, hidden_dim)
w3 = nn.Linear(hidden_dim, output_dim)

# 创建 SwiGLU 模块
model = SwiGLU(w1, w2, w3)

# 测试输入
x = torch.randn(2, input_dim)
y = model(x)

print("输入形状:", x.shape)
print("输出形状:", y.shape)
