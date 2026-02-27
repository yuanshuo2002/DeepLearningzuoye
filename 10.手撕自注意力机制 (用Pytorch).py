# ===============================
# 1. 导入库
# ===============================
import torch
import torch.nn as nn
from math import sqrt

# 为了结果可复现
torch.manual_seed(42)


# ===============================
# 2. 定义 SelfAttention
# ===============================
class SelfAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v):
        super().__init__()

        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v

        # 线性映射 Q K V
        self.linear_q = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_q, dim_v, bias=False)

        # 缩放因子
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, x):
        batch, n, dim_q = x.shape
        assert dim_q == self.dim_q

        # 得到 Q K V
        q = self.linear_q(x)  # (batch, n, dim_k)
        k = self.linear_k(x)  # (batch, n, dim_k)
        v = self.linear_v(x)  # (batch, n, dim_v)

        # 计算注意力分数
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # (batch, n, n)

        # softmax
        dist = torch.softmax(dist, dim=-1)

        # 加权 V
        att = torch.bmm(dist, v)  # (batch, n, dim_v)

        return att


# ===============================
# 3. 构造测试输入
# ===============================
batch_size = 2
seq_len = 5
dim_q = 8
dim_k = 8
dim_v = 6

# 随机输入
x = torch.randn(batch_size, seq_len, dim_q)

print("输入形状:", x.shape)


# ===============================
# 4. 创建模型并前向传播
# ===============================
model = SelfAttention(dim_q, dim_k, dim_v)

output = model(x)

print("输出形状:", output.shape)


# ===============================
# 5. 查看部分输出
# ===============================
print("\n输出示例:")
print(output)
