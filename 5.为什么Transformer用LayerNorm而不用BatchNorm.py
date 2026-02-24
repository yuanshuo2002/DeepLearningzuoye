import torch
import torch.nn as nn

# 一个简单的序列模型，分别用 BatchNorm 和 LayerNorm
class SeqModelBN(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.fc = nn.Linear(10, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)  # BatchNorm
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [batch, seq_len, features]
        x = self.fc(x)
        # BatchNorm 需要把 batch 和 seq_len 展平到 batch 维度
        x = self.bn(x.view(-1, x.size(-1))).view(x.size())
        return self.out(self.relu(x))

class SeqModelLN(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.fc = nn.Linear(10, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)  # LayerNorm
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc(x)
        x = self.ln(x)  # LayerNorm 按特征维度归一化
        return self.out(self.relu(x))

# 模拟输入
batch_small = torch.randn(2, 5, 10)   # batch=2, seq_len=5
batch_large = torch.randn(32, 5, 10)  # batch=32, seq_len=5

model_bn = SeqModelBN()
model_ln = SeqModelLN()

print("=== 小 batch (2) ===")
print("BatchNorm 输出:", model_bn(batch_small)[0].detach())
print("LayerNorm 输出:", model_ln(batch_small)[0].detach())

print("\n=== 大 batch (32) ===")
print("BatchNorm 输出:", model_bn(batch_large)[0].detach())
print("LayerNorm 输出:", model_ln(batch_large)[0].detach())
