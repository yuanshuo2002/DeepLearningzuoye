import torch
import torch.nn as nn

# 定义 Transformer 前馈网络，使用 ReLU 激活函数
class TransformerFFN(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)   # 扩展维度
        self.relu = nn.ReLU()                     # ReLU 激活函数
        self.linear2 = nn.Linear(d_ff, d_model)   # 映射回原始维度

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

# 测试代码
x = torch.randn(2, 10, 512)  # batch=2, seq_len=10, d_model=512
ffn = TransformerFFN()
output = ffn(x)
print(output.shape)  # 结果应为 (2, 10, 512)
