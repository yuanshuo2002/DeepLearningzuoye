import torch
import torch.nn as nn

# 定义 Transformer 前馈网络
class TransformerFFN(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, activation="relu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError("Unsupported activation")
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))

# 测试
x = torch.randn(2, 10, 512)  # batch=2, seq_len=10, d_model=512
ffn = TransformerFFN(activation="gelu")
output = ffn(x)
print(output.shape)  # (2, 10, 512)
