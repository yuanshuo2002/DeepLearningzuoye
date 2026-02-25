import torch
import torch.nn as nn
import torch.nn.functional as F

# 简化版的注意力层
class SimpleAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

    def forward(self, x, kv_cache=None):
        # x: [batch, seq_len, d_model]
        Q = self.Wq(x)

        # 如果有缓存，直接用缓存的 K/V + 新的 K/V
        K_new = self.Wk(x)
        V_new = self.Wv(x)

        if kv_cache is None:
            K = K_new
            V = V_new
        else:
            K = torch.cat([kv_cache["K"], K_new], dim=1)
            V = torch.cat([kv_cache["V"], V_new], dim=1)

        # 注意力计算
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        # 返回输出和新的缓存
        return output, {"K": K, "V": V}

# 模拟解码过程
d_model = 16
attention = SimpleAttention(d_model)

kv_cache = None
batch_size = 1

# 模拟逐步生成序列
for step in range(5):
    # 每次输入一个新 token 的 embedding
    x = torch.randn(batch_size, 1, d_model)

    out, kv_cache = attention(x, kv_cache)
    print(f"Step {step+1}, Output shape: {out.shape}, Cache K shape: {kv_cache['K'].shape}")
