import torch

def rope(x, position_ids, dim=64):
    """
    x: [batch, seq_len, dim]
    position_ids: [seq_len]
    """
    half_dim = dim // 2
    freq = torch.exp(-torch.arange(0, half_dim, dtype=torch.float32) / half_dim * torch.log(torch.tensor(10000.0)))
    angles = position_ids[:, None] * freq[None, :]
    sin, cos = torch.sin(angles), torch.cos(angles)

    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    x_rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rotated

# 示例：对一个序列做 RoPE
batch, seq_len, dim = 1, 5, 64
x = torch.randn(batch, seq_len, dim)
position_ids = torch.arange(seq_len)
x_rope = rope(x, position_ids)
print(x_rope.shape)  # [1, 5, 64]
