import torch
import time

# 定义一个简单的矩阵乘法函数，测试不同精度的速度和结果
def test_precision(dtype, name):
    a = torch.randn(2000, 2000, dtype=dtype)
    b = torch.randn(2000, 2000, dtype=dtype)

    start = time.time()
    c = torch.matmul(a, b)  # 矩阵乘法
    end = time.time()

    print(f"{name}: dtype={dtype}, time={end-start:.4f}s, result mean={c.mean().item():.6f}")

# 测试三种精度
test_precision(torch.float32, "FP32")
test_precision(torch.float16, "FP16")
test_precision(torch.bfloat16, "BF16")
