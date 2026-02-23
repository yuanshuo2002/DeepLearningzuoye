import torch
import torch.nn as nn
import torch.optim as optim

# 简单模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 2048)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2048, 512)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# 初始化模型、优化器和损失函数
model = SimpleModel()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# 模拟训练循环
for epoch in range(3):
    # 输入用 FP16，模拟低精度
    inputs = torch.randn(32, 512).half()
    targets = torch.randn(32, 512).float()  # 标签保持 FP32

    # 前向传播：模型参数是 FP32，但输入是 FP16
    outputs = model(inputs.float())  # 转换为 FP32 保证稳定性
    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
