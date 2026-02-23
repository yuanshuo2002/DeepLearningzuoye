import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

# 一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 2048)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2048, 512)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# 初始化模型、优化器和损失函数
model = SimpleModel().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# 混合精度工具
scaler = GradScaler()

# 模拟训练循环
for epoch in range(3):
    inputs = torch.randn(32, 512).cuda()
    targets = torch.randn(32, 512).cuda()

    optimizer.zero_grad()

    # autocast: 在这里自动使用 FP16 进行前向和反向传播
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    # 使用 scaler 缩放梯度，避免 FP16 下梯度过小
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
