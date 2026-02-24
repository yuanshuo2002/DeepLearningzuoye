import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

# 简单模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 2048)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2048, 512)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = SimpleModel().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

scaler = GradScaler()  # 自动处理 loss scaling

for epoch in range(3):
    inputs = torch.randn(32, 512).cuda()
    targets = torch.randn(32, 512).cuda()

    optimizer.zero_grad()

    # autocast 自动选择 FP16/FP32
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    # 使用 scaler 缩放梯度，避免溢出/下溢
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
