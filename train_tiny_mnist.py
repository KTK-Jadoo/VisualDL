import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T

# --- Model ---
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5)   # 1x28x28 -> 8x24x24
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(8*24*24, 10)

    def forward(self, x):
        h1 = self.relu(self.conv1(x))          # name: conv1
        logits = self.fc(h1.view(x.size(0), -1))
        return logits, h1

# --- Training ---
transform = T.Compose([T.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

model = TinyCNN()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(1):  # just 1 epoch for quick prototype
    for imgs, labels in trainloader:
        optimizer.zero_grad()
        out, _ = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: loss={loss.item():.3f}")

# --- Export to ONNX ---
dummy = torch.randn(1, 1, 28, 28)
torch.onnx.export(
    model, dummy, "public/model.onnx",
    input_names=["input"], output_names=["probs", "conv1"],
    dynamic_axes={"input": {0: "N"}, "probs": {0: "N"}, "conv1": {0: "N"}},
    opset_version=13
)
print("Exported to public/model.onnx")
