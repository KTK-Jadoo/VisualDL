import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5)      # 1x28x28 -> 8x24x24
        self.pool  = nn.MaxPool2d(2, 2)                  # 8x24x24 -> 8x12x12
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)     # 8x12x12 -> 16x10x10
        self.relu  = nn.ReLU(inplace=True)
        self.fc    = nn.Linear(16*10*10, 10)

    def forward(self, x):
        h1 = self.relu(self.conv1(x))        # 8x24x24
        p1 = self.pool(h1)                   # 8x12x12
        h2 = self.relu(self.conv2(p1))       # 16x10x10
        logits = self.fc(h2.view(x.size(0), -1))
        return logits, h1, h2                # expose both conv blocks

# --- training (same as before; 1 quick epoch is enough for demo) ---
transform = T.Compose([T.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

model = TinyCNN()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(1):
    for imgs, labels in trainloader:
        optimizer.zero_grad()
        out, _, _ = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: loss={loss.item():.3f}")

# --- ONNX export (note the extra output 'conv2') ---
dummy = torch.randn(1, 1, 28, 28)
torch.onnx.export(
    model, dummy, "public/model.onnx",
    input_names=["input"],
    output_names=["probs", "conv1", "conv2"],
    dynamic_axes={"input": {0:"N"}, "probs": {0:"N"}, "conv1": {0:"N"}, "conv2": {0:"N"}},
    opset_version=13
)
print("Exported to public/model.onnx")
