import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# 1. 随机种子
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# =========================
# 2. 生成三类点云
# =========================
def sample_sphere(n_points=128):
    pts = []
    for _ in range(n_points):
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        pts.append([x, y, z])
    return np.array(pts, dtype=np.float32)

def sample_cube(n_points=128):
    pts = np.random.uniform(-1, 1, size=(n_points, 3)).astype(np.float32)
    return pts

def sample_cylinder(n_points=128):
    pts = []
    for _ in range(n_points):
        theta = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, 1)
        z = np.random.uniform(-1, 1)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        pts.append([x, y, z])
    return np.array(pts, dtype=np.float32)

def normalize_points(points):
    centroid = points.mean(axis=0, keepdims=True)
    points = points - centroid
    scale = np.max(np.linalg.norm(points, axis=1))
    points = points / (scale + 1e-8)
    return points.astype(np.float32)

def jitter_points(points, sigma=0.01, clip=0.03):
    noise = np.clip(sigma * np.random.randn(*points.shape), -clip, clip)
    return (points + noise).astype(np.float32)

# =========================
# 3. 数据集
# =========================
class ToyPointCloudDataset(Dataset):
    def __init__(self, split="train", samples_per_class=200, n_points=128):
        self.data = []
        self.labels = []
        self.n_points = n_points

        shape_fns = [
            (sample_sphere, 0),
            (sample_cube, 1),
            (sample_cylinder, 2),
        ]

        if split == "train":
            count = samples_per_class
        else:
            count = samples_per_class // 4

        for fn, label in shape_fns:
            for _ in range(count):
                pts = fn(n_points)
                pts = normalize_points(pts)
                if split == "train":
                    pts = jitter_points(pts)
                self.data.append(pts)
                self.labels.append(label)

        self.data = np.stack(self.data)   # [N, P, 3]
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pts = self.data[idx]
        label = self.labels[idx]
        return torch.from_numpy(pts), torch.tensor(label)

# =========================
# 4. 小型 PointNet
# =========================
class SmallPointNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: [B, P, 3]
        feat = self.mlp(x)             # [B, P, 256]
        global_feat, _ = torch.max(feat, dim=1)  # [B, 256]
        logits = self.cls_head(global_feat)
        return logits

# =========================
# 5. 训练与评估
# =========================
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for pts, labels in loader:
            pts = pts.to(device)
            labels = labels.to(device)
            logits = model(pts)
            pred = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_set = ToyPointCloudDataset(split="train", samples_per_class=200, n_points=128)
    test_set = ToyPointCloudDataset(split="test", samples_per_class=200, n_points=128)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)

    model = SmallPointNet(num_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 5
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for pts, labels in train_loader:
            pts = pts.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(pts)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Test Acc: {acc:.4f}")

    torch.save(model.state_dict(), "toy_pointcloud_model.pth")
    print("Training done. Model saved to toy_pointcloud_model.pth")

if __name__ == "__main__":
    main()