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

"""
归一化。
网络会被物体在空间中的平移、物体整体尺寸大小的无关因素干扰。归一化后，网络更关注：这个点云的“形状结构”是什么
"""
def normalize_points(points):
    centroid = points.mean(axis=0, keepdims=True)
    points = points - centroid
    scale = np.max(np.linalg.norm(points, axis=1))
    points = points / (scale + 1e-8)
    return points.astype(np.float32)

"""
加扰动————简单的数据增强。提高泛化能力，防止过拟合（训练样本学的太好了，导致泛化能力下降）
让模型不要死记硬背训练数据，而是学更稳健的几何特征
"""
def jitter_points(points, sigma=0.01, clip=0.03):
    noise = np.clip(sigma * np.random.randn(*points.shape), -clip, clip)
    return (points + noise).astype(np.float32)

# =========================
# 3. 数据集————整个训练流程的数据入口
# =========================
class ToyPointCloudDataset(Dataset):
    def __init__(self, split="train", samples_per_class=200, n_points=128):
        self.data = []
        self.labels = []
        self.n_points = n_points

        shape_fns = [           #定义类别
            (sample_sphere, 0),
            (sample_cube, 1),
            (sample_cylinder, 2),
        ]

        if split == "train":
            count = samples_per_class       #训练集200
        else:
            count = samples_per_class // 4      # //是整除运算符，测试集50

        for fn, label in shape_fns:
            for _ in range(count):
                pts = fn(n_points)      #等价于sample_sphere(256)；sample_cube(256)；sample_cylinder(256)。即 生成一个形状=用n_points个点描述
                pts = normalize_points(pts)
                if split == "train":        #只给训练集加噪声，测试集不加。 训练时增强数据，测试时保持干净
                    pts = jitter_points(pts)
                self.data.append(pts)
                self.labels.append(label)

        self.data = np.stack(self.data)   # [N, P, 3]
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):      #告诉pytorch:这个数据集总共有多少个样本
        return len(self.labels)

    def __getitem__(self, idx):     # 当索引第 idx 个样本时，返回对应的点云和标签。这里把 numpy 数据转成 PyTorch tensor，返回的是pts:[128,3],label:一个整数类别
        pts = self.data[idx]
        label = self.labels[idx]
        return torch.from_numpy(pts), torch.tensor(label)

# =========================
# 4. 小型 PointNet模型
#点云是无序的点集，网络必须对点的排列顺序不敏感。
#PointNet 的做法：先用一个多层感知机（MLP）对每个点独立提取特征（得到每个点的 256 维特征向量），然后对所有点的特征向量做逐元素的最大池化（max pooling），得到一个全局的 256 维向量，最后用全连接层分类。
#为什么用最大池化？因为最大值操作对输入顺序不敏感，而且能捕捉到每个点云最显著的特征（例如某个方向最突出的点）。
# =========================
class SmallPointNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.mlp = nn.Sequential(       #逐点特征提取 MLP。每个点都过同样的网络，但参数共享。
            nn.Linear(3, 64),       #把每个点从3维映射到64维
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )
        self.cls_head = nn.Sequential(      #分类头
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):       #前向传播
        # x: [B, P, 3]
        feat = self.mlp(x)             # [B, P, 256]
        global_feat, _ = torch.max(feat, dim=1)  # [B, 256]  PointNet灵魂、在“点这个维度”上做 max pooling，即把 128 个点的信息压缩成一个全局向量
        logits = self.cls_head(global_feat)
        return logits

# =========================
# 5. 训练与评估
# =========================
def evaluate(model, loader, device):        #评估函数专门来算测试准确率
    model.eval()        #切换到评估模式（关闭 Dropout、BatchNorm 等训练专用行为）
    total = 0
    correct = 0
    with torch.no_grad():       #不需要计算梯度，节省内存和计算
        for pts, labels in loader:
            pts = pts.to(device)
            labels = labels.to(device)
            logits = model(pts)
            pred = logits.argmax(dim=1)     #取分数最高的类别索引
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return correct / total

def main():     #主函数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")       #如果GPU可用，就用CUDA；否则退回CPU
    print("Using device:", device)

    #建数据集（训练集、测试集各建一份）
    train_set = ToyPointCloudDataset(split="train", samples_per_class=200, n_points=256)
    test_set = ToyPointCloudDataset(split="test", samples_per_class=200, n_points=256)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)

    #建模型、Adam优化器(负责根据梯度更新模型参数)、损失函数(多分类任务的标准损失)
    model = SmallPointNet(num_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    #训练5轮
    epochs = 5
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for pts, labels in train_loader:
            pts = pts.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()               #清空之前梯度
            logits = model(pts)                 #前向传播
            loss = criterion(logits, labels)    #用预测结果和真实标签计算损失
            loss.backward()                     #反向传播求每个参数的梯度
            optimizer.step()                    #更新参数
            running_loss += loss.item()         #记录损失

        #每轮结束后评估
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Test Acc: {acc:.4f}")

    #保存模型。把模型参数保存成文件，以后可以重新加载它，不用每次都重训。
    torch.save(model.state_dict(), "toy_pointcloud_model.pth")
    print("Training done. Model saved to toy_pointcloud_model.pth")

#程序入口
if __name__ == "__main__":
    main()