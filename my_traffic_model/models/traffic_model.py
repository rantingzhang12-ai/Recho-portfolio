import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

# ======================
# 1. 配置参数
# ======================
class Config:
    TRAIN_ROOT = r"D:\download\archive_traffic\train"
    VAL_ROOT = r"D:\download\archive_traffic\valid"
    IMG_SIZE = 224
    BATCH_SIZE = 32    # 批处理大小
    EPOCHS = 15
    LR = 1e-4
    NUM_CLASSES = 30
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_SAVE_PATH = "final_traffic_sign_model.pth"

# ======================
# 2. 自定义数据集类
# ======================
# 继承PyTorch的Dataset类，实现自定义数据加载逻辑。
class TrafficSignDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if os.path.isfile(img_path):
                    samples.append((img_path, self.class_to_idx[class_name]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')    # 读取图像并转为RGB格式
        if self.transform:
            image = self.transform(image)    # 应用数据增强/预处理
        return image, label

# ======================
# 3. 数据增强和预处理
# ======================
# 训练集增强(随机裁剪、翻转、颜色抖动等)
# 训练集使用强数据增强提升模型泛化能力。
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(Config.IMG_SIZE),  # 随机裁剪并缩放至224x224
    transforms.RandomHorizontalFlip(),              # 随机水平翻转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),      # 颜色抖动
    transforms.ToTensor(),    # 转为张量[0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # ImageNet标准化
])

# 验证集预处理(仅缩放和中心裁剪)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(Config.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ======================
# 4. 创建数据集和数据加载器
# ======================
train_dataset = TrafficSignDataset(Config.TRAIN_ROOT, transform=train_transform)
val_dataset = TrafficSignDataset(Config.VAL_ROOT, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)

# ======================
# 5. 模型构建 (与测试脚本完全匹配)
# ======================
class TrafficSignModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 使用预训练ResNet18作为骨干网络
        # 迁移学习：利用预训练ResNet18的骨干网络提取通用特征。
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # 移除原分类头/全连接层

        # 添加自定义分类头
        # 微调策略：替换分类头以适应交通标志分类任务。
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),    # 正则化：防止过拟合
            nn.Linear(256, num_classes) # 输出30类
        )

    def forward(self, x):
        features = self.backbone(x)         # 提取特征
        return self.classifier(features)    # 分类预测

model = TrafficSignModel(Config.NUM_CLASSES).to(Config.DEVICE)

# ======================
# 6. 损失函数和优化器
# ======================
criterion = nn.CrossEntropyLoss()
# 损失函数：多分类交叉熵损失特别适用于多分类问题（如本例的30类交通标志识别）。
# 它的本质是衡量模型预测的概率分布与真实标签的分布之间的差异，目标是让预测分布尽可能接近真实分布。
optimizer = optim.AdamW(model.parameters(), lr=Config.LR)
# AdamW优化器  不同参数（如ResNet浅层和深层）可以有不同的更新步长，加速收敛。
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)  # 余弦退火学习率

# ======================
# 7. 训练和验证函数
# ======================
# 完整训练流程中的协同作用
# 前向传播：模型计算预测logits → 通过CrossEntropyLoss计算损失。
# 反向传播：损失对模型参数求梯度。
# 优化器更新：AdamW根据梯度更新参数，同时应用解耦的权重衰减。
# 学习率调整：每轮epoch后，CosineAnnealingLR调整学习率。
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 前向传播 → 计算损失 → 反向传播 → 参数更新
    # 使用tqdm显示进度条和实时指标(损失/准确率)
    progress_bar = tqdm(loader, desc="Training", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{100. * correct / total:.2f}%"
        })

    train_loss = running_loss / len(loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # 仅前向传播计算指标(不更新参数)
    # 同样使用tqdm显示进度
    progress_bar = tqdm(loader, desc="Validation", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{100. * correct / total:.2f}%"
        })

    val_loss = running_loss / len(loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

# ======================
# 8. 主训练循环
# ======================
if __name__ == '__main__':
    best_acc = 0.0

    for epoch in range(Config.EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, Config.DEVICE)
        scheduler.step()   # 更新学习率

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch + 1}/{Config.EPOCHS}] | "
              f"LR: {current_lr:.2e} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # 每轮训练后保存验证集最佳模型(best_acc跟踪)。
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"新的最佳模型已保存！验证准确率: {best_acc:.2f}%")

    torch.save(model.state_dict(), "final_traffic_sign_model.pth")
    print("训练完成！最终模型已保存为 final_traffic_sign_model.pth")