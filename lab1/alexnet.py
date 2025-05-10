import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

# 数据准备
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将图像尺寸调整为224×224
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化处理
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# AlexNet模型实现
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),  # 第一层卷积
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),  # 第二层卷积
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # 第三层卷积
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # 第四层卷积
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 第五层卷积
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),  # 第一层全连接
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),  # 第二层全连接
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),  # 第三层全连接，10个神经元输出
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

model = AlexNet()
# 检查模型参数规模
print("模型参数规模：", sum(p.numel() for p in model.parameters()))

# 验证模型向前计算的正确性
x = torch.randn(1, 1, 224, 224)
output = model(x)
print("模型输出形状：", output.shape)

# 验证模型反向传播的正确性
criterion = nn.CrossEntropyLoss()
loss = criterion(output, torch.tensor([1]))
loss.backward()
print("反向传播成功")

# 模型训练
class AlexNetTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.logger = TensorBoardLogger(save_dir="logs", name=f"lr_{learning_rate}")

    def train(self, train_loader, val_loader, num_epochs=10):
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset)
            self.logger.log_metrics({"train_loss": epoch_loss, "train_acc": epoch_acc}, step=epoch)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
            self.validate(val_loader, epoch)

    def validate(self, val_loader, epoch):
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)
        self.logger.log_metrics({"val_loss": epoch_loss, "val_acc": epoch_acc}, step=epoch)
        print(f"Epoch {epoch+1}, Val Loss: {epoch_loss:.4f}, Val Acc: {epoch_acc:.4f}")

# 对比不同超参数（学习率）的影响
learning_rates = [0.001, 0.01]
for lr in learning_rates:
    model = AlexNet()
    trainer = AlexNetTrainer(model, learning_rate=lr)
    trainer.train(train_loader, test_loader, num_epochs=4)
    # 测试集最终准确率评估
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(trainer.device), labels.to(trainer.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    print(f'学习率：{lr}, 测试集准确率: {100 * correct / total:.2f}%')
    # 混淆矩阵分析易混淆类别
    conf_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Learning Rate: {lr})')
    plt.show()

    # 可视化第一层卷积核
    first_layer_weights = model.features[0].weight.data.cpu().numpy()
    fig, axes = plt.subplots(8, 8, figsize=(16, 16))
    for i, ax in enumerate(axes.flat):
        if i < len(first_layer_weights):
            ax.imshow(first_layer_weights[i][0], cmap='gray')
            ax.axis('off')
    plt.suptitle(f'First Layer Convolutional Kernels (Learning Rate: {lr})')
    plt.show()