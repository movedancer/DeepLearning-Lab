import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 自定义的CIFAR-10数据集类
class CIFAR10Dataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.transform = transform
        self.train = train
        self.data_dir = data_dir
        self.data, self.targets = self.load_data()

    def load_data(self):
        if self.train:
            filenames = [f'data_batch_{i}' for i in range(1, 6)]
        else:
            filenames = ['test_batch']

        data = []
        targets = []

        for filename in filenames:
            file_path = os.path.join(self.data_dir, filename)
            with open(file_path, 'rb') as f:
                import pickle
                entry = pickle.load(f, encoding='latin1')
                data.append(entry['data'])
                targets.extend(entry['labels'])

        data = np.vstack(data).reshape(-1, 3, 32, 32)
        data = data.transpose((0, 2, 3, 1))  # 转换为HWC格式
        return data, targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx], self.targets[idx]
        image = Image.fromarray(image)  # 将NumPy数组转换为PIL图像方便后续数据增强操作
        if self.transform:
            image = self.transform(image)
        return image, label

# 自定义的卷积神经网络CNN模型
class CIFAR10CNN(nn.Module):
    def __init__(self, dropout_rate=0.5, l2_reg=0.0):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.l2_reg = l2_reg

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# 数据预处理
def data_preprocess(data_dir, batch_size, data_augmentation=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    #数据增强
    if data_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        train_transform = transform

    # 使用自定义的CIFAR-10 Dataset加载数据
    train_dataset = CIFAR10Dataset(data_dir=data_dir, train=True, transform=train_transform)
    test_dataset = CIFAR10Dataset(data_dir=data_dir, train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# 训练模型
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    losses = []
    train_accuracies = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total
        losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}')
    return losses, train_accuracies

# 测试模型
def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy, all_preds, all_labels

# 绘制损失曲线和准确率曲线
def plot_curves(losses, train_accuracies, test_accuracy):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.axhline(y=test_accuracy, color='r', linestyle='--', label=f'Test Accuracy ({test_accuracy:.4f})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.show()

# 绘制混淆矩阵
def plot_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# 主函数
def main(optimizer_name='Adam', use_dropout=True, use_l2_reg=True, use_data_augmentation=True):
    # 超参数设置
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 20
    dropout_rate = 0.5 if use_dropout else 0.0
    l2_reg = 0.001 if use_l2_reg else 0.0
    data_augmentation = use_data_augmentation

    # 数据集路径
    data_dir = './dataset/cifar-10-batches-py'  # 本地数据集路径

    # 数据加载
    train_loader, test_loader = data_preprocess(data_dir, batch_size, data_augmentation)

    # 训练设备、模型、损失函数和优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CIFAR10CNN(dropout_rate=dropout_rate, l2_reg=l2_reg).to(device)
    criterion = nn.CrossEntropyLoss()

    # 选择优化器
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'Momentum':
        momentum = 0.9
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    else:
        raise ValueError("Unsupported optimizer. Choose from 'Adam', 'SGD', or 'Momentum'.")

    # 输出模型设置信息
    print("优化器:", optimizer_name,
          "正则化策略:use_dropout:", use_dropout,
          ", use_l2_reg", use_l2_reg,
          ", use_data_augmentation", use_data_augmentation)
    # 训练模型
    losses, train_accuracies = train_model(model, train_loader, criterion, optimizer, num_epochs, device)

    # 测试模型
    test_accuracy, preds, labels = test_model(model, test_loader, device)

    # 绘制损失曲线和准确率曲线
    plot_curves(losses, train_accuracies, test_accuracy)

    # 绘制混淆矩阵
    plot_confusion_matrix(labels, preds)

# 调用主函数并选择优化器和正则化技术
if __name__ == '__main__':
    main(optimizer_name='Adam', use_dropout=True, use_l2_reg=True, use_data_augmentation=True)