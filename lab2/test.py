import os
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.callbacks import ProgbarLogger


def unpickle(file):
    """加载 CIFAR-10 数据集的单个文件"""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10_data(data_dir):
    """加载本地 CIFAR-10 数据集"""
    x_train = []
    y_train = []
    # 加载训练集
    for i in range(1, 6):
        filename = os.path.join(data_dir, f"data_batch_{i}")
        data_dict = unpickle(filename)
        x_train.append(data_dict[b'data'])
        y_train.extend(data_dict[b'labels'])
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.array(y_train)
    # 加载测试集
    test_dict = unpickle(os.path.join(data_dir, "test_batch"))
    x_test = test_dict[b'data']
    y_test = np.array(test_dict[b'labels'])
    # 将数据从一维数组转换为图像格式 (32x32x3)
    x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    # 归一化
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    # one-hot 编码
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


# 构建CNN模型
def build_model(optimizer, dropout_rate=0.0, l2_reg=0.0, data_augmentation=False):
    model = models.Sequential()
    # 第一层卷积
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3),
                            kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(dropout_rate))
    # 第二层卷积
    model.add(layers.Conv2D(64, (3, 3), activation='relu',
                            kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(dropout_rate))
    # 第三层卷积
    model.add(layers.Conv2D(64, (3, 3), activation='relu',
                            kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(dropout_rate))
    # 全连接层
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(10, activation='softmax'))  # 输出层
    # 编译模型
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 训练模型
def train_model(model, x_train, y_train, x_test, y_test, batch_size, epochs, data_augmentation=False):
    if data_augmentation:
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        )
        datagen.fit(x_train)
        history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            steps_per_epoch=len(x_train) / batch_size,
                            callbacks=[ProgbarLogger(count_mode='steps', stateful_metrics=None)])
    else:
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(x_test, y_test),
                            callbacks=[ProgbarLogger(count_mode='steps', stateful_metrics=None)])
    return history


# 绘制损失曲线和准确率曲线
def plot_history(history):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# 绘制混淆矩阵
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# 主函数
def main():
    # 设置 GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # 设置 TensorFlow 仅使用第一个 GPU
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            # 设置 GPU 内存增长，避免 TensorFlow 占用全部 GPU 内存
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("Using GPU for training")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU available, using CPU for training")

    data_dir = "dataset/cifar-10-batches-py"  # 本地数据集路径
    x_train, y_train, x_test, y_test = load_cifar10_data(data_dir)  # 加载本地数据集

    # 超参数
    batch_size = 256
    epochs = 10
    learning_rate = 0.001
    dropout_rate = 0.3
    l2_reg = 0.001

    # 优化器
    optimizers = {
        'SGD': tf.keras.optimizers.SGD(learning_rate=learning_rate),
        'Momentum': tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
        'Adam': tf.keras.optimizers.Adam(learning_rate=learning_rate)
    }

    # 正则化技术
    regularizations = {
        'None': {'dropout_rate': 0.0, 'l2_reg': 0.0, 'data_augmentation': False},
        'Dropout': {'dropout_rate': dropout_rate, 'l2_reg': 0.0, 'data_augmentation': False},
        'L2': {'dropout_rate': 0.0, 'l2_reg': l2_reg, 'data_augmentation': False},
        'DataAugmentation': {'dropout_rate': 0.0, 'l2_reg': 0.0, 'data_augmentation': True}
    }

    for opt_name, optimizer in optimizers.items():
        for reg_name, reg_params in regularizations.items():
            print(f"Training with Optimizer: {opt_name}, Regularization: {reg_name}")
            model = build_model(optimizer, dropout_rate=reg_params['dropout_rate'],
                                l2_reg=reg_params['l2_reg'], data_augmentation=reg_params['data_augmentation'])
            history = train_model(model, x_train, y_train, x_test, y_test, batch_size, epochs,
                                  data_augmentation=reg_params['data_augmentation'])
            plot_history(history)

            # 模型评估
            y_pred = model.predict(x_test)
            print(f"Confusion Matrix for Optimizer: {opt_name}, Regularization: {reg_name}")
            plot_confusion_matrix(y_test, y_pred)


if __name__ == "__main__":
    main()