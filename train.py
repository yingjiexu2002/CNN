import torch.cuda
import torch_directml
from torch.optim import SGD, Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from net import MyModel, AlexNet
import os

dml = torch_directml.device()

# 训练轮数
epochs = 100
batch_size = 32
# 损失函数
loss_function = torch.nn.CrossEntropyLoss()
# 学习率
learning_rate = 0.01
learning_rate_after_60epoch = 0.001

# 模型存储路径
model_dir = "model/"
writer = SummaryWriter(log_dir='logs')

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 水平翻转
    transforms.RandomCrop(32, padding=4),  # 随机裁剪
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_data_set = datasets.CIFAR10('./dataset', train=True, transform=transform, download=True)
test_data_set = datasets.CIFAR10('./dataset', train=False, transform=transform, download=True)

# 加载数据集
train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=True)

# 定义网络
myModel = AlexNet()

# 使用GPU
use_gpu = torch.cuda.is_available()
if use_gpu:
    print('GPU')
    myModel = myModel.cuda()
else:
    print('Directml')
    myModel = myModel.to(dml)

# 优化器
optimizer = SGD(myModel.parameters(), lr=learning_rate)
# optimizer = Adam(myModel.parameters(), lr=learning_rate)
# 数据集大小
train_data_size = len(train_data_set)
test_data_size = len(test_data_set)
print('train_size = {}, test_size = {}'.format(train_data_size, test_data_size))

for epoch in range(epochs):
    print("训练轮数：{}/{}".format(epoch + 1, epochs))
    if epoch == 60:
        optimizer = SGD(myModel.parameters(), lr=learning_rate_after_60epoch)
    # 损失
    train_total_loss = 0.0
    test_total_loss = 0.0
    # 准确率
    train_total_acc = 0.0
    test_total_acc = 0.0

    # 开始训练
    for data in train_data_loader:
        inputs, labels = data
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        else:
            inputs = inputs.to(dml)
            labels = labels.to(dml)

        outputs = myModel(inputs)

        # 计算损失
        optimizer.zero_grad()
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        # 得到预测值最大的值和下标
        _, index = torch.max(outputs, 1)
        acc = torch.sum(index == labels).item()
        train_total_loss += loss.item()
        train_total_acc += acc

    if ((epoch + 1) % 10) == 0:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        # 保存模型
        torch.save(myModel, model_dir + 'CNN_model_{}.pth'.format(epoch + 1))

    # 测试
    with torch.no_grad():
        for data in test_data_loader:
            inputs, labels = data
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            else:
                inputs = inputs.to(dml)
                labels = labels.to(dml)

            outputs = myModel(inputs)

            loss = loss_function(outputs, labels)

            # 得到预测值最大的值和下标
            _, index = torch.max(outputs, 1)
            acc = torch.sum(index == labels).item()
            test_total_loss += loss.item()
            test_total_acc += acc

    print("train total loss:{},acc:{}  test total loss:{},acc:{}".format(train_total_loss,
                                                                         train_total_acc / train_data_size,
                                                                         test_total_loss,
                                                                         test_total_acc / test_data_size))
    # 将训练过程保存下来
    writer.add_scalar('loss/train', train_total_loss, epoch + 1)
    writer.add_scalar('acc/train', train_total_acc / train_data_size, epoch + 1)
    writer.add_scalar('loss/test', test_total_loss, epoch + 1)
    writer.add_scalar('acc/test', test_total_acc / test_data_size, epoch + 1)
