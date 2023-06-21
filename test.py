import torch.cuda
import torch_directml
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

dml = torch_directml.device()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_data_set = datasets.CIFAR10('./dataset', train=False, transform=transform, download=True)
# 加载数据集
test_data_loader = DataLoader(test_data_set, batch_size=64, shuffle=True)

# 损失函数
loss_function = torch.nn.CrossEntropyLoss()

# 定义网络
myModel = torch.load("model/CNN_model_100.pth")

# 使用GPU
use_gpu = torch.cuda.is_available()
if use_gpu:
    print('GPU')
    myModel = myModel.cuda()
else:
    print('Directml')
    myModel = myModel.to(dml)

# 损失
test_total_loss = 0.0
# 准确率
test_total_acc = 0.0
# 数据集大小
test_data_size = len(test_data_set)

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

print("test total loss:{},acc:{}".format(test_total_loss, test_total_acc / test_data_size))
