# - 데이터 가져오기
# - 유효성 검사 데이터셋 만들기
# - CNN 모델 구축
# - 모델 학습 및 검증

# --------- 데이터 가져오기 ---------
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets
import torch.nn.functional as F
transformation = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307),(0.3081))]
)

train_dataset = datasets.MNIST('./data/', train=True, transform=transformation, download=True)
test_dataset = datasets.MNIST('./data/', train=False, transform=transformation, download=True)
print('------Dataset------')
print(train_dataset)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=32)


# -------- 이미지 시각화 ---------
import matplotlib.pyplot as plt
import numpy as np
def plot_img(img):
    image = img.numpy()[0]
    print(image.shape)
    # mean = 0.1307
    # std = 0.3081
    # image = ((mean*image) + std)
    plt.imshow(image, cmap='gray')
    plt.show()

# sample_data = next(iter(train_loader))
# plot_img(sample_data[0][1])
# plot_img(sample_data[0][2])

print(len(train_loader.dataset))

print(len(train_loader))
# -------- 아키텍쳐 구현 ---------
from torch.autograd import Variable
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), (2,2)))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

def fit(epoch, model, data_loader, phase='training', volatile=False):
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        volatile=True
    running_loss = 0.0
    running_correct = 0

    for batch_idx, (data, target) in enumerate(data_loader):
        if is_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile), Variable(target)
        if phase == 'training':
            optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
    
        running_loss += loss
        pred = torch.argmax(output, 1)
        running_correct += torch.sum(pred == target)

        if phase == 'training':
            loss.backward()
            optimizer.step()

    loss = running_loss/len(data_loader.dataset)
    accuracy = 100. * running_correct/len(data_loader.dataset)

    print('{0} loss is {1:.4f} and {0} accuracy is {2}/{3}={4:.4f}'.format(phase, loss, running_correct, len(data_loader.dataset), accuracy))
    return loss, accuracy

is_cuda=False
model = Net()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
train_losses, train_accuracy = [], []
val_losses, val_accuracy = [], []
for epoch in range(1, 20):
    epoch_loss, epoch_accuracy = fit(epoch, model, train_loader, phase='training')
    val_epoch_loss, val_epoch_accuracy = fit(epoch, model, test_loader, phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)

plt.plot(range(1, len(train_losses)+1), train_losses, 'bo', label='학습 오차')
plt.plot(range(1, len(val_losses)+1), val_losses, 'r', label='검증 오차')
plt.legend()