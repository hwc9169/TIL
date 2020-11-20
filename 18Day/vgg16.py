from torchvision import models, datasets
from torch.autograd import Variable
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F 
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import matplotlib.pyplot as plt

vgg = models.vgg16(pretrained=True)
print(vgg)

transformation = transforms.Compose(
    [transforms.Resize((224,224)),transforms.ToTensor()]
    )

train_dataset = ImageFolder('../16Day/train', transform=transformation)
test_dataset = ImageFolder('../16Day/valid', transform=transformation)
# train_dataset = datasets.MNIST('./data/', transform=transformation, train=True, download=True)
# test_dataset = datasets.MNIST('./data/', transform=transformation, train=False, download=True)
print('------Dataset------')
print(train_dataset)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=32)


def fit(epoch, model, data_loader, phase='training'):
    if phase=='training':
        model.train()
    if phase=='validation':
        model.eval()
        volatile=True

    running_loss = 0.0
    running_correct = 0.0
    
    for batch_index, (data, target) in enumerate(data_loader):
        if is_cuda:
            data, target = data.cuda(), target.cuda()

        if phase== 'training':
            optimizer.zero_grad()

        print(epoch)
        output = model(data)
        loss = nn.NLLLoss()(output, target)

        if phase== 'training':
            loss.backward()
            optimizer.step()
        if phase== 'validation':
            pass
            # exp_lr_scheduler.step()
        
    loss = running_loss/len(data_loader.dataset)
    accuracy = 100. * running_correct/len(data_loader.dataset)

    print ('[{}] epoch: {:2d} loss: {:.8f} accuracy: {:.8f}'.format(phase, epoch, loss, accuracy))
    return loss, accuracy

is_cuda = False
optimizer = optim.SGD(vgg.classifier.parameters(), lr=0.0001, momentum=0.5)
# exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 특징을 추출하는 부분의 레이어는 고정시키고 fully connected 부분만 가중치를 학습시킨다.
for param in vgg.features.parameters():
    param.repuired_grad = False

#1000개의 출력에서 2개만 분류하도록 한다.(개, 고양이)
vgg.classifier[6].out_features = 2 

# Dropout을 0.5에서 0.2로 수정
for layer in vgg.classifier.children:
    if(type(layer)==nn.Dropout):
        layer.p = 0.2

train_loss, train_accuracy = [], []
val_loss, val_accuracy = [], []
for epoch in range(0, 1):
    epoch_loss, epoch_accuracy = fit(epoch, vgg, train_loader, phase='training')
    val_epoch_loss, val_epoch_accuracy = fit(epoch, vgg, test_loader, phase='validation')
    train_loss.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)


plt.rcParams["figure.figsize"] = (15, 6)
fig = plt.figure()
loss = fig.add_subplot()
accuracy = fig.add_subplot()

loss.plot(range(1, len(train_loss)+1), train_loss, 'b', label='학습 오차')
loss.plot(range(1, len(val_loss)+1), val_loss, 'r', label='검증 오차')
loss.legend()

accuracy.plot(range(1, len(train_accuracy)+1), train_accuracy, 'b', label='학습 정확도')
accuracy.plot(range(1, len(val_accuracy)+1), val_accuracy, 'b', label='검증 정확도')
accuracy.legend()

plt.show()
