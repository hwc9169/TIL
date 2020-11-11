# CNN은  합성곱층(Convolution layer)와 풀링층(Pooling layer)로 구성된다
# 이미지의 공간적인 구조 정보를 보존하면서 학습하는 방법이 필오하기에 합성곱 신경망이 생겼다.

import torch
import torch.nn as nn
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.nn.init
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

lr = 0.001
epochs = 15
batch_size = 100

mnist_train = dataset.MNIST(root='MNIST_data/',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True
                            )
mnist_test = dataset.MNIST(root='MNIST_data/',
                           train=False,
                           transform=transforms.ToTensor,
                           download=True)
data_load = DataLoader(dataset=mnist_train,
                       batch_size=batch_size,
                       shuffle=True,
                       drop_last=True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), padding=1, stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), padding=0, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, (3, 3), padding=1, stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), padding=0, stride=(2, 2))
        )

        self.fc = nn.Linear(7 * 7 * 64, 10, bias=True)

        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


model = CNN().to(device)
batch = len(data_load)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epcoh in range(epochs + 1):
    avg_cost = 0
    for x, y in data_load:
        x = x.to(device)
        y = y.to(device)
        h = model(x)
        cost = criterion(h, y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        avg_cost += cost / batch

    print('Epoch : {:4d}/{} Cost : {:.6f}'.format(epcoh, epochs, avg_cost))

with torch.no_grad():
    x_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    y_test = mnist_test.test_labels.to(device)

    prediction = model(x_test)
    correct_prediction = torch.argmax(prediction, 1) == y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy', accuracy.item()) 