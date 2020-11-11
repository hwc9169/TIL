import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.init

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)

if device == 'cuda':
    torch.cuda.manual_seed_all(777)

batch_size = 100
lr = 0.001
epochs = 5

mnist_train = datasets.MNIST(root='MNIST_data/',
                             train=True,
                             transform=transforms.ToTensor(),
                             download=True)

mnist_test = datasets.MNIST(root='MNIST_data/',
                            train=False,
                            transform=transforms.ToTensor(),
                            download=True)

data_load = DataLoader(dataset=mnist_train, 
                       shuffle=True,
                       drop_last=True,
                       batch_size=batch_size)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2, padding=1)
        )
        self.fc1 = torch.nn.Linear(4 * 4 * 128, 625, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc2 = nn.Linear(625, 10, bias=True)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.shape[0], -1)
        x = self.layer4(x)
        x = self.fc2(x)

        return x


model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

total_batch = len(data_load)
print('총 배치수 {}'.format(total_batch))

for epoch in range(epochs + 1):
    avg_cost = 0
    for x, y in data_load:
        x = x.to(device)
        y = y.to(device)

        h = model(x)
        cost = criterion(h, y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

        print('Epoch {:4d}/{}, Cost{:.6f}'.format(epoch, epochs, avg_cost))

with torch.no_grad():
    x_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    y_test = mnist_test.test_labels.to(device)

    prediction = model(x_test)
    correct_prediction = torch.argmax(prediction, 1) == y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy', accuracy.item())
