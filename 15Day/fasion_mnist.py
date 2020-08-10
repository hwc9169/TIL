import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

output_mapping = {
    0: "T-shirt/Top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot"
}

# 하이퍼파라미터
batch_size = 64
epochs = 100
lr = 0.0001

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5,), std=(0.5,))])
trainset = datasets.FashionMNIST('data/trainset', download=True, train=True, transform=transform)
testset = datasets.FashionMNIST('data/testset', download=True, train=False, transform=transform)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, drop_last=True)


# 모델
class ClassifyFashion(nn.Module):
    def __init__(self):
        super(ClassifyFashion, self).__init__()
        self.fc1 = nn.Linear(784, 128, bias=True)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 32, bias=True)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 10, bias=True)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x - self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x


model = ClassifyFashion().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

print("training machine with {}".format(device))
for epoch in range(1, epochs + 1):
    avg_cost = 0
    for x, y in trainloader:
        x = x.to(device)
        y = y.to(device)
        output = model(x.view(batch_size, -1))
        cost = criterion(output, y)
        avg_cost += cost.item() / len(trainloader)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

    print('EPOCH : {:3d}/{}\tCost : {:.6f}'.format(epoch, epochs, avg_cost))

with torch.no_grad():
    x_test = testset.test_data.view(-1,28*28).float().to(device)
    y_test = testset.test_labels.to(device)
    prediction = model(x_test)

    cost = criterion(prediction, y_test)
    accuracy = sum(prediction.argmax(1) == y_test).float()/len(x_test)
    print('cost : {:.6f}\taccuracy : {}'.format(cost.item(), accuracy))