import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print("다음 기기로 학습합니다: ",device)

random.seed(777)
torch.manual_seed(777)

if device =='cuda':
    torch.cuda.manual_seed_all(777)

epochs = 20
batch_size =100

train = dataset.MNIST(root='MNIST_data/',
                        train=True,
                        transform = transforms.ToTensor(),
                        download=True)

test = dataset.MNIST(root='MNIST_data/',
                     train=False,
                     transform = transforms.ToTensor(),
                     download=True)

data_loader = DataLoader(dataset = train,
                         batch_size = batch_size,
                         shuffle=True,
                         drop_last=True) #마지막 배치를 버린다

linear = nn.Linear(784, 10, bias=True).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(linear.parameters(),lr=0.1)

for epoch in range(epochs+1): # 앞서 training_epochs의 값은 15로 지정함.
    avg_cost = 0
    total_batch = len(data_loader)

    for X, Y in data_loader:
        # 배치 크기가 100이므로 아래의 연산에서 X는 (100, 784)의 텐서가 된다.
        X = X.view(-1, 28 * 28).to(device)
        # 레이블은 원-핫 인코딩이 된 상태가 아니라 0 ~ 9의 정수.
        Y = Y.to(device)
        hypothesis = linear(X)
        cost = criterion(hypothesis, Y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        avg_cost += cost
    avg_cost /=  total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
print('Learning finished')

with torch.no_grad():
    x_test = test.test_data.view(-1,28,28).float().to(device)
    y_test = test.test_labels.to(device)

    prediction = linear(x_test)
    correct_prediction = torch.argmax(prediction,1) ==y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:',accuracy.item())

    r = random.randint(0, len(test) - 1)
    x_single_data = test.test_labels[r:r+1].view(-1,28*28).float().to(device)
    y_single_data = test.test_labels[r:r+1].to(device)

    print('Label: ',y_single_data.item())
    single_prediction = linear(x_single_data)
    print('Prediction', torch.argmax(single_prediction,1).item())

    plt.imshow(test.test_data[r:r+1].view(28,28), cmap='Greys' , interpolation='nearest')
    plt.show()
