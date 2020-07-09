import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

y_one_hot = torch.zeros(8,3)
y_one_hot.scatter_(1,y_train.unsqueeze(1),1)
# low level --------------------------------------------------
'''
W = torch.zeros((4,3),requires_grad=True)
b = torch.zeros(1,requires_grad=True)
optimizer = optim.SGD([W,b],lr=0.1)

epochs =3000
for epoch in range(epochs+1):
    h = x_train.matmul(W) + b
    h = F.softmax(h, dim=1)
    cost = (y_one_hot * -torch.log(h)).sum(dim=1).mean()

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch %100 ==0:
        print("Epoch : {:4d}/{} Cost : {:.6f}".format(epoch,epochs, cost.item()))
'''
# high level---------------------------------------------------
'''
W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
optimizer = optim.SGD([W, b], lr=0.1)

epochs = 3000
for epoch in range(epochs+1):
    z = x_train.matmul(W) + b
    cost = F.cross_entropy(z, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch%100 ==0:
        print("Epoch : {:4d}/{} Cost : {:.6f}".format(epoch, epochs, cost.item()))
'''
#nn.Module로 구현하기 -----------------------------------------
'''
model = nn.Sequential(
    nn.Linear(4,3)
)
optimizer = optim.SGD(model.parameters(),lr = 0.1)

epochs = 3000
for epoch in range(epochs):
    h = model(x_train)
    cost = F.cross_entropy(h,y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch%100 ==0:
        print("Epoch : {:4d}/{} Cost : {:.6f}".format(epoch, epochs, cost.item()))
#nll : Negative Log Likelihood
'''
#nn.Module을 상속받은 클래스로 구현---------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4,3)
        self.fc2 = nn.Softmax()

    def forward(self,x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = Net()
optimizer = optim.SGD(model.parameters(),lr=0.1)
epochs = 3000
for epoch in range(epochs):
    h = model(x_train)
    cost = (y_one_hot*-torch.log(h)).sum(dim=1).mean()

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch%100 ==0:
        print("Epoch : {:4d}/{} Cost : {:.6f}".format(epoch, epochs, cost.item()))
'''
                      higt                 --->                   low
F.cross_entropy = F.nll_loss(F.softmax(z),y) = (y_one_hot * -F.log_softmax(z,dim=1)).sum(dim=1).mean() = (y_one_hot * -torch.log(F.softmax(z,dim=1)).sum(dim=1).mean()


'''

