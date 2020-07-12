import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)
x_data =[[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)


model = nn.Sequential(
    nn.Linear(2,1),
    nn.Sigmoid()
)

optimizer = optim.SGD(model.parameters(), lr=1)

epochs = 1000
for epoch in range(epochs+1):
    h = model(x_train)
    cost = F.binary_cross_entropy(h,y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch %10 ==0:
        prediction = h >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == y_train
        accuracy = correct_prediction.sum().item()/len(correct_prediction)
        print('Epoch: {:4d}/{}, Cost: {:.6f} Accuracy {:2.2f}%'.format(epoch,epochs,cost.item(),accuracy*100))

#로지스틱 회귀는 인공 신경망으로 간주할 수 있다.
#로지스틱 회귀를 식으로 표현하자면 H(x) = sigmoid(x1w1 + x2w2 + b)이다.
