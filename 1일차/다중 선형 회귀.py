import torch
import torch.nn as nn
import torch.nn.functional as F

x_train = torch.FloatTensor([[1, 2, 3],
                             [2, 3, 4],
                             [3, 4, 5],
                             [4, 5, 6],
                             [7, 8, 9]])
y_train = torch.FloatTensor([6,9,12,15,24])


model = nn.Linear(3,1)
optimizer = torch.optim.SGD(model.parameters(),lr=0.000001)

epochs = 50000
for epoch in range(epochs+1):
    h = model(x_train)
    cost = F.mse_loss(h, y_train)
    costs.append(cost)
    Ws.append()
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch%100 ==0 :
        print('Epoch : {:4d}/{} Cost : {:.6f}'.format(epoch,epochs,cost.item()))

print(model(torch.FloatTensor([[5,6,7]])))
print(list(model.parameters()))


