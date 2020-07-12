import torch
import torch.nn as nn
import torch.nn.functional as F

x_train  =  torch.FloatTensor([[73,  80,  75],
                               [93,  88,  93],
                               [89,  91,  90],
                               [96,  98,  100],
                               [73,  66,  70]])
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])
model = nn.Linear(3,1)
optimizer = torch.optim.SGD(model.parameters(),lr=0.000001)

epochs = 50000
for epoch in range(epochs+1):
    h = model(x_train)
    cost = F.mse_loss(h, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch%100 ==0 :
        print('Epoch : {:4d}/{} Cost : {:.6f}'.format(epoch,epochs,cost.item()))

print(model(torch.FloatTensor([[73,80,75]])))


