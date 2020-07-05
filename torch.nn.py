import torch
import torch.nn as nn
import torch.nn.functional as F

x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])

model = nn.Linear(1,1)
print(list(model.parameters()))

optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

epochs = 10000
for epoch in range(epochs+1):
    h = model(x_train)
    cost = F.mse_loss(h,y_train)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch%100==0:
        print("Epoch : {:4d}/{} Cost : {:.6f}".format(epoch,epochs,cost.item()))

print("training end")
x_test = torch.FloatTensor([[4],[5],[6]])
y_test = model(x_test)
print(y_test)
print(list(model.parameters()))
