import torch
import torch.nn as nn
import torch.nn.functional as F

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3,1)
    def __call__(self,x): #forward 함수의 경우 model 객체를 데이터와 함께 호출하면 자동으로 실행 된다 ex)model(x_train)
        return self.fc1(x)

model = Net()
optimizer = torch.optim.SGD(model.parameters(),lr=1e-5)

epochs = 20000
for epoch in range(epochs+1):
    h = model(x_train)
    cost = F.mse_loss(h,y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        # 100번마다 로그 출력
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch,epochs, cost.item()))
x_test  = torch.FloatTensor([[73,80,75]])
y_test = model(x_test)
print(y_test)


