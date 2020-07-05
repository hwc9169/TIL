import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])

W = torch.zeros(1,requires_grad= True)
b = torch.zeros(1, requires_grad= True)

optimizer = optim.SGD([W,b], lr=0.01)
#optimizer.zero_grad() # 미분을 통해 얻은 기울기를 초기화
#cost.backward() # W와 b에 대한 기울기가 계산된다
#optimizer.step() #인수로 들어간 [W,b]에서 리턴되는 변수의 기울기에 학습률을 곱하여 뺀값을 업데잍한다

#비용 함수 == 손실 함수 == 오차 함수 == 목적 함수
#비용함수(cost function) : 오차를 구하는 함수다. 그냥 오차를 더해버리면 음수가 나오는 경우가 있기 때문에 제곱 합이나 cross-entropy방법을 쓴다
#옵티마이저 : 옵티마이저란 비용함수의 값을 최소로하는 방법을 말한다. 그래서 결국 옵티마이저와 비용함수는 아주 밀접한 관계가 있다고 말 할 수 있다.

#꿀팁 : 선형 회귀 문제에 가장 적합한 cost function은 제곱 오차이고 optimizer는 SGE(Stochastic Gradient Descent)
epoch = 10000
for i in range(epoch):
    h = x_train*W+b
    cost = torch.mean((h-y_train)**2)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    if i%100==0:
        print("Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}".format(i,epoch,W.item(),b.item(),cost.item()))
print(10*W+b)
