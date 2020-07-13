import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits

digits = load_digits()
x = digits.data
y = digits.target

model = nn.Sequential(
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,16),
    nn.ReLU(),
    nn.Linear(16,10)
)

x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.int64)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

losses = []

epochs =100
for epoch in range(epochs+1):
    h = model(x)
    cost = criterion(h,y)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch%10 ==0:
        print("Epoch {:4d}{}, Cost {:.6f}".format(epoch,epochs,cost.item()))

    losses.append(cost)

plt.plot(losses)
plt.show()


