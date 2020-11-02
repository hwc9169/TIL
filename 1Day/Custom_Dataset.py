import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class CustomDatast(Dataset):
    def __init__(self):
        self.x_data = [[73, 80, 75],
                       [93, 88, 93],
                       [89, 91, 90],
                       [96, 98, 100],
                       [73, 66, 70]]
        self.y_data = [[152], [185], [180], [196], [142]]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x,y

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3,1)

    def forward(self,x):
        return self.fc1(x)


dataset = CustomDatast()
dataloader = DataLoader(dataset=dataset,batch_size=2,shuffle=True)
model = Net()
optimizer = torch.optim.SGD(model.parameters(),lr=1e-5)
epochs = 20
for epoch in range(epochs+1):
    for batch_idx,samples in enumerate(dataloader):
        x_train,y_train = samples
        h = model(x_train)
        cost = F.mse_loss(h,y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print("Epoch {:4d}/{} Batch {:2d}/{:2d} Cost {:.6f}".format(epoch,epochs,batch_idx,len(dataloader),cost.item()))

# 임의의 입력 [73, 80, 75]를 선언
new_var =  torch.FloatTensor([[73, 80, 75]])
# 입력한 값 [73, 80, 75]에 대해서 예측값 y를 리턴받아서 pred_y에 저장
pred_y = model(new_var)
print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y)