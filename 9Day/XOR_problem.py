import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

x = torch.FloatTensor([[0,0],[0,1],[1,0],[1,1]]).to(device)
y = torch.FloatTensor([[0],[1],[1],[0]])

linear1 = nn.Linear(2,2,bias=True)
sigmoid = nn.Sigmoid()
linear2 = nn.Linear(2,1,bias=True)
sigmoid2 = nn.Sigmoid()
model = nn.Sequential(linear1,sigmoid,linear2,sigmoid2).to(device)


criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(),lr=1)

epochs = 10000
for epoch in range(epochs+1):
    h = model(x)
    cost = criterion(h,y)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch%100 ==0:
        print("Epoch : {:4d}/{} Cost : {:.6f}".format(epoch,epochs,cost.item()))


with torch.no_grad():
    hypothesis = model(x)
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == y).float().mean()
    print('모델의 출력값(Hypothesis): ', hypothesis.detach().cpu().numpy())
    print('모델의 예측값(Predicted): ', predicted.detach().cpu().numpy())
    print('실제값(Y): ', y.cpu().numpy())
    print('정확도(Accuracy): ', accuracy.item())