import torch
import torch.nn as nn
import torch.nn.functional as F

z = torch.rand(3,5,requires_grad=True)
y = torch.randint(5,(3,)).long()

y_one_hot = torch.zeros_like(h)
y_one_hot.scatter_(1,y.unsqueeze(1),1)
print(y)
print(y_one_hot)
#h = torch.softmax(z,dim=1)
#cost = (y_one_hot * -F.log_softmax(z, dim=1)).sum(dim=1).mean()
cost = F.nll_loss(F.log_softmax(z,dim=1),y)
print(cost)
print(torch.log(F.softmax(z, dim=1)))
print(F.log_softmax(z,dim=1))
#print(h.sum(dim=1))

#nll : Negative Log Likelihood