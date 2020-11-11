import torch
import torch.nn as nn

#1. RNN
input_size = 5
hidden_size = 8

inputs = torch.Tensor(1, 10, 5)

cell = nn.RNN(input_size, hidden_size, batch_first=True)
outputs, status = cell(inputs)

#2. Deep RNN
import torch.nn as nn
import torch

inputs = torch.Tensor(1, 10, 5)
cell = nn.RNN(input_size=5, hidden_size=8, num_layers=2, batch_first=True)
outputs, status = cell(inputs)

print(outputs.shape) #torch.Size([1, 10, 8])
print(status.shape) #torch.Size([2, 1, 8]]

#3. Bidirectional RNN
import torch.nn as nn
import torch

inputs = torch.Tensor(1, 10, 5)
cell = nn.RNN(input_size=5, hidden_size=8, num_layers=2, batch_first=True, bidirectional=True)
outputs, status = cell(inputs)

print(outputs.shape) #torch.Size([1, 10, 8x2])
print(status.shape) #torch.Size([2x2, 1 ,8]) 
