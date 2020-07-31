import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

char_set = list(set(sentence))
char_dic = {v : i for i, v in enumerate(char_set)}
dic_size = len(char_dic)

input_size = dic_size
hidden_size = dic_size
output_size = dic_size
layer = 2
sequence_size = 10
learning_rate = 0.1

x_data = []
y_data = []

for i in range(0, len(sentence) - sequence_size):
    x_str = sentence[i : i + sequence_size]
    y_str = sentence[i + 1 : i + 1 + sequence_size]
    #print(i, x_str, '->', y_str)

    x_data.append([char_dic[c] for c in x_str])
    y_data.append([char_dic[c] for c in y_str])
#print(x_data[0])
#print(y_data[0])

x_one_hot = [np.eye(dic_size)[x] for x in x_data]
X = torch.FloatTensor(x_one_hot)
Y = torch.LongTensor(y_data)
#print(X.shape)
#print(Y.shape)

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(Net, self).__init__()

        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, x):
        x, status = self.rnn(x)
        x = self.fc(x)

        return x

net = Net(input_size, hidden_size, output_size, layer)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

epochs = 100

for epoch in range(epochs):
    output = net(X)
    cost = criterion(output.view(-1, dic_size), Y.view(-1))

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    results = output.data.numpy().argmax(axis=2)
    result_str = ""
    for i, result in enumerate(results):
        if i == 0:
            result_str += ''.join([char_set[i] for i in result])
        else :
            result_str += char_set[result[-1]]

print(result_str)
