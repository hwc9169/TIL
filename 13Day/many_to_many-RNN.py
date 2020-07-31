import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

input_str = 'apple'
label_str = 'pple!'
char_vocab = sorted(list(set(input_str+label_str)))
vocab_size = len(char_vocab)
# print(char_vocab)

input_size = vocab_size
hidden_size = 5
output_size = 5
learning_rate = 0.1

char_to_index = { v : i for i, v in enumerate(char_vocab)}
# print(char_to_index)

index_to_char = {i : v for i, v in enumerate(char_vocab)}
# print(index_to_char)

x_data = [char_to_index[i] for i in list(input_str)]
y_data = [char_to_index[i] for i in list(label_str)]
#print(x_data)
#print(y_data)



x_one_hot = [np.eye(vocab_size)[x] for x in x_data]
#print(x_one_hot)

X = torch.FloatTensor(x_one_hot)
Y = torch.LongTensor(y_data)
X = X.unsqueeze(0)
Y = Y.unsqueeze(0)
#print(X.shape)
#print(Y.shape)

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x):
        x, status = self.rnn(x)
        x = self.fc(x)

        return x

net = Net(input_size, hidden_size, output_size)
outputs = net(X)
#print(outputs.shape)

epochs = 100
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

for epoch in range(epochs+1):
    outputs = net(X)

    cost = criterion(outputs.view(-1, output_size), Y.view(-1))
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    result = np.squeeze(outputs.data.numpy().argmax(axis=2))
    result_str = ''.join([index_to_char[x] for x in result])

    print("Epoch : {:4d}/{} cost : {:.7f} prediction : {}, label : {} prediction str : {}".format(epoch, epochs, cost.item(), result, y_data, result_str))

print(result)
print(result_str)