import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
sentence = "Repeat is the best medicine for memory".split()

vocab = list(set(sentence))
#print(vocab)

word2index = {v : i+1 for i, v in enumerate(vocab)}
word2index['<unk>'] = 0
index2word = {i : v for v, i in word2index.items()}
#print(word2index)
#print(index2word)

def build_data(sentence, word2index):
    encoded = [word2index[token] for token in sentence]

    input_seq, label_seq = encoded[:-1], encoded[1:]

    input_seq = torch.LongTensor(input_seq).unsqueeze(0)

    label_seq = torch.LongTensor(label_seq).unsqueeze(0)

    return input_seq, label_seq

X, Y = build_data(sentence, word2index)
# print(X)
# print(Y)

class Net(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size):
        super(Net, self).__init__()

        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size , embedding_dim=input_size)

        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, vocab_size, bias=True)

    def forward(self, x):
        x = self.embedding_layer(x)

        x, status = self.rnn(x)

        x = self.fc(x)

        return x.view(-1, x.size(2))

vocab_size = len(word2index)
input_size =  5
hidden_size = 20


model = Net(vocab_size, input_size, hidden_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

#output = model(X)
#print(output)
print(Y)

def decode(x):
    return [index2word[i] for i in x]

epochs = 200
for epoch in range(epochs+1):

    output = model(X)
    cost = criterion(output, Y.view(-1))

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()


    if epoch % 40 == 0:
        pred = output.argmax(-1).tolist()
        print('Epoch {:4d}/{} cost : {:.6f} \n prediction : {}'.format(epoch, epochs, cost.item(),
                                                                       " ".join(["Repeat"] + decode(pred))))