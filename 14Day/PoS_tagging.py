import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data, datasets
import torch.optim as optim
import time
import random

SEED=1234
random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TEXT = data.Field(lower=True)
UD_TAGS = data.Field(unk_token=None)
PTB_TAGS = data.Field(unk_token=None)

fields = (("text", TEXT),("udtags", UD_TAGS),("ptbtags", PTB_TAGS))

trainset, validset, testset = datasets.UDPOS.splits(fields)

TEXT.build_vocab(trainset, min_freq=5, vectors="glove.6B.100d")
UD_TAGS.build_vocab(trainset)
PTB_TAGS.build_vocab(trainset)

def tag_percentage(tag_count):
    total_count = sum([count for tag, count in tag_count])
    tag_counts_percentages = [(tag, count, count/total_count) for tag, count in tag_count]

    return tag_counts_percentages

for tag, count, percent in tag_percentage(UD_TAGS.vocab.freqs.most_common()):
    print('{}\t{}\t{:.4f}'.format(tag, count, percent))

batch_size = 64
train_iter, valid_iter, test_iter = data.BucketIterator.splits(
    (trainset, validset, testset),
    batch_size=batch_size,
    device=device
)

batch = next(iter(train_iter))
print(batch)


class RNNPOSTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(RNNPOSTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional, num_layers=n_layers)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x, status = self.rnn(x)
        x = self.fc(x)
        x = self.dropout(x)

        return x


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = len(UD_TAGS.vocab)
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

model = RNNPOSTagger(INPUT_DIM,
                     EMBEDDING_DIM,
                     HIDDEN_DIM,
                     OUTPUT_DIM,
                     N_LAYERS,
                     BIDIRECTIONAL,
                     DROPOUT)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print('The model has {} trainable parameters'.format(count_parameters(model)))

pretrained_embedding = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embedding)

PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

model = model.to(device)
criterion = criterion.to(device)

output = model(batch.text)
print(output)

epochs = 10
prediction = []
for epoch in range(1, epochs+1):
  for b, batch in enumerate(train_iter):
    x, y = batch.text.to(device), batch.udtags.to(device)
    output = model(x)

    cost = criterion(output.view(-1, output.shape[-1]), y.view(-1))
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    #prediction.append([tag for tag in UD_TAGS.vocab.itos[output.argmax(axis=1)]])

    print(prediction)

  print("EPOCH {:4d}/{}  COST {:.6f}  PREDICTION ".format(epoch, epochs, cost.item()))

corrects, total_loss = 0, 0
for batch in valid_iter:
  x, y  = batch.text.to(device), batch.udtags.to(device)
  output = model(x)
  cost = criterion(output, y)
  total_loss += cost.item()

  print()
  #corrects += sum(output.view(-1, output.shape[-1]).argmax(axis=1) == y)
