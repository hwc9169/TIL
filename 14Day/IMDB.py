# 필요한 도구 임포트
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext import data, datasets
import random

# 랜덤 시드 고정
SEED = 5
random.seed(SEED)
torch.manual_seed(SEED)

# 하이퍼파라미터
batch_size = 64
lr = 0.001
epochs = 10

# GPU설정
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('다음 기기로 학습함 : {}'.format(device))

# torchtext.data의 Field 클래스로 TEXT, LABEL 생성
TEXT = data.Field(sequential=True, batch_first=True, lower=True)
LABEL = data.Field(sequential=False, batch_first=True)

# torch.datasets로 데이터 로드
trainset, testset = datasets.IMDB.splits(TEXT, LABEL)

# 단어 집합 만들기
TEXT.build_vocab(trainset, min_freq=5)
LABEL.build_vocab(trainset)

vocab_size = len(TEXT.vocab)
n_classes = 2

# 검증 데이터 만들기
trainset, valset = trainset.split(split_ratio=0.8)

# 데이터 로더 만들기
train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (trainset, valset, testset),
    batch_size=batch_size,
    repeat=False,
    shuffle=True
)


class GRU(nn.Module):
    def __init__(self, n_layers, hidden_dim, vocab_size, embed_dim, n_classes, dropout=0.2):
        super(GRU, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=self.n_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, n_classes)

    def init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()

    def forward(self, x):
        x = self.embed(x)
        h_0 = self.init_state(batch_size=x.size(0))
        x, _ = self.gru(x, h_0)
        h_t = x[:, -1, :]
        self.dropout(h_t)
        logit = self.out(h_t)

        return logit


model = GRU(1, 256, vocab_size, 128, 2, dropout=0.5).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)


def train(model, optimizer, train_iter):
    model.train()
    for b, batch in enumerate(train_iter):
        x, y = batch.text.to(device), batch.label.to(device)
        y.data.sub_(1)
        logit = model(x)
        cost = F.cross_entropy(logit, y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()


def evaluate(model, val_iter):
    model.eval()
    corrects, total_loss = 0, 0
    for batch in val_iter:
        x, y = batch.text.to(device), batch.label.to(device)
        y.data.sub_(1)
        logit = model(x)
        cost = F.cross_entropy(logit, y, reduction='sum')
        total_loss += cost.item()
        corrects += (logit.argmax(axis=1).view(-1).data == y.data).sum()

    size = len(val_iter.dataset)
    avg_accuracy = 100.0 * corrects / size
    avg_loss = total_loss / size

    return avg_loss, avg_accuracy


best_val_loss = None
for epoch in range(1, epochs + 1):
    train(model, optimizer, train_iter)
    val_loss, val_accuracy = evaluate(model, val_iter)

    print('EPOCH {:4d}/{} Loss : {:.6f} accuracy : {:.4f}'.format(epoch, epochs, val_loss, val_accuracy))

    if not best_val_loss or val_loss < best_val_loss:
        if not os.path.isdir('snapshot'):
            os.makedirs('snapshot')
        torch.save(model.state_dict(), './snapshot/txtclassification.pt')
        best_val_loss = val_loss
