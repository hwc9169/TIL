#1. 룩업 테이블 과정을 nn.Embedding()을 사용하지 않고 구현한 코드
import torch

train_data = 'you need to know how to code'
word_set = set(train_data.split())

vocab = {word : i+2 for i,word in enumerate(word_set)}
vocab['<unk>'] = 0
vocab['<pad>'] = 1
#print(vocab)

embedding_table = torch.FloatTensor([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.2, 0.9, 0.3],
    [0.1, 0.5, 0.7],
    [0.2, 0.1, 0.8],
    [0.4, 0.1, 0.1],
    [0.1, 0.8, 0.9],
    [0.6, 0.1, 0.1]])

sample ='you need to run'.split()
idxes=[]
for word in sample:
    try:
        idxes.append(vocab[word])
    except KeyError:
        idxes.append(vocab['<unk>'])
idxes = torch.LongTensor(idxes)

lookup_table = embedding_table[idxes,:]
#print(lookup_table)

#2. nn.Embedding()을 사용하여 룩업 테이블 만들기
import torch.nn as nn

train_data = 'you need to know how to code'
word_set = set(train_data.split())

vocab ={word : i+2 for i,word in enumerate(word_set)}
vocab['<unk>'] = 0
vocab['<pad>'] = 1

embedding_layer = nn.Embedding(num_embeddings= len(vocab), #임베딩 할 단어들의 개수
                               embedding_dim= 3,           #임베딩 할 벡터의 차원
                               padding_idx=1)              #패딩 토큰의 인덱스(option)

#print(embedding_layer.weight)

#3. 이미 훈련된 임베딩 벡터 사용
from torchtext import data, datasets

TEXT = data.Field(sequential=True,
                  batch_first=True,
                  lower=True)

LABEL = data.Field(sequential=False,
                   batch_first=True)

trainset, testset = datasets.IMDB.splits(TEXT,LABEL)
print(vars(trainset[0]))

#4. 사전 훈련된 Word2Vec을 초기 임베딩을 사용하기
from gensim.models import KeyedVectors
import torch
import torch.nn as nn
from torchtext.vocab import Vectors

word2vec_model = KeyedVectors.load_word2vec_format('eng_w2v')
print(word2vec_model['this']) # 'this'의 임베딩 벡터값 출력

vectors = Vectors(name="eng_w2v")
TEXT.build_vocab(trainset, vectors=vectors, min_freq=10, max_size=10000)
print(TEXT.vocab.stoi)

embedding_layer = nn.Embedding.from_pretrained(TEXT.vocab.vectors, freeze=False)
print(embedding_layer(torch.LongTensor([10])))

#5. 토치텍스트에서 제공하는 사전 훈련된 워드 임베딩
#1. fasttest.en.300d
#2. fasttext.simpple.300d
#3. glove.42B.300d
#4. glove.twitter.27B.25d
#5. glove.6B.300d <-- 우리가 사용할 데이터

from torchtext.vocab import GloVe
TEXT.build_vocab(trainset, vectors=GloVe(name='6B', dim=300), max_size=10000, min_freq=10)
LABEL.build_vocab(trainset)

embedding_layer = nn.Embedding.from_pretrained(TEXT.vocab.vectors, freeze=False)
embedding_layer(torch.LongTensor([10]))
