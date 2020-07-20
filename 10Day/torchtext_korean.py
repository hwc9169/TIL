import urllib.request
import pandas as pd

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

train_df = pd.read_table('ratings_train.txt')
test_df = pd.read_table('ratings_test.txt')
train_df.head()
print('훈련 데이터 샘플의 개수 : {}'.format(len(train_df)))
print('테스트 데이터 샘플의 개수 : {}'.format(len(test_df)))

from torchtext import data
from konlpy.tag import Mecab

#필드 정의
tokenizer = Mecab()

ID = data.Field(sequential = False,
                use_vocab = False)

TEXT = data.Field(sequential = True,
                  use_vocab = True,
                  tokenize = tokenizer.morphs,
                  lower = True,
                  batch_first = True,
                  fix_length = 20)

LABEL = data.Field(sequential = False,
                   use_vocab = False,
                   is_traget = True)

#데이터셋 생성
from torchtext.data import TabularDataset
train_data, test_data = TabularDataset.splits(
    path='.', train='ratings_train.txt', test='ratings_test.txt', format='tsv',
    fields=[('id',ID), ('text',TEXT), ('label', LABEL)], skip_header=True
)

print(vars(train_data[0]))

#단어 집합 생성
TEXT.build_vocab(train_data, min_freq=10, max_size=10000)
print('단어 집합의 크기 : {}'.format(len(TEXT.vocab)))
print(TEXT.vocab.stoi)

#데이터로더 생성
from torchtext.data import Iterator
batch_size = 5
train_loader = Iterator(dataset=train_data, batch_size = batch_size)
test_loader = Iterator(dataset=test_data, batch_size = batch_size)

print('훈련 데이터의 미니 배치 수 : {}'.format(len(train_loader)))
print('테스트 데이터의 미니 배치 수 : {}'.format(len(test_loader)))

batch = next(iter(train_loader))
print(batch.text)



