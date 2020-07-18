import urllib.request
import pandas as pd

#1. data download
urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", filename="IMDb_Reviews.csv")
df = pd.read_csv('IMDb_Reviews.csv', encoding='latin1')
print(df.head())
print('전체 샘플의 수 : {}'.format(len(df)))

train_df = df[:25000]
test_df = df[25000:]

train_df.to_csv("./data/train_data.csv", index=False)
test_df.to_csv("./data/test_data.csv", index=False)

from torchtext import data

#2. field definition
text = data.Field(sequential=True,  #시퀀스 데이터여부
                  use_vocab=True,   #단어 집합을 만들 것인지 여부
                  tokenize=str.split, #어떤 토큰화 함수를 사용할 것인지 설정
                  lower = True,     #모두 소문자화
                  batch_first=True, #미니 배치 차원을 맨 앞으로 하여 데이터를 불러옴
                  fix_length=150)    # 최대 혀용 길이. 이 길이에 맞춰 패딩 작업이 진행된다.

label = data.Field(sequential=False,
                   use_vocab=False,
                   batch_first=False,
                   is_target=True)  #레이블 데이터 여부
#데이터셋 만들기
from torchtext.data import TabularDataset

train_data, test_data = TabularDataset.splits(
    path='./data', train='train_data.csv', test='test_data.csv', format='csv',
    fields=[('text',text), ('label',label)], skip_header = True
)
# path : 파일 경로
# format : 데이터 포맷
# fields : 위에서 정의한 필드 지정. 첫번째 원소는 데이터 셋 내에서 해당 필드의 호칭, 두번째 원소는 필드
# skip_header : 데이터의 첫줄을 무시

print(vars(train_data[0]))
print(train_data.fields.items())

#4. Vocabulary
#각 단어에 대한 정수 인코딩 작업이 필요하다. 이 전처리를 위해서는 우선 단어 집합을 만들어야한다.

text.build_vocab(train_data,min_freq=10,max_size=10000) #정의한 필드에 .build_vocab()을 사용하면 단어 집합을 생성한다.
#min_freq : 최소 등장 빈도 조건
#max_size : 단어 집합의 최대 크기 지정

print("단어 집합의 크기 : {}".format(len(text.vocab)))
print(text.vocab.stoi) #생성된 단어 집합 내의 단어들은 .stoi를 통해 확인 가능하다.

#5. 데이터로더 만들기
from torchtext.data import Iterator

batch_size = 5
train_loader = Iterator(dataset=train_data, batch_size = batch_size)
test_loader = Iterator(dataset=test_data, batch_size = batch_size)

batch = next(iter(train_loader))
print(batch.text)