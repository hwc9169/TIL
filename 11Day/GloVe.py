#GloVe는 카운트 기반과 예측 기반을 모두 사용하는 단어 임베딩 방법론이다.

import nltk
nltk.download('punkt')

import urllib.request
import zipfile
from lxml import etree
import re
from nltk.tokenize import word_tokenize, sent_tokenize

urllib.request.urlretrieve("https://raw.githubusercontent.com/GaoleMeng/RNN-and-FFNN-textClassification/master/ted_en-20160408.xml", filename="ted_en-20160408.xml")
targetXML=open('ted_en-20160408.xml', 'r', encoding='UTF8')
target_text = etree.parse(targetXML)
parse_text = '\n'.join(target_text.xpath('//content/text()'))
content_text = re.sub(r'\([^)]*\)', '', parse_text)
sent_text = sent_tokenize(content_text)

normalized_text =[]
for string in sent_text:
  tokens = re.sub(r"[^a-z0-9]+", " ",string.lower())
  normalized_text.append(tokens)

result = [word_tokenize(sentence) for sentence in normalized_text]

from glove import Corpus, Glove

corpus = Corpus()
corpus.fit(result,window=5)
#훈련 데이터로부터 GloVe에서 사용할 동시 등장 행렬 생성

glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=20, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)
#학습에 이용할 쓰레드의 개수는 4, 에포크는 20

model_result1 = glove.most_similar('man')
print(model_result1)
model_result2 = glove.most_similar('boy')
print(model_result2)