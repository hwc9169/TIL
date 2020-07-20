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
for line in result[0:3]:
  print(line)


from gensim.models import Word2Vec
model = Word2Vec(sentences=result, size = 100, window = 5, min_count = 5, workers=4, sg = 0)
#size = 워드 벡터 특징 값으로 임베딩 된 벡터의 차원
#window = 컨텍스트 윈도우 크기
#min_count = 단어 최소 빈도 수 제한(빈도가 적은 단어들은 학습하지 않는다)
#workers = 프로세스 수
# sg = 0 : CBOW, 1 : Skip-gram

model_result = model.wv.most_similar("man")

model.wv.save_word2vec_format('./eng_w2v')
print(model_result)