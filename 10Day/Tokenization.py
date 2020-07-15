en_text = 'A Dog Run back corner near spare bedrooms'

#nltk
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')
print(word_tokenize(en_text))

# spacy
import spacy
spacy_en = spacy.load('en')

def tokenize(en_text):
    return [tok.text for tok in spacy_en.tokenizer(en_text)]

print(tokenize(en_text))

#뛰어쓰기 토큰화
print(en_text.split())

#형태소 토큰화
