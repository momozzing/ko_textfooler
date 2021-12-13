# import pandas as pd

# train_data = pd.read_csv("data/ratings_train.txt", delimiter="\t")
# train_data = train_data.dropna(axis=0)



# print(train_data)
# print(train_data.isnull().sum())

from kobert_tokenizer import KoBERTTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification


tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
autotokenizer = AutoTokenizer.from_pretrained('skt/kobert-base-v1')

print(tokenizer.encode("한국어 모델을 공유합니다."))
print(autotokenizer.encode("한국어 모델을 공유합니다."))