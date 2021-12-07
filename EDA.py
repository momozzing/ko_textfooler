import pandas as pd
from tqdm import tqdm
from koeda import EDA
from koeda import EDA
from koeda import SynonymReplacement

train_data = pd.read_csv("data/ratings_train.txt", delimiter="\t")

sample_data = train_data

# sample_data = train_data[:100]
# eda = EDA(
#     morpheme_analyzer="Mecab", alpha_sr=0.3, 
#     # alpha_ri=0.0, alpha_rs=0.0, prob_rd=0.0
# )

augmenter = SynonymReplacement(morpheme_analyzer="Mecab", 
              stopword= True
            )


text_list = []
question_list = []
all_list = []

for idx in tqdm(range(len(sample_data))):
    text, labels = sample_data["document"][idx], sample_data["label"][idx]

    text_result = augmenter(text, 0.3
    # , repetition=2
    )

    all_list.append([text_result] + [labels])

# print(all_list)

eda_data = pd.DataFrame(all_list, columns=['document', 'label'])
eda_data.to_csv('data/eda_data.txt', index=False, sep = '\t')
# text = "아버지가 방에 들어가신다"

# result = eda(text)
# print(result)
# # 아버지가 정실에 들어가신다

# result = eda(text, p=(0.9, 0.9, 0.9, 0.9), repetition=2)
# print(result)
# # ['아버지가 객실 아빠 안방 방에 정실 들어가신다', '아버지가 탈의실 방 휴게실 에 안방 탈의실 들어가신다']

################################ AEDA
# from koeda import AEDA

# text = "인문과학 또는 인문학(人文學, 영어: humanities)은 인간과 인간의 근원문제, 인간과 인간의 문화에 관심을 갖거나 인간의 가치와 인간만이 지닌 자기표현 능력을 바르게 이해하기 위한 과학적인 연구 방법에 관심을 갖는 학문 분야로서 인간의 사상과 문화에 관해 탐구하는 학문이다. 자연과학과 사회과학이 경험적인 접근을 주로 사용하는 것과는 달리, 분석적이고 비판적이며 사변적인 방법을 폭넓게 사용한다."

# result = aeda(text)
# print(result)
# # 어머니가 ! 집을 , 나가신다

# result = aeda(text, p=0.9, repetition=2)
# print(result)
# # ['! 어머니가 ! 집 ; 을 ? 나가신다', '. 어머니 ? 가 . 집 , 을 , 나가신다']
