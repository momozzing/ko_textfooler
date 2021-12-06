import pandas as pd

train_data = pd.read_csv("data/ratings_train.txt", delimiter="\t")
train_data = train_data.dropna(axis=0)



print(train_data)
print(train_data.isnull().sum())