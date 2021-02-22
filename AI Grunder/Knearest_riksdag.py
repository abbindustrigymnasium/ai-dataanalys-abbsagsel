import numpy as np
from sklearn import preprocessing, neighbors, model_selection
import pandas as pd

df = pd.read_csv('votering.csv')
print(list(df))

df.drop(['punkt'], 1, inplace=True)

df = df[["rost", "parti", "fodd", "kon", "intressent_id"]]
print(df.head(3))
input_labels=['kvinna','man']
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)
df["kon"]=encoder.transform(df["kon"])
for i, item in enumerate(encoder.classes_):
    print(item,"-->",i)