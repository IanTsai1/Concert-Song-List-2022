import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("baseline.csv")

#scaling
data1 = df.filter(['stream_cnt_ratio'])
data2 = df.filter(['uniq_users_ratio'])
dataset1 = data1.values
dataset2 = data2.values

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset1)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data1 = scaler.fit_transform(dataset1)
scaled_data2 = scaler.fit_transform(dataset2)


for i in range(df.shape[0]):
    df.loc[i,'scaled_stream_cnt'] = scaled_data1[i]

for i in range(df.shape[0]):
    df.loc[i,'scaled_uniq_users'] = scaled_data2[i]





#calculating sum
df['sum'] = df.loc[: , ['scaled_stream_cnt', 'scaled_uniq_users']].sum(axis=1)

#sort sum
#df.sort_values(by=['sum'])



amount = int(input("Type in amount of songs:"))

tp_count = 0
tn_count = 0
bin = []
for i in range(amount):
    bin.append(df.loc[i, 'is_concert'])
    if df.loc[i,'is_concert'] == True:
        tp_count+=1

accuracy = tp_count / amount
print(accuracy)
print(bin)



df.to_csv('baseline.csv', index=False)

#Accuracy = 10% if I chose 20 songs in list

