import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix

df = pd.read_csv("baseline.csv")
target = np.array(df['is_concert'])
df = df.drop(['is_concert','song_name_clean'],axis=1)
df_list = list(df.columns)
df = np.array(df)
print(df)

X_train, X_test, y_train, y_test = train_test_split(df,target, test_size = 0.25, random_state = 42)

model = LogisticRegression()
model.fit(X_train, y_train)
score = model.score(X_test,y_test)
print(score)




#scaling
data1 = df.filter(['stream_cnt_ratio'])
data2 = df.filter(['uniq_users_ratio'])
dataset1 = data1.values
dataset2 = data2.values

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data1 = scaler.fit_transform(dataset1)
scaled_data2 = scaler.fit_transform(dataset2)


for i in range(len(scaled_data1)):
    df.at[i,'scaled_stream_cnt'] = scaled_data1[i]

for i in range(len(scaled_data2)):
    df.at[i,'scaled_uniq_users_ratio'] = scaled_data2[i]


df.to_csv('baseline.csv', index=False)



