import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statistics
from sklearn.tree import export_graphviz
import pydot
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def merge(list1, list2):
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return merged_list


def Sort_Tuple(tup):
    # getting length of list of tuples
    lst = len(tup)
    for i in range(0, lst):
        for j in range(0, lst - i - 1):
            if (tup[j][0] < tup[j + 1][0]):
                temp = tup[j]
                tup[j] = tup[j + 1]
                tup[j + 1] = temp
    return tup
"""
df = pd.read_csv('Copy_Train_Combined.csv')
target = np.array(df['_is_concert'])
df = df.drop(['_stream_cnt','_uniq_users','_is_concert','song_name_clean'], axis=1)
"""
df = pd.read_csv('Combined_Data.csv')
target = np.array(df['_is_concert'])
df = df.drop(["_is_concert","uu.other","pc.other","avg.other",
                             "uu.core","pc.core","avg.core","r","u","a","is_concert"], axis=1)
df_list = list(df.columns)
df = np.array(df)

x_train, x_test, y_train, y_test = train_test_split(df, target, test_size = 0.25, random_state = 42)


rf = RandomForestClassifier(random_state=42)
rf.fit(x_train,y_train)


probabilities = rf.predict_proba(x_test)[:,1]
probabilites = probabilities.tolist()

merged_list = merge(probabilites,y_test)


sorted = Sort_Tuple(merged_list)

tp = 0
amount = int(input("Amount of songs:"))
print(Sort_Tuple(merged_list)[:amount])
for i in range(amount):
    if sorted[i][1] == 1:
        tp += 1
accuracy = tp / amount
print(accuracy) #95% = 20 songs, 96.5% = 30 songs

importances = rf.feature_importances_
sorted_idx = rf.feature_importances_.argsort()
plt.barh(df_list,importances)
plt.show()





