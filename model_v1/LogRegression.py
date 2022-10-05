import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

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


df = pd.read_csv('Copy_Train_Combined.csv')
target = np.array(df['_is_concert'])
df = df.drop(['_stream_cnt','_uniq_users','_is_concert','song_name_clean'], axis=1)
df_list = list(df.columns)
df = np.array(df)




x_train, x_test, y_train, y_test = train_test_split(df, target, test_size = 0.25, random_state = 42)
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

logisticRegr = LogisticRegression(class_weight="balanced")
logisticRegr.fit(x_train, y_train)


predictions = logisticRegr.predict(x_test)


print(logisticRegr.predict_proba(x_test))
probabilities = logisticRegr.predict_proba(x_test)[:,1] #predicted probabilities for the positive label
probabilites = probabilities.tolist()
merged_list = merge(probabilites,y_test)



importance = logisticRegr.coef_[0]
#importance is a list so you can plot it.
feat_importances = pd.Series(importance)
feat_importances.nlargest(20).plot(kind='barh',title = 'Feature Importance')
#plt.show() #stream_cnt_ratio


sorted = Sort_Tuple(merged_list)
amount = int(input("Amount of songs:"))
print(Sort_Tuple(merged_list)[:amount])
tp = 0
for i in range(amount):
    if sorted[i][1] == 1:
        tp += 1
accuracy = tp / amount
print(accuracy)



#print(y_test)
#print(probabilities)


"""
# Grid search
grid_values = {'penalty': ['l1','l2'], 'C': [0.001,0.01,0.1,1,10,100,1000],
    'max_iter': [100, 1000,2500, 5000]}

logReg = LogisticRegression(solver='liblinear')
# Instantiate the grid search model
grid_search = GridSearchCV(logReg, param_grid = grid_values,cv = 3, verbose=True, n_jobs=-1)
grid_search.fit(x_train, y_train)
best_grid = grid_search.best_estimator_
print("Best Grid:")
print(best_grid) #LogisticRegression(C=100, max_iter=1000, penalty='l1', solver='liblinear')

predictions = grid_search.predict(x_test)


probabilities = grid_search.predict_proba(x_test)[:,1] #predicted probabilities for the positive label
probabilites = probabilities.tolist()
merged_list = merge(probabilites,y_test)
print(Sort_Tuple(merged_list))

sorted = Sort_Tuple(merged_list)

tp = 0
for i in range(20):
    if sorted[i][1] == 1:
        tp += 1
accuracy = tp / 20
print(accuracy) #55% accuracy
"""









