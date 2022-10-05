import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statistics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
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

df = pd.read_csv('Copy_Train_Combined.csv')
target = np.array(df['_is_concert'])
df = df.drop(['_stream_cnt','_uniq_users','_is_concert','song_name_clean'], axis=1)
df_list = list(df.columns)
df = np.array(df)

x_train, x_test, y_train, y_test = train_test_split(df, target, test_size = 0.25, random_state = 42)

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
rf = RandomForestClassifier(n_estimators=600, min_samples_split=5, min_samples_leaf=1, max_features= 'auto', max_depth= 80, bootstrap = True)
rf.fit(x_train,y_train)

"""
#randomized search
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(x_train, y_train)

print(rf_random.best_params_)
#{'n_estimators': 600, 'min_samples_split': 5,
# 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 80, 'bootstrap': True}
"""

probabilites = rf.predict_proba(x_test)[:,1]
probabilites = probabilites.tolist()
merged_list = merge(probabilites,y_test)
print(Sort_Tuple(merged_list))

sorted = Sort_Tuple(merged_list)

tp = 0
amount = int(input("Amount of songs:"))
for i in range(amount):
    if sorted[i][1] == 1:
        tp += 1
accuracy = tp / amount
print(accuracy)


