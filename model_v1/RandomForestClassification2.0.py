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



df = pd.read_csv('Copy_Train_Combined.csv')
target = np.array(df['smote_is_concert'])
df = df.drop(['is_concert','stream_cnt_ratio','days_diff_song_concert','uniq_users_ratio','smote_is_concert'], axis=1)
df_list = list(df.columns)
df = np.array(df)



train_data, test_data, train_target, test_target = train_test_split(df, target, test_size = 0.25, random_state = 42)



# Number of trees in random forest
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

"""
rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(train_data, train_target)

#best_random = rf_random.best_params_
#print(best_random)
print(rf_random.score(test_data,test_target))

test_predicted = rf_random.predict(test_data)
cm = confusion_matrix(test_target,test_predicted) #actual vs predicted
print(cm)

"""

"""
test = RandomForestClassifier(n_estimators=600, min_samples_split=5, min_samples_leaf = 1, max_features='auto', max_depth=80, bootstrap=True)
test.fit(train_data,train_target)
print(test.score(test_data,test_target))

"""


#accuracy_diff = (test.score(test_data,test_target) - model.score(test_data,test_target))*100
#print(f"Accuracy Difference:{accuracy_diff}%")

# Create the parameter grid based on the results of random search
param_grid = {
    'bootstrap': [True],
    'max_depth': [75, 80, 85],
    'max_features': [2,3],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [4, 5, 6],
    'n_estimators': [500,600,700]
}

# Grid search
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                          cv = 3, n_jobs = -1, verbose = 2)
grid_search.fit(train_data, train_target)
best_grid = grid_search.best_estimator_
print(best_grid)
print(grid_search.score(test_data,test_target))





