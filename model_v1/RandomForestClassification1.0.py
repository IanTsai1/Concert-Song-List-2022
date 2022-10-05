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



df = pd.read_csv('Train_Combined.csv',parse_dates=[1])
data = pd.read_csv('Train_Combined.csv',parse_dates=[1])

target = np.array(df['is_concert'])
df = df.drop(['is_concert','song_name_clean'], axis=1)
df_list = list(df.columns)
df = np.array(df)



train_data, test_data, train_target, test_target = train_test_split(df, target, test_size = 0.25, random_state = 42)


#baseline ZeroR method
count = 0
for i in range(len(data)-1):
    x = data.iloc[i]['is_concert']
    if x == False:
        count +=1
accuracy = count / len(data)
#print(f'Baseline accuracy:{accuracy} %')


#train
model = RandomForestClassifier(random_state=42) #changing amount of trees
model.fit(train_data,train_target)
#check if model is overtraining or not
print(model.score(train_data,train_target))
print(model.score(test_data,test_target))




test_predicted = model.predict(test_data)
cm = confusion_matrix(test_target,test_predicted) #actual vs predicted
print(cm)

importances = model.feature_importances_
sorted_idx = model.feature_importances_.argsort()
plt.barh(df_list,importances)
#plt.show()

#print(pd.DataFrame(model.data_importances_, index=train_data.columns,columns=['importance']).sort_values('importance', ascending=False))



#visualizing decision tree
rf_small = RandomForestClassifier(n_estimators=10, max_depth = 3)
rf_small.fit(train_data, train_target)
# Extract the small tree
tree_small = rf_small.estimators_[5]
# Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = df_list, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
#graph.write_png('small_tree.png');

#feature selection
#rf_most_important = RandomForestClassifier(n_estimators= 35)
#important_indices = [df_list.index('stream_cnt_ratio'),df_list.index('days_diff_song_concert'),df_list.index('uniq_users_ratio')]
#train_important = train_data[:, important_indices]
#test_important = test_data[:, important_indices]

#rf_most_important.fit(train_important, train_target)
#print(rf_most_important.score(train_important,train_target))
#print(rf_most_important.score(test_important,test_target))


#print(model.score(test_data,test_target))

#rf parameters; most important are n_estimators and max_features
#print('Parameters currently in use:\n')
#pprint(model.get_params())



















