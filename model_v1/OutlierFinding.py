import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def merge(list1, list2):
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return merged_list


def Sort_Tuple(tup):
    # getting length of list of tuples
    lst = len(tup)
    for i in range(0, lst):
        for j in range(0, lst - i - 1):
            if (tup[j][1] < tup[j + 1][1]):
                temp = tup[j]
                tup[j] = tup[j + 1]
                tup[j + 1] = temp
    return tup


df = pd.read_csv('Copy_Train_Combined.csv')
names = df['song_name_clean']
z = np.abs(stats.zscore(df['_stream_cnt_ratio']))
merged_list = merge(names,z)
sorted = Sort_Tuple(merged_list)

print(sorted)

name_list = []

for i in range(len(z)):
    if sorted[i][1] > 3:
        name_list.append(sorted[i][0])

print(name_list)


#print(z)
#print(np.where(z > 3))



