import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer

sns.set_style("darkgrid")
sns.set_context("poster")
plt.rcParams["figure.figsize"] = [8,6]

#import file
spam_dataset = pd.read_csv("Combined_Data.csv")
spam_dataset = spam_dataset[["is_concert","uu.other","pc.other","avg.other",
                             "uu.core","pc.core","avg.core","r","u","a"]]
spam_dataset.head()

print("Before Upsampling:")
print(spam_dataset["is_concert"].value_counts())


X = spam_dataset[["uu.other","pc.other","avg.other",
                             "uu.core","pc.core","avg.core","r","u","a"]]


#extract label set
y = spam_dataset[['is_concert']]

#Use SMOTE for upsampling
su = SMOTE(random_state=42)
X_su, y_su = su.fit_resample(X, y)

print("After Upsampling:")
print(y_su["is_concert"].value_counts())

"""
y_su.groupby('v1').size().plot(kind='pie',
                                       y = "v1",
                                       label = "Type",
                                       autopct='%1.1f%%')
"""


for i in range(len(X_su["uu.other"])):
    spam_dataset.at[i, '_is_concert'] = y_su["is_concert"][i]

for i in range(len(X_su["uu.other"])):
    spam_dataset.at[i,'_uu.other'] = X_su["uu.other"][i]

for i in range(len(X_su["uu.other"])):
    spam_dataset.at[i,'_pc.other'] = X_su["pc.other"][i]

for i in range(len(X_su["uu.other"])):
    spam_dataset.at[i,'_avg.other'] = X_su["avg.other"][i]

for i in range(len(X_su["uu.other"])):
    spam_dataset.at[i,'_uu.core'] = X_su["uu.core"][i]

for i in range(len(X_su["uu.other"])):
    spam_dataset.at[i,'_pc.core'] = X_su["pc.core"][i]

for i in range(len(X_su["uu.other"])):
    spam_dataset.at[i,'_r'] = X_su["r"][i]

for i in range(len(X_su["uu.other"])):
    spam_dataset.at[i,'_u'] = X_su["u"][i]

for i in range(len(X_su["uu.other"])):
    spam_dataset.at[i,'_a'] = X_su["a"][i]



spam_dataset.to_csv('Combined_Data.csv', index=False)
