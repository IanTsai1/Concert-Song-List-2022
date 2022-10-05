import numpy as np
import pandas as pd
import os
import warnings
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Dense, Dropout, Input, MaxPooling2D
import matplotlib.pyplot as plt


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

df = pd.read_csv('Combined_Data.csv')
target = np.array(df['_is_concert'])
df = df.drop(["_is_concert","uu.other","pc.other","avg.other",
                             "uu.core","pc.core","avg.core","r","u","a","is_concert"], axis=1)
df_list = list(df.columns)
df = np.array(df)


x_train, x_test, y_train, y_test = train_test_split(df, target, test_size = 0.25, random_state = 42)

inputs = keras.Input(shape=(8,)) # number of dimension(features)
x = layers.Dense(64, activation="relu")(inputs) #layers = drawing an arrow from input to layer
x = Dropout(0.5)(x)
x = layers.Dense(64, activation="relu")(x)
x = Dropout(0.5)(x)
outputs = layers.Dense(10)(x)

model = keras.Model(inputs=inputs, outputs=outputs)
print(model.summary())


model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)

probabilities = model.predict_on_batch(x_test)[:,1]
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
print(accuracy) #90%


"""
test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
"""

