import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('Combined_Data.csv')
target = np.array(df['_is_concert'])
df = df.drop(["_is_concert","uu.other","pc.other","avg.other",
                             "uu.core","pc.core","avg.core","r","u","a","is_concert"], axis=1)
df_list = list(df.columns)
df = np.array(df)


x_train, x_test, y_train, y_test = train_test_split(df, target, test_size = 0.25, random_state = 42)

input_shape = (7,) #dimension of input data, 8 = # of neuron
batch_size = 128 #determines # of sample that run through model before refreshing
kernel_size = 3 #dimension of filter mask
filters = 64 #weight
dropout = 0.5 #random neurons are neglected, prevents overfitting

# utiliaing functional API to build cnn layers
inputs = Input(shape=(7,1,1))

y = Conv2D(filters=64, #hidden layer 1
         kernel_size=3,
         activation='relu',padding="same")(inputs)
#y = MaxPooling2D()(y)
y = Dropout(dropout)(y)

y = Conv2D(filters=64, #hidden layer 2
         kernel_size=3,
         activation='relu',padding="same")(y)
#y = MaxPooling2D()(y)
y = Dropout(dropout)(y)

y = Conv2D(filters=64, #hidden layer 3
     kernel_size=3,
     activation='relu',padding="same")(y)

# dropout regularization, prevents overfitting
y = Dropout(dropout)(y)

outputs = Dense(10,activation='softmax')(y) #can try 'sigmoid' as activation

 # model building by supplying inputs/outputs
model = Model(inputs=inputs, outputs=outputs)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train,
          y_train,
          validation_data=(x_test, y_test),
          epochs=20,
          batch_size=batch_size)
