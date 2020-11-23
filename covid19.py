# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 22:40:21 2020

@author: baziz.M
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importation des données 

# covid_data_fr= pd.read_csv('https://www.data.gouv.fr/fr/datasets/r/d3a98a30-893f-47f7-96c5-2f4bcaaa0d71')

covid_data_fr= pd.read_csv('https://www.data.gouv.fr/en/datasets/r/381a9472-ce83-407d-9a64-1b8c23af83df')

covid_data_fr=covid_data_fr.iloc[75:242,]

# X= covid_data_fr.iloc[:,[0,1,3,4]].values
training_set=covid_data_fr.iloc[:,2:3].values

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 167):
    X_train.append(training_set[i-60:i, 0])
    y_train.append(training_set[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 30, return_sequences = True, 
                   input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

# Getting the predicted R in november
dataset_test= pd.DataFrame(np.zeros((20,5)))
dataset_total = pd.concat((covid_data_fr['R'], dataset_test[2]), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values

inputs = inputs.reshape(-1,1)

X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_R = regressor.predict(X_test)


# Visualising the results

plt.plot(predicted_R, color = 'blue', label = 'predicted_R')
plt.title('Prediction de la reproduction du covid 19 en France')
plt.xlabel('Time')
plt.ylabel('R effectif')
plt.legend()
plt.show()


