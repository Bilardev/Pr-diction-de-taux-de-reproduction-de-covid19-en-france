# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 22:40:21 2020

@author: baziz.M
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importation des données 

covid_data_fr= pd.read_csv('https://www.data.gouv.fr/en/datasets/r/381a9472-ce83-407d-9a64-1b8c23af83df')

covid_data_fr=covid_data_fr.iloc[75:242,]

training_set=covid_data_fr.iloc[:,2:3].values

# Creation d'une structure de donnée avec 60 timesteps et 1 sortie
X_train = []
y_train = []
for i in range(60, 167):
    X_train.append(training_set[i-60:i, 0])
    y_train.append(training_set[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Construction du RNN

# Importation de librairies et packages Keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialisation du model RNN
model = Sequential()

# Ajout de  couches LSTM et Dropout regularisation

model.add(LSTM(units = 30, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))


model.add(LSTM(units = 30, return_sequences = True))
model.add(Dropout(0.2))


model.add(LSTM(units = 30, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 30))
model.add(Dropout(0.2))

# ajout de couche de sortie
model.add(Dense(units = 1))


# Compilation du  RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs = 100, batch_size = 16)

# prédiction
dataset_test= pd.DataFrame(np.zeros((60,5)))
dataset_total = pd.concat((covid_data_fr['R'], dataset_test[2]), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values

inputs = inputs.reshape(-1,1)

X_test = []
for i in range(60, 120):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_R = model.predict(X_test)


# Visualisation des resultats

plt.plot(predicted_R, color = 'blue', label = 'predicted_R')
plt.title('Prédiction du taux de reproduction du covid 19 en France (15/11-14/01)')
plt.xlabel('Time')
plt.ylabel('R effectif')
plt.legend()
plt.show()


