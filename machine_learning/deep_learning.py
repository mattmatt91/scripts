from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from helpers.helpers import Helpers as hp
import numpy as np

def deep_learing(features:pd.DataFrame, infos:pd.DataFrame):
    dataset = loadtxt('machine_learning\\test_data.csv', delimiter=',')
    # split into input (X) and output (y) variables
    X = dataset[:,0:8]
    y = dataset[:,8]
    print(X.shape, y.shape)
    print(type(X), type(y))
    
    sample_dict, numbers =  hp.sample_to_numbers(infos['sample'])
    X = features.to_numpy()
    y = np.array(numbers) # infos['sample']
    print(X.shape, y.shape)
    print(type(X), type(y))
    
    # define the keras model
    model = Sequential()
    model.add(Dense(12, input_shape=(8,), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    
    # fit the keras model on the dataset
    model.fit(X, y, epochs=150, batch_size=10, verbose=0)
    exit()
    # make class predictions with the model
    predictions = (model.predict(X) > 0.5).astype(int)
    
    # summarize the first 5 cases
    for i in range(5):
        print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))





