# -*- coding: utf-8 -*-
"""
Deep Learning Model

"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Flatten, Dropout, Bidirectional, Input, GRU
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import optimizers 
import re
import tensorflow as tf 
from nltk import word_tokenize
from google.colab import drive 
import os
from keras import layers
from sklearn.utils import class_weight 

#loading data form drive
drive.mount('/content/drive')
df_path = "/content/drive/MyDrive/AIM_Task/preprocessed_data.csv"
df = pd.read_csv(df_path)

#Copying dataframe
df_copy = df.copy()

#splitting the data to train and test
X = df_copy['text'].values
X_test = X[:20000]
X_train = X[20000:]

y = pd.get_dummies(df_copy['dialect']).values
y_test = y[:20000]
y_train = y[20000:]

#Tekonizing 
t = Tokenizer(oov_token='<UNK>')
t.fit_on_texts(X_train)

#Converting text to sequence
train_sequences = t.texts_to_sequences(X_train)
test_sequences = t.texts_to_sequences(X_test)

# Max number of words in each complaint
MAX_SEQUENCE_LENGTH = 1000

#Sequence Padding
X_train = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
X_test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

#Compute waughts for every class
class_weights = class_weight.compute_class_weight('balanced',
                                                   classes = np.unique(df_copy['dialect']),
                                                   y = df_copy['dialect'])

class_weights = dict(enumerate(class_weights))

EMBEDDING_DIM = 100

#Getting vocablary size
VOCAB_SIZE = len(t.word_index)

#Building the model
model = Sequential()
model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=X_train.shape[1]))

model.add(LSTM(units=128, return_sequences=True, return_state=False))
model.add(Dropout(0.2))

model.add(LSTM(units=64, return_sequences=True, return_state=False))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(18, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Training
batch_size = 128
history = model.fit(X_train, y_train, 
                    epochs=10, 
                    batch_size=batch_size, 
                    validation_data=(X_test,y_test), 
                    class_weight=class_weights,
                    verbose=1,
                    callbacks=[EarlyStopping(monitor='val_accuracy', mode='max', patience=3, min_delta=0.001)])

#model evaluating 
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(history)

#seve model
import pickle
filename = 'DL_model.pkl'
pickle.dump(model, open(filename, 'wb'))

#save tekonizer
filename = 'tokenizer.pkl'
pickle.dump(t, open(filename, 'wb'))