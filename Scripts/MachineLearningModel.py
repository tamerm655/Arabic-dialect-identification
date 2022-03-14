# -*- coding: utf-8 -*-
"""
Machine Learning Models

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
import os
from google.colab import drive
from sklearn.utils import class_weight 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

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

# y = pd.get_dummies(df_copy['dialect']).values
y = df_copy['dialect']
y_test = y[:20000]
y_train = y[20000:]

#Tekonizing 
t = Tokenizer(oov_token='<UNK>')
t.fit_on_texts(X_train)

#Converting text to sequence
train_sequences = t.texts_to_sequences(X_train)
test_sequences = t.texts_to_sequences(X_test)

# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 1000

#Sequence Padding
X_train = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
X_test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

#Compute waughts for every class
class_weights = class_weight.compute_class_weight('balanced',
                                                   classes = np.unique(df_copy['dialect']),
                                                   y = df_copy['dialect'])

class_weights = dict(enumerate(class_weights))

#Training
def train_model(model, data, targets):
    text_clf = Pipeline([
    # ('vect', CountVectorizer()),
    # ('tfidf', TfidfTransformer()),
    ('clf', model),
    ])
    text_clf.fit(data, targets)
    return text_clf
def get_accuracy(trained_model,X, y):
    predicted = trained_model.predict(X)
    accuracy = np.mean(predicted == y)
    return accuracy

#Multinomial naive bayes model
from sklearn.naive_bayes import MultinomialNB
trained_clf_multinomial_nb = train_model(MultinomialNB(), X_train, y_train)
accuracy = get_accuracy(trained_clf_multinomial_nb,X_test, y_test)
print(accuracy)

#Logistic Regression Model
from sklearn.linear_model import LogisticRegression
trained_clf_LogisticRegression = train_model(LogisticRegression(class_weight=class_weights), X_train, y_train)
accuracy = get_accuracy(trained_clf_LogisticRegression,X_test, y_test)
print(accuracy)

# SGDClassifier Model
from sklearn.linear_model import SGDClassifier
trained_clf_SGDClassifier = train_model(SGDClassifier(class_weight=class_weights), X_train, y_train)
accuracy = get_accuracy(trained_clf_SGDClassifier,X_test, y_test)
print(accuracy)

#Linear SVM Model
from sklearn.svm import LinearSVC
trained_clf_LinearSVC = train_model(LinearSVC(), X_train, y_train)
accuracy = get_accuracy(trained_clf_LinearSVC,X_test, y_test)
print(accuracy)