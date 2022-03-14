# -*- coding: utf-8 -*-
"""
Data Preprocessing

"""

from google.colab import drive 
import os
import pandas as pd
import json
import numpy as np
import re
from nltk.tokenize import RegexpTokenizer

#loading data form drive
drive.mount('/content/drive')
labels_path = "/content/drive/MyDrive/AIM_Task/dialect_dataset.csv"
texts_path = "/content/drive/MyDrive/AIM_Task/requested_data.csv"

labels_df = pd.read_csv(labels_path)
text_df = pd.read_csv(texts_path, lineterminator='\n')

#Joining tweets dataframe with classes dataframe
df = labels_df.join(text_df.set_index('id'), on='id')

df_copy = df.copy()

#Dropping ID column
df_copy = df_copy.drop('id', axis=1)

#Shuffeling data
df_copy = df_copy.sample(frac=1).reset_index(drop=True)


def clean_text(text):
    #Remove English characters and numbers
    clean_eng_chars = re.compile(r'[a-zA-Z]+|[0-9]')
    text = re.sub(clean_eng_chars, '', text)
    
    #Remove tashkil and underscore
    tashkil_underscore = re.compile(r'[\u0617-\u061A\u064B-\u0652]|_')
    text = re.sub(tashkil_underscore, ' ', text)

    #Remove longation
    p_longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    text = re.sub(p_longation, subst, text)
    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')
    return text

#Remove Punctuation
def removePunctuation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    new_text = " ".join(tokenizer.tokenize(text))
    return clean_text(new_text)

df_copy['text'] = df_copy['text'].apply(removePunctuation)

#Converting Classes names to numbers
dict = {'TN':0, 'YE':1, 'MA':2, 'SD':3, 'IQ':4, 'DZ':5, 'SY':6, 'OM':7,
        'BH':8, 'AE':9, 'SA':10, 'LB':11, 'JO':12, 'QA':13, 'LY':14, 'KW':15, 'PL':16, 'EG':17}

df_copy['dialect'] = df_copy['dialect'].map(dict)

df_copy.to_csv('/content/drive/MyDrive/AIM_Task/preprocessed_data.csv', index=False)

