import os
import re
import pickle
import numpy as np
from flask import Flask,render_template,url_for,request
from keras.preprocessing.sequence import pad_sequences

#Loading model pickel file
model = pickle.load(open('DL_model.pkl', 'rb'))
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

#Cleaning the text
def clean_text(text):
    clean_eng_chars = re.compile(r'[a-zA-Z]+|[0-9]')
    text = re.sub(clean_eng_chars, '', text)
    
    #remove underscore
    tashkil_underscore = re.compile(r'[\u0617-\u061A\u064B-\u0652]|_')
    text = re.sub(tashkil_underscore, ' ', text)

    #remove longation
    p_longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    text = re.sub(p_longation, subst, text)
    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')
    return text

#Tokenizing the text
from nltk.tokenize import RegexpTokenizer
def removePunctuation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    new_text = " ".join(tokenizer.tokenize(text))
    return clean_text(new_text)


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    text = request.values['text']
    text = removePunctuation(text)
    MAX_SEQUENCE_LENGTH = 250
    
    print(text)
    text = tokenizer.texts_to_sequences([text])
    text = pad_sequences(text, maxlen=MAX_SEQUENCE_LENGTH)
    
    if text == "":
        return render_template( 'home.html' ,pred ="  ")
    
    prediction = model.predict([text])[0]
    
    return render_template('home.html' , prediction = prediction)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
