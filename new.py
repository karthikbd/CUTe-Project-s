# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 13:01:59 2018

@author: Karthikeyan
"""
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Load data set
DATA_FILE = 'E:\\Internship\\Sentiment_analysis\\train.csv'
df = pd.read_csv(DATA_FILE, quoting = 3)
df = df.replace(np.nan,'clinical list changes', regex = True)
print(df.head())
df.isnull().sum()
#Category = df.categories
Conversation = df.converse.astype(str)
type(df.converse)

#removing the spaces & punctuation -- #stemming
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

corpus = []
for i in range(0, 45824):
    Conversation = df.converse.astype(str)
    Conversation = re.sub('[^a-zA-Z]', ' ', Conversation[i])
    Conversation = Conversation.lower()
    Conversation = Conversation.split()
    Conversation = [word for word in Conversation if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    Conversation = [ps.stem(word) for word in Conversation if not word in set(stopwords.words('english'))]
    Conversation = ' '.join(Conversation)
    corpus.append(Conversation)  

'''corpus1 = 'E:\\Internship\\Sentiment_analysis\\corpus.csv'
corpus = pd.read_csv(corpus1, quoting = 3) '''



#creating bag of word model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 20000)
X = cv.fit_transform(corpus).toarray()

y = df.categories[:41943,]
le = LabelEncoder()
y = le.fit_transform(y)
y = np_utils.to_categorical(y)

##Model building

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import time
from keras import metrics

def get_simple_model():
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(20000,)))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(6, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    print('compile done')
    return model

def check_model(model,x,y):
    model.fit(x,y,batch_size=32,epochs=10,verbose=1,validation_split=0.2)

m = get_simple_model()
check_model(m,X,y)


test_new = pd.read_csv('E:\\Internship\\Sentiment_analysis\\test.csv', encoding = 'utf-8')
test_new = test_new.replace(np.nan,'clinical list changes', regex = True)
test_new = test_new.converse.astype(str)
test_new.isnull().sum()

corpus1 = []
for i in range(0, 11456):
    Conversation1 = test_new
    Conversation1 = re.sub('[^a-zA-Z]', ' ', Conversation1[i])
    Conversation1 = Conversation1.lower()
    Conversation1 = Conversation1.split()
    Conversation1 = [word for word in Conversation1 if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    Conversation1 = [ps.stem(word) for word in Conversation1 if not word in set(stopwords.words('english'))]
    Conversation1 = ' '.join(Conversation1)
    corpus1.append(Conversation1)  


from sklearn.feature_extraction.text import CountVectorizer
cv1 = CountVectorizer(max_features = 1500)
test = cv1.fit_transform(corpus1).toarray()

pred = m.predict(test)
pred = argmax(pred, axis = 1)



































