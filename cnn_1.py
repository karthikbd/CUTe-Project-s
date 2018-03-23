# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 20:12:32 2018

@author: Karthikeyan
"""
###Cnn model 2

import pandas as pd
import numpy as np


DATA_FILE = 'E:\\Internship\\Sentiment_analysis\\corpus.csv'
df = pd.read_csv(DATA_FILE,encoding='utf-8')
print(df.head())

Category = df.categories
Conversation = df.corpus.astype(str)


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

num_max = 1000
# preprocess
le = LabelEncoder()
Category = le.fit_transform(Category)
Category = np_utils.to_categorical(Category)
tok = Tokenizer(num_words=num_max)
tok.fit_on_texts(Conversation)
mat_texts = tok.texts_to_matrix(Conversation,mode='count')
print(Category[:5])
print(mat_texts[:5])
print(Category.shape,mat_texts.shape)

# for cnn preproces
max_len = 200
cnn_texts_seq = tok.texts_to_sequences(Conversation)
print(cnn_texts_seq[0])
cnn_texts_mat = sequence.pad_sequences(cnn_texts_seq,maxlen=max_len)
print(cnn_texts_mat[0])
print(cnn_texts_mat.shape)

#cnn model 2
def get_cnn_model_v2():    # added filter
    model = Sequential()
    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    # 1000 is num_max
    model.add(Embedding(1000,
                        20,
                        input_length=max_len))
    model.add(Dropout(0.2))
    model.add(Conv1D(256, #!!!!!!!!!!!!!!!!!!!
                     3,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(256))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(6))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy',metrics.categorical_accuracy])
    return model

def check_model(model,x,y):
    model.fit(x,y,batch_size=32,epochs=25,verbose=1,validation_split=0.2)

m = get_cnn_model_v2()
check_model(m,cnn_texts_mat,Category)

test_new = pd.read_csv('E:\\Internship\\Sentiment_analysis\\corpus1.csv', encoding = 'utf-8')

test_new = test_new.corpus.astype(str)

tok = Tokenizer(num_words=num_max)
sequences = tok.fit_on_texts(test_new)
sequences = tok.texts_to_sequences(test_new)
sequences_matrix1 = sequence.pad_sequences(sequences,maxlen=max_len)

cnn_texts_seq = tok.texts_to_sequences(test_new)
print(cnn_texts_seq[0])
cnn_texts_mat = sequence.pad_sequences(cnn_texts_seq,maxlen=max_len)
print(cnn_texts_mat[0])
print(cnn_texts_mat.shape)

pred = m.predict(cnn_texts_mat)
pred = np.argmax(pred,axis=1)

