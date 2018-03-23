# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 16:34:39 2018

@author: Karthikeyan
"""

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.utils import np_utils

#loading data set
df = pd.read_csv('E:\\Internship\\Sentiment_analysis\\train.csv',delimiter=',',encoding='utf-8')
df.head()

df.info
df.dropna(thresh = 1)

#categories plot
sns.countplot(df.categories,)
plt.xlabel('Label')
plt.title('Categories')

#Independent and target variabes
X = df.converse.astype(str)
Y = df.categories
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = np_utils.to_categorical(Y)

#test train split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)

##Process
max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
sequences = tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

#building rnn model
def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(6,name='out_layer')(layer)
    layer = Activation('softmax')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

model = RNN()
model.summary()
model.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

model.fit(sequences_matrix,Y_train,batch_size=128,epochs=10,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
predict = model.predict(test_sequences_matrix)
predict = np.argmax(predict,axis=1)
accr = model.evaluate(test_sequences_matrix,Y_test)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

test_new = pd.read_csv('E:\\Internship\\Sentiment_analysis\\test.csv', encoding = 'utf-8')

test_new = test_new.converse.astype(str)

tok = Tokenizer(num_words=max_words)
sequences = tok.fit_on_texts(test_new)
sequences = tok.texts_to_sequences(test_new)
sequences_matrix1 = sequence.pad_sequences(sequences,maxlen=max_len)


pred = model.predict(sequences_matrix1)
pred = np.argmax(pred,axis=1)







