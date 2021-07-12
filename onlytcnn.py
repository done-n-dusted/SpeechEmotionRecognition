import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

import pandas as pd
import numpy as np
from sklearn import preprocessing
from STFE.TextCNN import *
from tqdm import tqdm

# key = {
#     'anger' : [1, 0],
#     'sadness' : [0, 1]
# }

class_names = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

key = {
    'anger'     : [1, 0, 0, 0, 0, 0, 0],
    'disgust'   : [0, 1, 0, 0, 0, 0, 0],
    'fear'      : [0, 0, 1, 0, 0, 0, 0],
    'joy'       : [0, 0, 0, 1, 0, 0, 0],
    'neutral'   : [0, 0, 0, 0, 1, 0, 0],
    'sadness'   : [0, 0, 0, 0, 0, 1, 0],
    'surprise'  : [0, 0, 0, 0, 0, 0, 1]
}

cws = {
    0: 4.0,
    1: 15.0,
    2: 15.0,
    3: 3.0,
    4: 1.0,
    5: 6.0,
    6: 3.0
}
# [4.0, 15.0, 15.0, 3.0, 1.0, 6.0, 3.0].
train_csv = pd.read_csv('text_csv/train_sent_emo.csv')
test_csv = pd.read_csv('text_csv/test_sent_emo.csv')
dev_csv = pd.read_csv('text_csv/dev_sent_emo.csv')

# train_csv = train_csv[(train_csv['Emotion'] == 'anger') | (train_csv['Emotion'] == 'sadness')]
# test_csv = test_csv[(test_csv['Emotion'] == 'anger') | (test_csv['Emotion'] == 'sadness')]
# dev_csv = dev_csv[(dev_csv['Emotion'] == 'anger') | (dev_csv['Emotion'] == 'sadness')]

tokenizer = Tokenizer(num_words = 5000)
tokenizer.fit_on_texts(train_csv['Utterance'])
vocab_size = len(tokenizer.word_index) + 1
maxlen = 200

def transform_docs(docs, tokenizer = tokenizer, maxlen = maxlen):
    temp = tokenizer.texts_to_sequences(docs)
    temp = pad_sequences(temp, padding = 'post', maxlen = maxlen)

    return temp

X_train = transform_docs(train_csv['Utterance'])
X_test = transform_docs(test_csv['Utterance'])
X_dev = transform_docs(dev_csv['Utterance'])


y_train = np.array([key[x] for x in train_csv['Emotion']])
y_test = np.array([key[x] for x in test_csv['Emotion']])
y_dev = np.array([key[x] for x in dev_csv['Emotion']])

print(X_test.shape)
model = Sequential()
embedding_dim = 80

model.add(layers.Embedding(input_dim = vocab_size, output_dim = embedding_dim, input_length = maxlen))
# model.add(layers.Reshape((maxlen, embedding_dim, 1)))
# model.add(layers.Conv2D(64, (3, embedding_dim), padding = 'same', activation = 'relu', kernel_initializer='normal'))
# model.add(layers.Conv2D(64, (4, embedding_dim), padding = 'same', activation = 'relu', kernel_initializer='normal'))
# model.add(layers.Conv2D(64, (5, embedding_dim), padding = 'same', activation = 'relu', kernel_initializer='normal'))
# model.add(layers.GlobalMaxPool2D())
# model.add(layers.MaxPooling2D(pool_size = (3, 3), padding = 'valid'))
# model.add(layers.MaxPooling2D(pool_size = (2, 2), padding = 'valid'))
model.add(layers.Conv1D(128, 5, activation = 'relu'))
model.add(layers.Conv1D(128, 3, activation = 'relu'))
model.add(layers.Conv1D(128, 2, activation = 'relu'))
model.add(layers.GlobalMaxPool1D())

model.add(layers.Flatten())
model.add(layers.Dense(150, activation = 'relu'))
model.add(layers.Dense(7, activation = 'softmax'))

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(X_dev.shape, y_dev.shape)

model.fit(X_train, y_train, validation_data = (X_dev, y_dev), epochs = 50, class_weight = cws)

from sklearn.metrics import classification_report

y_pred = model.predict(X_test)

def get_tfarray(arr):
    result = []
    for i in arr:
        m = np.max(i)
        result.append(i == m)
    return np.array(result)

y_pred = get_tfarray(y_pred)

# accuracy_score(y_test, y_pred, normalize=False)

print(classification_report(y_test, y_pred, target_names=class_names, digits=4))