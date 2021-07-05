import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

import pandas as pd


class TextCNN:
    def __init__(self, sentences_train):

        self.tokenizer = Tokenizer(num_words=5000)
        self.tokenizer.fit_on_texts(sentences_train)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.maxlen = 100
        self.model = Sequential()

    def transform_docs(self, docs):
        temp = self.tokenizer.texts_to_sequences(docs)
        temp = pad_sequences(temp, padding = 'post', maxlen = self.maxlen)

        return temp

    def make_model(self):

        embedding_dim = 50
        self.model.add(layers.Embedding(input_dim = self.vocab_size, output_dim = embedding_dim, input_length = self.maxlen, trainable = False))
        self.model.add(layers.Reshape((self.maxlen, embedding_dim, 1)))
        self.model.add(layers.Conv2D(32, (3, embedding_dim), padding = 'same', activation = 'relu', kernel_initializer='normal', trainable = False))
        self.model.add(layers.Conv2D(32, (4, embedding_dim), padding = 'same', activation = 'relu', kernel_initializer='normal', trainable = False))
        self.model.add(layers.Conv2D(16, (5, embedding_dim), padding = 'same', activation = 'relu', kernel_initializer='normal', trainable = False))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(100, activation = 'tanh'))
        
        self.model.summary()
        # self.model.compile(optimizer='adam', loss='rmse', metrics=['accuracy'])
        return self.model

class TCNN_feature_extracter:
    def __init__(self, sec_train):
        self.TC = TextCNN(sec_train)
        self.fe = self.TC.make_model()

    def extract_features(self, text):
        seq = self.TC.transform_docs([text])
        return self.fe.predict(seq)


if __name__ == '__main__':
    sec_train = pd.read_csv('../text_csv/train_sent_emo.csv')['Utterance']
    
    TC = TextCNN(sec_train)
    model = TC.make_model()

    sen = 'Oh my god I see you walking by'
    seq = TC.transform_docs([sen])
    print(model.predict(seq))

