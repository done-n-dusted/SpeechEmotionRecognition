import numpy as np
# import nltk
import sklearn
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

key = {
    'anger' : [1, 0],
    'sadness' : [0, 1]
}

data_train = pd.read_csv('./text_csv/train_sent_emo.csv')[['Utterance', 'Emotion']]
data_train = data_train[(data_train['Emotion'] == 'anger') | (data_train['Emotion'] == 'sadness')]

data_test = pd.read_csv('./text_csv/test_sent_emo.csv')[['Utterance', 'Emotion']]
data_test = data_test[(data_test['Emotion'] == 'anger') | (data_test['Emotion'] == 'sadness')]

data_dev = pd.read_csv('./text_csv/dev_sent_emo.csv')[['Utterance', 'Emotion']]
data_dev = data_dev[(data_dev['Emotion'] == 'anger') | (data_dev['Emotion'] == 'sadness')]

vectorizer = CountVectorizer(min_df = 3)

train_bow = vectorizer.fit_transform(data_train['Utterance'])
test_bow = vectorizer.transform(data_test['Utterance'])
dev_bow = vectorizer.transform(data_dev['Utterance'])

X_train = train_bow.toarray()
X_test = test_bow.toarray()
X_dev = dev_bow.toarray()

y_train = np.array([key[x] for x in data_train['Emotion']])
y_test = np.array([key[x] for x in data_test['Emotion']])
y_dev = np.array([key[x] for x in data_dev['Emotion']])

vocab = vectorizer.get_feature_names()

np.save('bow_features.npy', [X_train, y_train, X_test, y_test, X_dev, y_dev])

with open('vocab.txt', 'w') as vfile:
    for item in vocab:
        vfile.write(str(item) + '\n')