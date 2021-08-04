# Saves all TCNN features as a npy file


import pandas as pd
import numpy as np
from sklearn import preprocessing
from STFE.TextCNN import *
from tqdm import tqdm

emotion_key = {
    'anger' : [1, 0],
    'sadness' : [0, 1]
}

train_csv = pd.read_csv('text_csv/train_sent_emo.csv')
test_csv = pd.read_csv('text_csv/test_sent_emo.csv')
dev_csv = pd.read_csv('text_csv/dev_sent_emo.csv')

FE = TCNN_feature_extracter(train_csv['Utterance'])

def data_to_target(data_frame, set):
    req = np.array(data_frame[['Utterance', 'Emotion', 'Dialogue_ID', 'Utterance_ID']])
    
    X = np.array([np.array([0]*150)])
    y = []

    print('\nPreparing data for ' + set)

    for u, e, d, uid in tqdm(req):
        if e == 'anger' or e == 'sadness':
            feats = FE.extract_features(u)
            # print(feats.shape, X.shape)
            X = np.concatenate([X, feats])
            y.append(emotion_key[e])

    return X[1:], np.array(y)

X_train, y_train = data_to_target(train_csv, 'train')
X_test, y_test = data_to_target(test_csv, 'test')
X_dev, y_dev = data_to_target(dev_csv, 'dev')

print('Done prepping data')
print(X_train.shape)

np.save('./tcnn_features.npy', np.array([X_train, y_train, X_test, y_test, X_dev, y_dev]))