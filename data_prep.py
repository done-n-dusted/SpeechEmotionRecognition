import pandas as pd
import numpy as np
from sklearn import preprocessing
from STFE.SpeechTextFeatures import *
from tqdm import tqdm

emotion_key = {
    'anger'     : 0,
    'disgust'   : 1,
    'fear'      : 2,
    'joy'       : 3,
    'neutral'   : 4,
    'sadness'   : 5,
    'surprise'  : 6
}


TFE = Text_Feature_Extracter(mname_txt)

def data_to_target(data_frame, set):
    req = np.array(data_frame[['Utterance', 'Emotion', 'Dialogue_ID', 'Utterance_ID']])

    X = np.array([0]*768)
    y = [emotion_key[x] for x in data_frame['Emotion']]

    print('Preparing data for ' + set)
    for u, e, d, uid in tqdm(req):
        feats = TFE.features_fromtext(u)
        pd.to_csv('./data/' + set + '/' + e + '/' + d + '_' + uid '.csv', header = None)
        np.concatenate([X, feats])
    
    return X[1:], np.array(y)


train_csv = pd.read_csv('train_sent_emo.csv')
test_csv = pd.read_csv('test_sent_emo.csv')
dev_csv = pd.read_csv('dev_sent_emo.csv')

X_train, y_train = data_to_target(train_csv, 'train')
X_test, y_test = data_to_target(test_csv, 'test')
X_dev, y_dev = data_to_target(dev_csv, 'dev')

print('Done prepping data')