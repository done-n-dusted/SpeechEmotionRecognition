import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import pandas as pd
import numpy as np
from sklearn import preprocessing
from STFE.SpeechTextFeatures import *
from tqdm import tqdm

# emotion_key = {
#     'anger'     : [1, 0, 0, 0, 0, 0, 0],
#     'disgust'   : [0, 1, 0, 0, 0, 0, 0],
#     'fear'      : [0, 0, 1, 0, 0, 0, 0],
#     'joy'       : [0, 0, 0, 1, 0, 0, 0],
#     'neutral'   : [0, 0, 0, 0, 1, 0, 0],
#     'sadness'   : [0, 0, 0, 0, 0, 1, 0],
#     'surprise'  : [0, 0, 0, 0, 0, 0, 1]
# }

emotion_key = {
    'anger' : [1, 0],
    'sadness' : [0, 1]
}

mname_txt = 'bert-base-cased'

TFE = BERT_Text_Feature_Extracter(mname_txt)

def data_to_target(data_frame, set):
    req = np.array(data_frame[['Utterance', 'Emotion', 'Dialogue_ID', 'Utterance_ID']])

    X = [np.array([0]*768)]
    # print(X)
    # y = [emotion_key[x] for x in data_frame['Emotion']]
    y = []
    # for x in data_frame['Emotion']:
    #     if x == 'anger' or x =='sadness':
    #         y.append(emotion_key[x])
            

    print('\nPreparing data for ' + set)
    for u, e, d, uid in tqdm(req):
        # print(d)
        if e == 'anger' or e == 'sadness':    
        
            feats = TFE.features_fromtext(u)
            # print(len(feats))
            # pd.DataFrame(feats).to_csv('../data/' + set + '/' + e + '/dia' + str(d) + '_utt' + str(uid) + '.csv', header = None, index = False)
            X = np.concatenate([X, [feats]])
            y.append(emotion_key[e])
    
    return X[1:], np.array(y)


train_csv = pd.read_csv('text_csv/train_sent_emo.csv')
test_csv = pd.read_csv('text_csv/test_sent_emo.csv')
dev_csv = pd.read_csv('text_csv/dev_sent_emo.csv')

X_train, y_train = data_to_target(train_csv, 'train')
X_test, y_test = data_to_target(test_csv, 'test')
X_dev, y_dev = data_to_target(dev_csv, 'dev')

print('Done prepping data')
print(X_train.shape)

np.save('./anger_and_sad_only.npy', np.array([X_train, y_train, X_test, y_test, X_dev, y_dev]))
