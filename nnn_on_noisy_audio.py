import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import json
import pandas as pd
import numpy as np
from sklearn import preprocessing
from STFE.SpeechTextFeatures import *
from tqdm import tqdm
from STFE import Models, DataPreparer
from tensorflow.keras import optimizers

emotion_key = {
    'anger' : [1, 0],
    'sad' : [0, 1]
}

def dump_dict(dict, file_name):
    with open(file_name, 'w') as convert_file:
        json.dump(dict, convert_file)


mname_txt = 'bert-base-cased'
TFE = BERT_Text_Feature_Extracter(mname_txt)

def data_to_target(data_frame, set):
    req = np.array(data_frame[['Utterance', 'Class']])

    X = np.array([np.array([0]*768)])
    y = []

    print('\nPreparing data for ' + set)
    for u, e in tqdm(req):
        
        if pd.isna(u) == False:
            feats  = TFE.features_fromtext(u)
            X = np.concatenate([X, [feats]])
            y.append(emotion_key[e])
        
    return X[1:], np.array(y)

noise_name = "airport"
db = "0dB"

name = noise_name + '_' + db
train_csv = pd.read_csv('noise_csv/train_' + noise_name + '_' + db + '.csv')
test_csv = pd.read_csv('noise_csv/test_' + noise_name + '_' + db + '.csv')
dev_csv = pd.read_csv('noise_csv/dev_' + noise_name + '_' + db + '.csv')

X_train, y_train = data_to_target(train_csv, 'train_' + noise_name + '_' + db + '.csv')
X_test, y_test = data_to_target(test_csv, 'test_' + noise_name + '_' + db + '.csv')
X_dev, y_dev = data_to_target(dev_csv, 'dev_' + noise_name + '_' + db + '.csv')

print("Saved files")

DP = DataPreparer.ParentDataPrep(X_train, y_train, X_test, y_test, X_dev, y_dev)

class_names = ['anger', 'sad']
cws = [1, 1.8]
class_weights = {}
for i in range(len(class_names)):
    class_weights[i] = cws[i]

DP.scale_data()
print("\nOrganized data for training\n")

X_train, y_train, X_test, y_test, X_dev, y_dev = DP.get_matrices()
sgd = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)

print('\nNEURAL NETWORK MODEL\n')

nnn = Models.NormalNeuralNetwork(0.3, class_names, (768, ))
nnn.model_compile(sgd)
nnn.model_fit(class_weights, 500, X_train, y_train, X_dev, y_dev, fig_name = name)

nnn_metrics = nnn.get_metrics(X_test, y_test)
print(nnn_metrics)
dump_dict(nnn_metrics, 'result/' + name + '.json')
print("METRICS\n")
print(nnn_metrics)