import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import sys

sys.path.insert(1, '../')

import json
import pandas as pd
import numpy as np
from sklearn import preprocessing
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

def get_df(par_dir, class_name):
    print('Processing', par_dir)
    file_list = [par_dir + x for x in os.listdir(par_dir)]
    full = pd.DataFrame(columns = list(range(223)) + ['class'])

    i = 0
    for f in tqdm(file_list):
        df = pd.read_csv(f, sep = ',', header = None)
        full.loc[i] = list(df.mean(axis = 0)) + [class_name]
        i += 1
    return full

def wrapper(grandpa_dir, set_name, noise):
    anger = grandpa_dir + set_name + '_anger_' + noise + '/msf/'
    sad = grandpa_dir + set_name + '_sad_' + noise + '/msf/'    
    
    df_anger = get_df(anger, 'anger')
    df_sad = get_df(sad, 'sad')
    
    df_csv = pd.concat([df_anger, df_sad], ignore_index = True)
    
    return df_csv

def splitXY(df):
    X = df.drop('class', axis = 1)
    y = [emotion_key[x] for x in df['class']]
    
    return np.array(X), np.array(y)

grandpa_dir = '../../mitacs/MELD_dataset_MSF/'
noise = 'clean'

name = 'msf_' + noise 

train_csv = wrapper(grandpa_dir, 'train', noise)
test_csv = wrapper(grandpa_dir, 'test', noise)
dev_csv = wrapper(grandpa_dir, 'dev', noise)

X_train, y_train = splitXY(train_csv)
X_test, y_test = splitXY(test_csv)
X_dev, y_dev = splitXY(dev_csv)

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

nnn = Models.NormalNeuralNetwork(0.3, class_names, (X_train.shape[1], ))
nnn.model_compile(sgd)
nnn.model_fit(class_weights, 800, X_train, y_train, X_dev, y_dev, fig_name = name)

nnn_metrics = nnn.get_metrics(X_test, y_test)
print(nnn_metrics)
dump_dict(nnn_metrics, 'result/' + name + '.json')
print("METRICS\n")
print(nnn_metrics)