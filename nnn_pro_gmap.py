import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

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

def get_df(pro_dir, mrs_dir, class_name):
    print('Processing', pro_dir, mrs_dir)
    pro_list = [x for x in os.listdir(pro_dir)]
    mrs_list = [x for x in os.listdir(mrs_dir)]

    common_list = list(set(pro_list).intersection(mrs_list))
    pro = [pro_dir + x for x in common_list]
    mrs = [mrs_dir + x for x in common_list]
    
    req_len = len(pd.read_csv(mrs[0], sep = ';', header = None, skiprows = [0]).columns)
    full = pd.DataFrame(columns = list(range(3 + req_len - 2)) + ['class'])
    
    i = 0
    # scalar = preprocessing.StandardScalar()
    for f in tqdm(common_list):
        pro_curr = pro_dir + f
        mrs_curr = mrs_dir + f
        
        pro_df = pd.read_csv(pro_curr, sep = ';', header = None, skiprows = [0])
        pro_df.drop([0, 1], axis = 1, inplace = True)

        # pro_df = (pro_df - pro_df.min()) / (pro_df.max() - pro_df.min())
        mrs_df = pd.read_csv(mrs_curr, sep = ';', header = None)
        mrs_df.drop([0, 1], axis = 1, inplace = True)

        # print(len(pro_df.columns), len(mrs_df.columns), len(full.columns), req_len)
        full.loc[i] = list(pro_df.mean(axis = 0)) + list(mrs_df) + [class_name]
        
        i += 1
    return full

def wrapper(pro_grand, mrs_grand, set_name, noise):
    pro_anger = pro_grand + set_name + '_anger_' + noise + '/'
    mrs_anger = mrs_grand + set_name + '_anger_' + noise + '/'
    
    pro_sad = pro_grand + set_name + '_sad_' + noise + '/'
    mrs_sad = mrs_grand + set_name + '_sad_' + noise + '/'
    
    df_anger = get_df(pro_anger, mrs_anger, 'anger')
    df_sad = get_df(pro_sad, mrs_sad, 'sad')
    
    df_csv = pd.concat([df_anger, df_sad], ignore_index = True)
    
    return df_csv

def splitXY(df):
    X = df.drop('class', axis = 1)
    y = [emotion_key[x] for x in df['class']]
    
    return np.array(X), np.array(y)

pro_grand = '../../mitacs/MELD_noise_prosodic_feat/'
mrs_grand = '../../mitacs/MELD_noise_eGEMAPS_feat/'

noise = 'babble_10dB'
name = 'pro_gmap_' + noise

train_csv = wrapper(pro_grand, mrs_grand, 'train', noise)
test_csv = wrapper(pro_grand, mrs_grand, 'test', noise)
dev_csv = wrapper(pro_grand, mrs_grand, 'dev', noise)

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

DP.scale_data(scaler = 'minmax')
print("\nOrganized data for training\n")

X_train, y_train, X_test, y_test, X_dev, y_dev = DP.get_matrices()
sgd = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)

print('\nNEURAL NETWORK MODEL\n')

nnn = Models.NormalNeuralNetwork(0.3, class_names, (X_train.shape[1], ))
nnn.model_compile(sgd)
nnn.model_fit(class_weights, 800, X_train, y_train, X_test, y_test, fig_name = name)

nnn_metrics = nnn.get_metrics(X_dev, y_dev)
print(nnn_metrics)
dump_dict(nnn_metrics, 'result/' + name + '.json')
print("METRICS\n")
print(nnn_metrics)