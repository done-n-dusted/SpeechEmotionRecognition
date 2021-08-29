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
from pickle import dump

emotion_key = {
    'anger' : 0,
    'sad' : 1
}


def get_df(gmap_dir, msf_dir, txt_dir, class_name):
    print('Processing', gmap_dir, msf_dir, txt_dir)
    gmap_list = [x for x in os.listdir(gmap_dir)]
    msf_list = [x for x in os.listdir(msf_dir)]
    common_list = list(set(gmap_list).intersection(msf_list))
    txt_list = [x for x in os.listdir(txt_dir)]

    common_list = list(set(common_list).intersection(txt_list))

    gmap = [gmap_dir + x for x in common_list]
    msf = [msf_dir + x for x in common_list]
    txt = [txt_dir + x for x in common_list]

    gmap_len = len(pd.read_csv(gmap[0], sep = ';', header = None, skiprows = [0]).columns)
    text_len = 768
    full = pd.DataFrame(columns = list(range(223 + gmap_len + text_len - 2)) + ['class'])
    i = 0

    for f in tqdm(common_list):
        gmap_curr = gmap_dir + f
        msf_curr = msf_dir + f
        txt_curr = txt_dir + f

        gmap_df = pd.read_csv(gmap_curr, sep = ';', header = None, skiprows = [0], index_col = False)
        gmap_df.drop([0, 1], axis = 1, inplace = True)
        msf_df = pd.read_csv(msf_curr, sep = ',', header = None).mean(axis = 0)
        txt_df = pd.read_csv(txt_curr)

        full.loc[i] = list(gmap_df.loc[0]) + list(msf_df) + list(txt_df['0']) + [class_name]
        i += 1
    return full


def wrapper(gmap_grand, msf_grand, txt_grand, set_name, prefix = ''):

    gmap_anger = gmap_grand +prefix +  set_name + '_anger_clean/'
    msf_anger = msf_grand + prefix + set_name + '_anger_clean/msf/'
    txt_anger = txt_grand + prefix + set_name + '_anger_clean/'
    
    gmap_sad = gmap_grand +prefix +  set_name + '_sad_clean/'
    msf_sad = msf_grand + prefix + set_name + '_sad_clean/msf/' 
    txt_sad = txt_grand + prefix + set_name + '_sad_clean/'
    

    df_anger = get_df(gmap_anger, msf_anger, txt_anger, 'anger')
    df_sad = get_df(gmap_sad, msf_sad, txt_sad, 'sad')
    
    df_csv = pd.concat([df_anger, df_sad], ignore_index = True)
    
    df_csv = df_csv.sample(frac = 1)

    return df_csv

def splitXY(df):
    X = df.drop('class', axis = 1)
    y = [emotion_key[x] for x in df['class']]

    def f(x):
        return np.float(x)
    f2 = np.vectorize(f)
    
    X = np.array(X)
    y = np.array(y)
    return f2(X), y


gmap_grand = '../../../mitacs/MELD_clean_aug_eGEMAPS_feat/' 
msf_grand = '../../../mitacs/MELD_clean_aug_MSF/'
txt_grand = '../../../mitacs/MELD_clean_aug_text/'

sets = ['dev', 'train']
emos = ['anger', 'sad']
original = []



types = 'ABCDEFGHIJKLMNPQR'

noise_list = [x + '_' for x in types]

# collecting clean data
train_csv = wrapper(gmap_grand, msf_grand, txt_grand, 'train')
dev_csv = wrapper(gmap_grand, msf_grand, txt_grand, 'dev')

X_train, y_train = splitXY(train_csv)
X_dev, y_dev = splitXY(dev_csv)

# print(X_train.shape)

print("Loaded Augmented data")

for n in noise_list:
    print('\n', n, '\n')

    dset_csv = wrapper(gmap_grand, msf_grand, txt_grand, 'train', prefix = n)
    dx_train, dy_train = splitXY(dset_csv)
    X_train = np.concatenate([X_train, dx_train], axis = 0)
    y_train = np.concatenate([y_train, dy_train], axis = 0)

scaler = preprocessing.StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_dev = scaler.transform(X_dev)

print("X_train shape", X_train.shape)

dump(scaler, open('../../models/clean_aug_scaler.pkl', 'wb'))

print("Done scaling data")

np.save('../aug_clean_all.npy', np.array([X_train, y_train, X_dev, y_dev]))
