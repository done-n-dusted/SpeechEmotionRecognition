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

X_train, y_train, X_dev, y_dev = np.load('../../aug_clean_all.npy', allow_pickle = True)

name = 'clean_aug'

class_names = ['anger', 'sad']
cws = [1, 1.9]
class_weights = {}
for i in range(len(class_names)):
    class_weights[i] = cws[i]

sgd = optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)

nnn = Models.NormalNeuralNetwork(0.3, class_names, (X_train.shape[1], ), type = 'AUG')
# nnn.model_compile(sgd)
nnn.model_compile(sgd)

nnn.model_fit(class_weights, 200, X_train, y_train, X_dev, y_dev, fig_name = name)

model = nnn.get_model()

# TODO change model name
model.save('../../models/clean_aug14.h5', include_optimizer = False)

'''
1 -> 2 extra layers
2 -> 
3 -> 
4 -> batch size = 64, no patience
5 -> batch size = 8, callbacks patience = 50
6 -> "              , patience = 5, Tanh, adam
7 -> dropout = 0, patience = 10, tanh, sgd(lr = 1e-4)
8 -> dropout = 0.3, patience = None, tanh, batch size = 8, epochs = 200
9 -> no batch norm, 128 nodes layers, patience 10, batch = 4, dropout = 0.3, one tanh
10 -> No batch norm, 128 nodes layers, patience 10, batch = 2
11 -> dp = 0.2 tanh, No batch Norm, 128 nodes 2 layers, patience 20, batch = 4
12 -> dp = 0.3, patience 5, batch = 4, momentum 0.95
13 -> dp = 0.3, patience 10, batch = 4, momentum 0.95, one leaky relu
14 -> dp = 0.3, lr = 1e-3, momentum = 0.9, no leaky
'''