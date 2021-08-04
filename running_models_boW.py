# training and testing on various model for BoW features


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from STFE import Models, DataPreparer
from tensorflow.keras import optimizers
import json


def dump_dict(dict, file_name):
    with open(file_name, 'w') as convert_file:
        json.dump(dict, convert_file)

DP = DataPreparer.DataPreparer('bow_features.npy')
class_names = ['anger', 'sadness']

vocab = open('vocab.txt')
vocab_size = len(vocab.readlines())

time_step = 30
sgd = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
cws = [1, 1.8]
class_weights = {}
for i in range(len(class_names)):
    class_weights[i] = cws[i]

print(class_weights)


DP.scale_data()

print("\nOrganized data for training\n")

while(True):
    model_name = input("BCLSTM or NNN or TEXTCNN\n")

    if model_name == 'BCLSTM':
        DP.set_timestep(time_step)
        X_train, y_train, X_test, y_test, X_dev, y_dev = DP.get_matrices()

        print('\nBCLSTM MODEL\n')
        bclstm = Models.BC_LSTM(10, 0.3, class_names, (30, vocab_size, ))
        bclstm.model_compile(sgd)
        bclstm.model_fit(class_weights, 150, X_train, y_train, X_dev, y_dev, fig_name = 'bow_BCLSTM')

        bclstm_metrics = bclstm.get_metrics(X_test, y_test)
        dump_dict(bclstm_metrics, 'result/bclstm_bow.json')
        print("METRICS\n")
        print(bclstm)
        break

    elif model_name == 'NNN':
        X_train, y_train, X_test, y_test, X_dev, y_dev = DP.get_matrices()

        print('\nNEURAL NETWORK MODEL\n')

        nnn = Models.NormalNeuralNetwork(0.3, class_names, (vocab_size, ))
        nnn.model_compile(sgd)
        nnn.model_fit(class_weights, 150, X_train, y_train, X_dev, y_dev, fig_name = 'bow_NNN')

        nnn_metrics = nnn.get_metrics(X_test, y_test)
        print(nnn_metrics)
        dump_dict(nnn_metrics, 'result/nnn_bow.json')
        print("METRICS\n")
        print(nnn_metrics)
        break

    elif model_name == 'TEXTCNN':

        DP.set_timestep(5)
        X_train, y_train, X_test, y_test, X_dev, y_dev = DP.get_matrices()

        print('\nTEXT CONV NEURAL NETWORK\n')

        nnn = Models.TextCNN(class_names, (5, vocab_size, ))
        nnn.model_compile(sgd)
        nnn.model_fit(class_weights, 150, X_train, y_train, X_dev, y_dev, fig_name = 'bow_TCNN')

        nnn_metrics = nnn.get_metrics(X_test, y_test)
        print(nnn_metrics)
        dump_dict(nnn_metrics, 'result/text_bow.json')
        print("METRICS\n")
        print(nnn_metrics)
        break
    else:
        print("Invalid Model name")
