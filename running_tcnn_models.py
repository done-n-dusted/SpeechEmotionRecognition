# Training and testing various models on tcnn features


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from STFE import Models, DataPreparer
from tensorflow.keras import optimizers
import json


def dump_dict(dict, file_name):
    with open(file_name, 'w') as convert_file:
        json.dump(dict, convert_file)


# DP = DataPreparer.DataPreparer('all_data.npy')

DP = DataPreparer.DataPreparer('tcnn_features.npy')


# class_names = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
class_names = ['anger', 'sadness']

time_step = 15
sgd = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
adam = optimizers.Adam(learning_rate = 1e-06)
cws = [1, 2]
class_weights = {}
for i in range(len(class_names)):
    class_weights[i] = cws[i]

print(class_weights)


DP.scale_data()

print("\nOrganized data for training\n")

while(True):
    model_name = input("TextCNN : BCLSTM or NNN or TEXTCNN\n")

    if model_name == 'BCLSTM':
        DP.set_timestep(time_step)
        X_train, y_train, X_test, y_test, X_dev, y_dev = DP.get_matrices()

        print('\nBCLSTM MODEL\n')
        bclstm = Models.BC_LSTM(150, 0.3, class_names, (time_step, 150, ))
        bclstm.model_compile(adam)
        bclstm.model_fit(class_weights, 150, X_train, y_train, X_dev, y_dev, fig_name = 'TCNN_BLSTM')

        bclstm_metrics = bclstm.get_metrics(X_test, y_test)
        dump_dict(bclstm_metrics, 'result/bclstm_textcnn.json')
        print("METRICS\n")
        print(bclstm)
        break

    elif model_name == 'NNN':
        X_train, y_train, X_test, y_test, X_dev, y_dev = DP.get_matrices()

        print('\nNEURAL NETWORK MODEL\n')

        nnn = Models.NormalNeuralNetwork(0.3, class_names, (150, ))
        nnn.model_compile(sgd)
        nnn.model_fit(class_weights, 500, X_train, y_train, X_dev, y_dev, fig_name = 'TCNN_NN')

        nnn_metrics = nnn.get_metrics(X_test, y_test)
        print(nnn_metrics)
        dump_dict(nnn_metrics, 'result/nnn_textcnn.json')
        print("METRICS\n")
        print(nnn_metrics)
        break

    elif model_name == 'TEXTCNN':
        DP.set_timestep(5)
        X_train, y_train, X_test, y_test, X_dev, y_dev = DP.get_matrices()

        print('\nTEXT CONV NEURAL NETWORK\n')

        nnn = Models.TextCNN(class_names, (5, 150, ))
        nnn.model_compile(sgd)
        nnn.model_fit(class_weights, 150, X_train, y_train, X_dev, y_dev, fig_name = 'tcnn_tcnn')

        nnn_metrics = nnn.get_metrics(X_test, y_test)
        print(nnn_metrics)
        dump_dict(nnn_metrics, 'result/text_cnn.json')
        print("METRICS\n")
        print(nnn_metrics)
        break

    else:
        print("Invalid Model name")