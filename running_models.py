from STFE import Models, DataPreparer
from tensorflow.keras import optimizers
import json


def dump_dict(dict, file_name):
    with open(file_name, 'w') as convert_file:
        convert_file.write(json.dumps(dict))


DP = DataPreparer.DataPreparer('all_data.npy')

class_names = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
time_step = 30
sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
class_weights = arr = [4.0, 15.0, 15.0, 3.0, 1.0, 6.0, 3.0]


DP.scale_data()
DP.set_timestep(time_step)
X_train, y_train, X_test, y_test, X_dev, y_dev = DP.get_matrices()

print("\nOrganized data for training\n")

while(True):
    model_name = input("BCLSTM or NNN\n")

    if model_name == 'BCLSTM':
        print('\nBCLSTM MODEL\n')
        bclstm = Models.BC_LSTM(10, 0.3, class_names, (30, 768, ))
        bclstm.model_compile(sgd)
        bclstm.model_fit(class_weights, 150, X_train, y_train, X_dev, y_dev)

        bclstm_metrics = bclstm.get_metrics(X_test, y_test)
        dump_dict(bclstm_metrics, 'result/bclstm.json')
        print("METRICS\n")
        print(bclstm)
        break

    elif model_name == 'NNN':
        print('\nNEURAL NETWORK MODEL\n')

        nnn = Models.NormalNeuralNetwork(0.3, class_names, (768, ))
        nnn.model_compile(sgd)
        nnn.model_fit(class_weights, 150, X_train, y_train, X_dev, y_dev)

        nnn_metrics = nnn.get_metrics(X_test, y_test)
        dump_dict(nnn_metrics, 'result/nnn.json')
        print("METRICS\n")
        print(nnn_metrics)
        break

    else:
        print("Invalid Model name")
