from STFE import Models, DataOrganizer
from tensorflow.keras import optimizers

DO = DataOrganizer.DataOrganizer('all_data.npy')

class_names = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
time_step = 30
sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
class_weights = arr = [4.0, 15.0, 15.0, 3.0, 1.0, 6.0, 3.0]


DO.scale_data()
DO.set_timestep(time_step)
X_train, y_train, X_test, y_test, X_dev, y_dev = DO.get_matrices()



print("\nOrganized data for training\n")

print('\nBCLSTM MODEL\n')
bclstm = Models.BC_LSTM(10, 0.3, class_names, (30, 768, ))
bclstm.model_compile(sgd)
bclstm.model_fit(class_weights, 150, X_train, y_train, X_dev, y_dev)

bclstm_metrics = bclstm.get_metrics(X_test, y_test)
print("METRICS\n")
print(bclstm)

print('\nNEURAL NETWORK MODEL\n')

nnn = Models.NormalNeuralNetwork(0.3, class_names, (768, ))
nnn.model_compile(sgd)
nnn.model_fit(class_weights, 150, X_train, y_train, X_dev, y_dev)

nnn_metrics = nnn.get_metrics(X_test, y_test)
print("METRICS\n")
print(nnn_metrics)



