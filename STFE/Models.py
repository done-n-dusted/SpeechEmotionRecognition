import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Concatenate, Reshape, GRU, Bidirectional
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix


class General_model:
    def __init__(self, class_names, name):
        self.metrics = {}
        self.name = name
        self.class_names = class_names
        self.num_classes = len(class_names)

        self.model = Sequential()

        self.early_stopping = EarlyStopping(monitor='loss', patience=10)

    def model_compile(self, opt):
        self.model.compile(loss = 'categorical_crossentropy',
                            optimizer = opt,
                            metrics = ['accuracy'])
        print('Model compiled')
    

    def model_fit(self, cw, num_epochs, X_train, y_train, X_dev, y_dev):

        hist = self.model.fit(X_train, y_train, validation_data = (X_dev, y_dev),
                        epochs = num_epochs, 
                        callbacks = [self.early_stopping],
                        class_weight = cw)

        plt.figure()
        plt.plot(ist.history['loss'])
        plt.title('Loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(self.name + '_loss.png')

    def get_model(self):
        return self.model
    
    def get_metrics(self, X_test, y_test, return_preds = False):
        score = model.evaluate(X_test, y_test, verbose=0)
        
        self.metrics = {'Loss' : score[0], 'Accuracy' = score[1]}

        y_pred = model.predict(X_test)

        y_pred = get_tfarray(y_pred)

        accuracy_score(y_test, y_pred, normalize=False)

        self.metrics["Classification Report"] = classification_report(y_test, y_pred, target_names=class_names, digits=4)

        cm =confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
        #Now the normalize the diagonal entries
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        self.metrics['Confusion_matrix'] = cm

        if return_preds:
            return self.metrics, preds

        return self.metrics


class NormalNeuralNetwork(General_model):
    def __init__(self, dropout_size, class_names, inp_shape):

        General_model.__init__(self, class_names, 'NNN')
        
        self.model.add(Input(shape = inp_shape))

        self.model.Dense(512, activation = 'relu')
        self.model.Dropout(dropout_size)
        self.model.Dense(128, activation = 'relu')
        self.model.Dense(64, activation = 'relu')

        self.model.add(Dense(self.num_classes, activation = 'softmax'))

        print(self.name + " model created")
        print(self.model.summary())        

class BC_LSTM(General_model):
    def __init__(self, lstmdim, dropout_size, class_names, inp_shape):
        
        General_model.__init__(self, class_names, 'bclstm')

        self.model.add(Input(shape = inp_shape))

        self.model.add(Bidirectional(LSTM(lstmdim, dropout=dropout_size, return_sequences = True)))
        self.model.add(Bidirectional(LSTM(lstmdim, dropout=dropout_size)))

        self.model.add(Dense(self.num_classes, activation = 'softmax'))

        print(self.name + " model created")
        print(self.model.summary())