import numpy as np

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Concatenate, Reshape, GRU, Bidirectional, Dropout
from tensorflow.keras.layers import Conv1D, Flatten, TimeDistributed
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing


class General_model:
    def __init__(self, class_names, name):
        self.metrics = {"NAME" : name}
        self.name = name
        self.class_names = class_names
        self.num_classes = len(class_names)

        self.model = Sequential()

        self.early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    def model_compile(self, opt):
        self.model.compile(loss = 'categorical_crossentropy',
                            optimizer = opt,
                            metrics = ['accuracy'])
        # self.metrics['opt'] = str(opt)
        print('Model compiled')
    

    def model_fit(self, cw, num_epochs, X_train, y_train, X_dev, y_dev, save_fig = True, fig_name = None):

        hist = self.model.fit(X_train, y_train, validation_data = (X_dev, y_dev),
                        epochs = num_epochs,
                        class_weight = cw, batch_size = 32)

        if save_fig:
            plt.figure()
            plt.plot(hist.history['loss'])
            plt.title('Loss')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.savefig('result/loss_graphs/' + fig_name + '_loss.png')

        return hist

    def get_model(self):
        return self.model
    
    def get_metrics(self, X_test, y_test, return_preds = False):
        score = self.model.evaluate(X_test, y_test, verbose=0)

        def get_tfarray(arr):
            result = []
            for i in arr:
                m = np.max(i)
                result.append(i == m)
            return np.array(result)
        
        # self.metrics = {'Loss' : score[0], 'Accuracy' : score[1]}
        self.metrics['Loss'] = score[0]
        self.metrics['Accuracy'] = score[1]

        y_pred = self.model.predict(X_test)

        y_pred = get_tfarray(y_pred)

        # accuracy_score(y_test, y_pred, normalize=False)

        self.metrics["Classification Report"] = classification_report(y_test, y_pred, target_names=self.class_names, digits=4)

        cm =confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
        #Now the normalize the diagonal entries
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        self.metrics['Confusion_matrix'] = cm.tolist()

        if return_preds:
            return self.metrics, preds

        return self.metrics

    def save_model(self, location):
        self.model.save(location)
        print("Saved model at ", location )

    def load_model(self, location):
        print("Loading model from ", location)
        self.model = load_model(location)


class NormalNeuralNetwork(General_model):
    def __init__(self, dropout_size, class_names, inp_shape):

        General_model.__init__(self, class_names, 'NNN')
        
        self.model.add(Input(shape = inp_shape))

        self.model.add(Dense(256, activation = 'tanh'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(dropout_size))
        self.model.add(Dense(64, activation = 'tanh'))
        self.model.add(BatchNormalization())
        self.model.add(Dense(16, activation = 'tanh'))

        self.model.add(Dense(self.num_classes, activation = 'softmax'))

        print(self.name + " model created")
        print(self.model.summary())        
        self.metrics["Summary"] = str(self.model.summary())

class BC_LSTM(General_model):
    def __init__(self, lstmdim, dropout_size, class_names, inp_shape):
        
        General_model.__init__(self, class_names, 'bclstm')

        self.model.add(Input(shape = inp_shape))

        self.model.add(Bidirectional(LSTM(lstmdim, return_sequences = True, kernel_initializer='random_normal')))
        self.model.add(BatchNormalization())
        self.model.add(Bidirectional(LSTM(lstmdim, kernel_initializer='random_normal')))
        self.model.add(BatchNormalization())
        # self.model.add(LSTM(128, kernel_initializer='random_normal'))
        # self.model.add(BatchNormalization())

        self.model.add(Dense(100, activation = 'relu'))
        self.model.add(Dropout(0.5))
        # self.model.add(Dense(50, activation = 'relu'))
        self.model.add(Dense(self.num_classes, activation = 'softmax'))

        print(self.name + " model created")
        print(self.model.summary())
        self.metrics["Summary"] = self.model.summary()

class TextCNN(General_model):
    def __init__(self, class_names, inp_shape):
        General_model.__init__(self, class_names, 'textCNN')

        self.model.add(Input(shape = inp_shape))

        self.model.add(Conv1D(32, 3, padding = 'same', activation = 'relu'))
        self.model.add(Conv1D(64, 4, padding = 'same', activation = 'relu'))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation = 'relu'))
        self.model.add(Dense(128, activation = 'relu'))
        self.model.add(Dense(self.num_classes, activation = 'softmax'))

        print(self.name + " model created")
        print(self.model.summary())
        self.metrics["Summary"] = self.model.summary()
