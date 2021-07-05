import numpy as np

from sklearn import preprocessing

class DataPreparer:

    def __init__(self, file_name):
        
        # def make_X(x):
        #     res = []
        #     # x = x[767:]
        #     for i in range(0, x.shape[0], 768):
        # #         print(i)
        #         res.append(np.array(x[i:i+768]))
        #     print(np.array(res).shape)
        #     return np.array(res)    

        data = np.load(file_name, allow_pickle = True)
        self.X_train = data[0]
        self.y_train = data[1]
        self.X_test = data[2]
        self.y_test = data[3]
        self.X_dev = data[4]
        self.y_dev = data[5]

        self.ss = preprocessing.StandardScaler()
        self.mm = preprocessing.MinMaxScaler()

    def scale_data(self, scaler = 'standard'):
        
        if scaler == 'standard':
            scaler = self.ss
        elif scaler == 'minmax':
            scaler = self.mm
        else:
            print("Invalid scaler. Choose either standard or minmax")

        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        self.X_dev = scaler.transform(self.X_dev)

    def set_timestep(self, time_step):


        def expand_on_time(X, timestep = time_step):
            tmp = np.reshape(X[0], (1, len(X[0])))
        #     print(tmp.shape)
            tmp = np.repeat(tmp, timestep, axis = 0)
        #     print(tmp.shape)
            ab = np.reshape(tmp, (1, len(tmp[:,:]), len(tmp[0])))
        #     print(ab.shape)
        #     print(X.shape)
            X = np.repeat(ab, X.shape[0], axis = 0)
        #     print(X.shape)
            return X

        self.X_train = expand_on_time(self.X_train)
        self.X_test = expand_on_time(self.X_test)
        self.X_dev = expand_on_time(self.X_dev)

    def get_matrices(self):
        return self.X_train, self.y_train, self.X_test, self.y_test, self.X_dev, self.y_dev

