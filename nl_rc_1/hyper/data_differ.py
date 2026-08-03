import numpy as np
from scipy import linalg
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class nlrc:
    def __init__(self, b=0.5,reg=1e-8,n=10,test_len = 10):
        self.b = b
        self.reg = reg
        self.n = n
        self.test_len = test_len

    def gls_neuron_gen(self,x):
        b = self.b
        neigh = 1e-8
        x = np.clip(x,neigh,1-neigh)
        
        if x>=b:
            return (1-x)/(1-b)
        return x/b
    
    def build_features(self,data):
        n = self.n
        history_x = np.zeros((1+n,len(data)))
        for i in range(len(data)):
            u = data[i]
            X = np.zeros(n)
            X[0] = u.item()
            for j in range(n-1):
                X[j+1] = self.gls_neuron_gen(X[j])
            history_x[:,i] = np.vstack((1,X.reshape(-1,1))).flatten()
        
        return history_x
    
    def fit(self,data,y_data):
        yt = y_data.reshape(-1,1)
        yt = yt.T
        reg = self.reg
        history_x = self.build_features(data)
        self.w_out = np.dot(np.dot(yt,history_x.T),linalg.inv(np.dot(history_x,history_x.T)+np.eye(self.n+1)*reg))
        return self
    
    def predict(self,test_u):
        test_len = self.test_len
        n = self.n
        Y = np.zeros((1,test_len))
        u = test_u
        for i in range(test_len):
            X = np.zeros(n)
            X[0] = u
            for j in range(n-1):
                X[j+1] = self.gls_neuron_gen(X[j])
            y = np.dot(self.w_out,np.vstack((1,X.reshape(-1,1))).flatten())
            Y[:,i] = y
            u = y.item()
        return Y.flatten()

def beta_data(a = 3.95, i = 0.5422,length = 1000):
    x = np.zeros(length)
    x[0] = i
    for t in range(length-1):
        x[t+1] = a*x[t]*(1-x[t])
    return x


data = beta_data()

train_len = 100
test_len = 10
shift = 10
n_list = np.arange(5,30,1)
mse_list = []


def data_shift(data,k):
    X_train = data[k:train_len+k]
    X_test = data[train_len+k:train_len+test_len+k]

    y_train = data[1+k:train_len+1+k]
    y_test = data[train_len+1+k:train_len+test_len+1+k]

    scalar = MinMaxScaler(feature_range=(0,1))


    X_train = scalar.fit_transform(X_train.reshape(-1,1))
    X_test = scalar.transform(X_test.reshape(-1,1))
    y_train = scalar.transform(y_train.reshape(-1,1))
    y_test = y_test.flatten()

    return X_train,X_test,y_train,y_test,scalar


for n in n_list:
    mse_cal = []
    for k in range(shift):
        model = nlrc(n=n,test_len=test_len)
        X_train,X_test,y_train,y_test,scalar = data_shift(data,k)
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test[0].item())
        y_pred = scalar.inverse_transform(y_pred.reshape(-1,1)).flatten()
        mse = mean_squared_error(y_test,y_pred)
        mse_cal.append(mse)
    mse = np.mean(mse_cal)
    print(f'{n} -----> {mse}')
    mse_list.append(mse)




