import numpy as np
import random
from scipy import linalg
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

'''
what does this code does??

1. it uses the gls neuron binary trace values as the feature extractor of the time series input of
of the previous data.
2. we do not keep a neighbour hood region instead we have a specific number of iterations that the
gls neuron can take
3. the made feature training matrix is used for the calculation of the w_out
4. next the name w_out is used for the calcualtion of the recursive value of the y,
(recusrive mean, the previous output is taken as the next input)
5. done. the remaining features are then quotedd in the code'''

class nlrc:
    def __init__(self, b=0.5,reg=1e-8,n=10,test_len = 10,start = 0.2,eps = 0.05):
        self.b = b
        self.reg = reg
        self.n = n
        self.test_len = test_len
        self.start = start
        self.eps = eps

    def gls_neuron_gen(self,x):
        """
        the gls neuron binary trace values as the feature extractor of the time series input of
        of the previous data.
        """
        b = self.b
        neigh = 1e-8
        x = np.clip(x,neigh,1-neigh)

        if x>=b:
            return (1-x)/(1-b)
        return x/b
    
    def build_features(self,data):
        '''
        building the features of the specific time point using the gls neuron binary trace values
        '''
        n = self.n
        start = self.start
        eps = self.eps
        history_x = np.zeros((1+n,len(data)))
        for i in range(len(data)):
            u = data[i] #setting the first value
            X = []
            X.append(start) #item becoz, u might be ---> array[u] instead of float(u)
            while X[-1] > u+eps or X[-1] < u-eps:
                X.append(self.gls_neuron_gen(X[-1]))
            if len(X) > n:
                X = X[-n:]
            else:
                while len(X) < n:
                    X.insert(0,0)
            X = np.array(X)
            history_x[:,i] = np.vstack((1,X.reshape(-1,1))).flatten()
        
        return history_x #printing the it with shape(n+1,train_len)
    
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
        start = self.start
        eps = self.eps
        Y = np.zeros((1,test_len))
        u = test_u
        for i in range(test_len):
            X = []
            X.append(start)
            while X[-1] > u+eps or X[-1] < u-eps:
                X.append(self.gls_neuron_gen(X[-1]))
            if len(X) > n:
                X = X[-n:]
            else:
                while len(X) < n:
                    X.insert(0,0)
            X = np.array(X)
            y = np.dot(self.w_out,np.vstack((1,X.reshape(-1,1))).flatten())
            Y[:,i] = y
            u = y.item() #y is an array, so we send only the float values further
        return Y.flatten()

def logistic_data(a = 3.95, i = 0.5422,length = 2000):
    x = np.zeros(length)
    x[0] = i
    for t in range(length-1):
        x[t+1] = a*x[t]*(1-x[t])
    return x

def mackey_glass(length=2000, tau=17, beta=0.2, gamma=0.1, n=10, dt=1.0):
    x = np.zeros(length)
    
    # initial condition (important!)
    x[:tau+1] = 1.2  
    
    for t in range(tau, length-1):
        x_tau = x[t - tau]
        x[t+1] = x[t] + dt * (beta * x_tau / (1 + x_tau**n) - gamma * x[t])
    
    return x

def tent_map(mu=1.9999, x0=0.5422, length=2000):
    x = np.zeros(length)
    x[0] = x0

    for t in range(length-1):
        if x[t] < 0.5:
            x[t+1] = mu*x[t]
        else:
            x[t+1] = mu*(1-x[t])

    return x

def chebyshev_map(length=1000,k=4,x0=0.123456):
    x = np.zeros(length)
    x[0] = x0

    for t in range(1,length):
        x[t] = np.cos(k*np.arccos(x[t-1]))
    return x



data = logistic_data()

train_len = 100
test_len = 10

X_train = data[:train_len]
X_test = data[train_len:train_len+test_len]

y_train = data[1:train_len+1]
y_test = data[train_len+1:train_len+test_len+1]

scalar = MinMaxScaler(feature_range=(0,1))
X_train = scalar.fit_transform(X_train.reshape(-1,1))
X_test = scalar.transform(X_test.reshape(-1,1))
y_train = scalar.transform(y_train.reshape(-1,1))
y_test = y_test.flatten()

n_list = np.arange(5,20,1)
start = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
mse_list = []

for n in n_list:
    for s in start:
        model = nlrc(n=n,test_len=test_len,start=s)
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test[0].item())
        y_pred = scalar.inverse_transform(y_pred.reshape(-1,1)).flatten()
        mse = mean_squared_error(y_test,y_pred)
        print(f'{n} and {s} -----> {mse}')
        mse_list.append(mse)