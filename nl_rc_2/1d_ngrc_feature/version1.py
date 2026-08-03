import numpy as np
import random
from scipy import linalg
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class nlfea:
    def __init__(self,b=0.499,n=10,test_len=10,reg=1e-8,k=3,degree=2):
        self.b = b
        self.n = n
        self.test_len = test_len
        self.reg = reg
        self.k = k
        self.degree = degree

    def ss_to_binary(self,x,thres):
        return (np.array(x)>thres).astype(int)

    def firing_rate(self,x):
        b = self.b
        c = 0
        for v in x:
            if v>b:
                c+=1
        return c/len(x)
    
    def variance(self,x):
        if len(x) ==0:
            return 0
        return np.var(x)
    
    def energy(self,x):
        if len(x) ==0:
            return 0
        return np.mean(np.square(x))
    
    def entropy(self,x):
        b = self.b
        if len(x) ==0:
            return 0
        x = self.ss_to_binary(x,b)
        p = np.count_nonzero(x)/len(x)
        eps = 1e-10
        p = np.clip(p,eps,1-eps)
        return -(p*np.log2(p)) - ((1-p)*np.log2(1-p))
    
    def neuron_gen(self,x):
        b = self.b
        neigh = 1e-8
        x = np.clip(x,neigh,1-neigh)

        if x>=b:
            return (1-x)/(1-b)
        return x/b
    
    def neuron_iterator(self,u):
        n = self.n
        X = np.zeros(n)
        X[0] = u.item()
        for j in range(n-1):
            X[j+1] = self.neuron_gen(X[j])
        return X
    
    def build_features(self,data,delay_X_train):
        k = self.k
        deg = self.degree
        self.poly = PolynomialFeatures(degree=deg)
        delay_buffer = delay_X_train.copy()
        dummy_lin = np.zeros(3*(k+1))
        dummy_poly = self.poly.fit_transform(dummy_lin.reshape(1,-1))
        self.feat_size = dummy_poly.shape[1]

        history_x = np.zeros((self.feat_size,len(data)))
        for i in range(len(data)):
            lin = np.zeros(3*(k+1))
            search_span = np.append(delay_buffer.flatten(),data[i])
            for j in range(len(search_span)):
                u = search_span[j]
                X = self.neuron_iterator(u)
                ene = self.energy(X)
                var = self.variance(X)
                lin[3*j+0] = u
                lin[3*j+1] = ene
                lin[3*j+2] = var
            delay_buffer = np.append(delay_buffer[1:],data[i]).reshape(-1,1)
            poly_feat = self.poly.transform(lin.reshape(1,-1))
            history_x[:,i] = poly_feat.flatten()
        return history_x    
        
    
    def fit(self,data,y_data,delay_X_train):
        yt = y_data.reshape(-1,1)
        yt = yt.T
        reg = self.reg
        history_x = self.build_features(data,delay_X_train)
        self.w_out = np.dot(np.dot(yt,history_x.T),linalg.inv(np.dot(history_x,history_x.T)+np.eye(history_x.shape[0])*reg))
        return self
    
    def predict(self,u,delay_X_test):
        test_len = self.test_len
        k = self.k
        Y = np.zeros((1,test_len))
        delay_buffer = delay_X_test.copy()
        for i in range(test_len):
            lin = np.zeros(3*(k+1))
            search_span = np.append(delay_buffer.flatten(),u)
            for j in range(len(search_span)):
                uj = search_span[j]
                X = self.neuron_iterator(uj)
                ene = self.energy(X)
                var = self.variance(X)
                lin[3*j+0] = uj
                lin[3*j+1] = ene
                lin[3*j+2] = var

            poly_feat = self.poly.transform(lin.reshape(1,-1))
            y = np.dot(self.w_out,poly_feat.flatten())
            y = np.clip(y,0,1)
            Y[:, i] = y
            delay_buffer = np.append(delay_buffer[1:],u).reshape(-1,1)
            u = y.item()

        return Y



def logistic_data(a = 3.95, i = 0.5422,length = 10000):
    x = np.zeros(length)
    x[0] = i
    for t in range(length-1):
        x[t+1] = a*x[t]*(1-x[t])
    return x

def mackey_glass(length=10000, tau=17, beta=0.2, gamma=0.1, n=10, dt=1.0):
    x = np.zeros(length)
    
    # initial condition (important!)
    x[:tau+1] = 1.2  
    
    for t in range(tau, length-1):
        x_tau = x[t - tau]
        x[t+1] = x[t] + dt * (beta * x_tau / (1 + x_tau**n) - gamma * x[t])
    
    return x[100:]

def tent_map(mu=1.9999, x0=0.5422, length=2000):
    x = np.zeros(length)
    x[0] = x0

    for t in range(length-1):
        if x[t] < 0.5:
            x[t+1] = mu*x[t]
        else:
            x[t+1] = mu*(1-x[t])

    return x

def chebyshev_map(length=1500,k=4,x0=0.123456):
    x = np.zeros(length)
    x[0] = x0

    for t in range(1,length):
        x[t] = np.cos(k*np.arccos(x[t-1]))
    return x

data = logistic_data()
train_len = 1000
test_len = 10
k = 3
p = 0
push = train_len+p+k

X_train = data[k:k+train_len]
delay_X_train = data[:k]
X_test = data[push:push+test_len]
delay_X_test = data[push-k:push]

y_train = data[k+1:k+train_len+1]
y_test = data[1+push:test_len+1+push]

scalar = MinMaxScaler(feature_range=(0,1))
X_train = scalar.fit_transform(X_train.reshape(-1,1))
delay_X_train = scalar.transform(delay_X_train.reshape(-1,1))
y_train = scalar.transform(y_train.reshape(-1,1))
X_test = scalar.transform(X_test.reshape(-1,1))
delay_X_test = scalar.transform(delay_X_test.reshape(-1,1))
y_test = y_test.flatten()


model = nlfea(test_len=test_len,n=12,degree=2,k=k,reg=1e-6)
model.fit(X_train,y_train,delay_X_train)
y_pred = model.predict(X_test[0].item(),delay_X_test)
y_pred = scalar.inverse_transform(y_pred.reshape(-1,1)).flatten()
mse = mean_squared_error(y_test,y_pred)
print(f'mse is {mse}')

plt.plot(y_test,label='actual')
plt.plot(y_pred,label='predicted')
plt.legend()
plt.show()