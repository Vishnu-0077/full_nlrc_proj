import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.base import BaseEstimator,RegressorMixin
import math

class NGRC():
    def __init__(self,k=4,deg=3,reg=10e-7,test_len=10):
        self.k = k
        self.deg = deg
        self.reg = reg
        self.test_len = test_len
    
    def build_features(self,data,delay_X_train):
        k = self.k
        delay_buffer = delay_X_train.copy()
        self.poly = PolynomialFeatures(degree=self.deg)
        dummy = np.zeros(k+1)
        dummy_features = self.poly.fit_transform(dummy.reshape(1,-1))
        total_featues = np.zeros((dummy_features.shape[1],len(data)))

        for t in range(len(data)):
            lin = np.concatenate((delay_buffer.flatten(),data[t]))
            poly_features = self.poly.transform(lin.reshape(1,-1))
            delay_buffer = np.append(delay_buffer[1:],data[t]).reshape(-1,1)
            total_featues[:,t] = poly_features.flatten()
        return total_featues
    
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
            lin = np.append(delay_buffer.flatten(),u)
            poly_feat = self.poly.transform(lin.reshape(1,-1))
            y = np.dot(self.w_out,poly_feat.flatten())
            y = np.clip(y,-3,3)
            Y[:, i] = y
            delay_buffer = np.append(delay_buffer[1:],u).reshape(-1,1)
            u = y.item()

        return Y
    

def mackey_glass(length=3000, tau=17, beta=0.2, gamma=0.1, n=10, dt=1.0):
    x = np.zeros(length)
    
    # initial condition (important!)
    x[:tau+1] = 1.2  
    
    for t in range(tau, length-1):
        x_tau = x[t - tau]
        x[t+1] = x[t] + dt * (beta * x_tau / (1 + x_tau**n) - gamma * x[t])
    
    return x

def logistic_data(a = 3.95, i = 0.5422,length = 2000):
    x = np.zeros(length)
    x[0] = i
    for t in range(length-1):
        x[t+1] = a*x[t]*(1-x[t])
    return x[100:]

data = logistic_data()
scalar = StandardScaler()
train_len = 100
test_len = 50

k = 3
p = 0
push = train_len+p+k

X_train = data[k:k+train_len]
y_train = data[k+1:k+train_len+1]
delay_X_train = data[:k]

X_test = data[push:push+test_len]
y_test = data[1+push:test_len+1+push]
delay_X_test = data[push-k:push]

X_train = scalar.fit_transform(X_train.reshape(-1,1))
y_train = scalar.transform(y_train.reshape(-1,1))
delay_X_train = scalar.transform(delay_X_train.reshape(-1,1))
X_test = scalar.transform(X_test.reshape(-1,1))
delay_X_test = scalar.transform(delay_X_test.reshape(-1,1))
y_test = y_test.flatten()

model = NGRC(test_len=test_len,deg=2,k=k,reg=1e-6)
model.fit(X_train,y_train,delay_X_train)
y_pred = model.predict(X_test[0].item(),delay_X_test)
y_pred = scalar.inverse_transform(y_pred.reshape(-1,1)).flatten()

print(np.mean((y_pred-y_test)**2))

plt.plot(y_pred,label = 'pred')
plt.plot(y_test,label = 'actual')
plt.show()


    