import numpy as np
import random
from scipy import linalg
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class nlfea:
    def __init__(self,b=0.5,n=10,test_len=10,reg=1e-8,k=3,degree=2):
        self.b = b
        self.n = n
        self.test_len = test_len
        self.reg = reg
        self.k = k
        self.degree = degree
    
    def build_features(self,data,delay_X_train):
        k = self.k
        deg = self.degree
        self.poly = PolynomialFeatures(degree=deg)
        delay_buffer = delay_X_train.copy()
        dummy_lin = np.zeros((k+1))
        dummy_poly = self.poly.fit_transform(dummy_lin.reshape(1,-1))
        self.feat_size = dummy_poly.shape[1]

        history_x = np.zeros((self.feat_size,len(data)))
        for i in range(len(data)):
            lin = np.append(delay_buffer.flatten(),data[i])
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
            lin = np.append(delay_buffer.flatten(),u)
            poly_feat = self.poly.transform(lin.reshape(1,-1))
            y = np.dot(self.w_out,poly_feat.flatten())
            y = np.clip(y,-3,3)
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

data = mackey_glass()
train_len = 1000
test_len = 200
k = 3
p = 0
push = train_len+p+k

X_train = data[k:k+train_len]
delay_X_train = data[:k]
X_test = data[push:push+test_len]
delay_X_test = data[push-k:push]

y_train = data[k+1:k+train_len+1]
y_test = data[1+push:test_len+1+push]

scalar = StandardScaler()
X_train = scalar.fit_transform(X_train.reshape(-1,1))
delay_X_train = scalar.transform(delay_X_train.reshape(-1,1))
y_train = scalar.transform(y_train.reshape(-1,1))
X_test = scalar.transform(X_test.reshape(-1,1))
delay_X_test = scalar.transform(delay_X_test.reshape(-1,1))
y_test = y_test.flatten()

reg_lst = [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3]
mse_lst = {}
for reg in reg_lst:
    model = nlfea(test_len=test_len,degree=2,k=k,reg=reg)
    model.fit(X_train,y_train,delay_X_train)
    y_pred = model.predict(X_test[0],delay_X_test).T
    y_pred = scalar.inverse_transform(y_pred).flatten()

    print(f' reg {reg} mse {mean_squared_error(y_test,y_pred)}')
    mse_lst[(reg)] = mean_squared_error(y_test,y_pred)

best_5 = sorted(mse_lst.items(), key=lambda x: x[1])[:5]
print('------best_5----')
print(best_5)
    