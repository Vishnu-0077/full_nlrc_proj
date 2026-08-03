import numpy as np
import random
from scipy import linalg
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class nlfea:
    def __init__(self,b=0.499,n=10,test_len=10,reg=1e-8,k=3,degree=2):
        self.b = b
        self.n = n
        self.test_len = test_len
        self.reg = reg
        self.k = k
        self.degree = degree
        self.insize = 3

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
    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def softsign(self,x):
        return 0.5*(x/(1+np.abs(x)) + 1)
    
    def atan01(self,x):
        return np.arctan(x)/np.pi + 0.5
    
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
        insize = self.insize
        delay_buffer = delay_X_train.copy()
        self.poly = PolynomialFeatures(degree=deg)
        dummy_lin = np.zeros(insize*3*(k+1))
        dummy_poly = self.poly.fit_transform(dummy_lin.reshape(1,-1))
        self.feat_size = dummy_poly.shape[1]

        history_x = np.zeros((self.feat_size,len(data)))
        for i in range(len(data)):
            lin = np.zeros(insize*3*(k+1))
            search_span = np.concatenate((delay_buffer.flatten(),data[i]))
            for j in range(len(search_span)):
                u = search_span[j]
                X = self.neuron_iterator(u)
                ene = self.energy(X)
                var = self.variance(X)
                lin[3*j+0] = u
                lin[3*j+1] = ene
                lin[3*j+2] = var
            delay_buffer = np.vstack((delay_buffer[1:],data[i]))
            poly_feat = self.poly.transform(lin.reshape(1,-1))
            history_x[:,i] = poly_feat.flatten()
        return history_x    
        
    
    def fit(self,data,y_data,delay_X_train):
        yt = y_data
        yt = yt.T
        reg = self.reg
        history_x = self.build_features(data,delay_X_train)
        self.w_out = np.dot(np.dot(yt,history_x.T),linalg.inv(np.dot(history_x,history_x.T)+np.eye(history_x.shape[0])*reg))
        return self
    
scalar = MinMaxScaler(feature_range=(0,1))
data = np.load('nl_rc_2/train_test_seperate/datasets/train/lorenz.npy')
train_len = 3000
k=3

X_train = data[k:k+train_len]
X_delay_train = data[:k]
y_train = data[k+1:k+train_len+1]


X_train = scalar.fit_transform(X_train)
X_delay_train = scalar.transform(X_delay_train)
y_train = scalar.transform(y_train)

model = nlfea(n=3,reg=1e-5,k=k,degree=2)
model.fit(X_train,y_train,X_delay_train)
w_out = model.w_out

import os
import joblib
os.makedirs('nl_rc_2/train_test_seperate/weights',exist_ok=True)
np.save('nl_rc_2/train_test_seperate/weights/w_out',w_out)
joblib.dump(scalar,'nl_rc_2/train_test_seperate/weights/scalar.joblib')
print('training_done')