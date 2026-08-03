import numpy as np
import random
from scipy import linalg
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class nlfea:
    def __init__(self,b=0.499,n=10,test_len=10,reg=1e-8,k=3,degree=3):
        self.b = b
        self.n = n
        self.test_len = test_len
        self.reg = reg
        self.k = k
        self.degree = degree
        self.insize = 3
        self.w_out = None

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
    
    def predict(self,u,delay_X_test):
        test_len = self.test_len
        k = self.k
        insize = self.insize
        deg = self.degree
        self.poly = PolynomialFeatures(degree=deg)
        dummy_lin = np.zeros(insize*3*(k+1))
        dummy_poly = self.poly.fit_transform(dummy_lin.reshape(1,-1))
        Y = np.zeros((3,test_len))
        delay_buffer = delay_X_test.copy()
        for i in range(test_len):
            lin = np.zeros(insize*3*(k+1))
            search_span = np.concatenate((delay_buffer.flatten(),u))
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
            delay_buffer = np.vstack((delay_buffer[1:],u))
            u = y

        return Y

import joblib
scalar = joblib.load('nl_rc_2/train_test_seperate/weights/scalar.joblib')
data = np.load('nl_rc_2/train_test_seperate/datasets/test/lorenz.npy')
train_len = 3000
test_len = 100
k=3

X_test = data[k:k+test_len]
X_delay_test = data[:k]
y_test = data[k+1:k+test_len+1]

X_test = scalar.transform(X_test)
X_delay_test = scalar.transform(X_delay_test)

model = nlfea(test_len=test_len,degree=2,k=k,n=3,reg=1e-5)
model.w_out = np.load('nl_rc_2/train_test_seperate/weights/w_out.npy')
y_pred = model.predict(X_test[0],X_delay_test).T
y_pred = scalar.inverse_transform(y_pred)

print(f'shape of Y is {y_pred.shape}')
print(f'mse of x is {mean_squared_error(y_test[:,0],y_pred[:,0])}')
print(f'mse of y is {mean_squared_error(y_test[:,1],y_pred[:,1])}')
print(f'mse of z is {mean_squared_error(y_test[:,2],y_pred[:,2])}')
print(f'mse of all is {mean_squared_error(y_test,y_pred)}')

plt.plot(np.arange(len(y_test)),y_test,c='r',label='real')
plt.plot(np.arange(len(y_pred)),y_pred,c='b',label='predicted')
plt.legend()
plt.show()

