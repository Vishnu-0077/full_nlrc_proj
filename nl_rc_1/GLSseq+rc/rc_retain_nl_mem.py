import numpy as np
from scipy import linalg
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def mackey_glass(length=2000, tau=17, beta=0.2, gamma=0.1, n=10, dt=1.0):
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

def tent_map(mu=1.9999, x0=0.5422, length=2000):
    x = np.zeros(length)
    x[0] = x0

    for t in range(length-1):
        if x[t] < 0.5:
            x[t+1] = mu*x[t]
        else:
            x[t+1] = mu*(1-x[t])

    return x

#------------------------------------
def neuron_gen(x):
        b = 0.5
        neigh = 1e-8
        x = np.clip(x,neigh,1-neigh)

        if x>=b:
            return (1-x)/(1-b)
        return x/b
    
def neuron_iterator(u,n):
    X = np.zeros(n)
    X[0] = u.item()
    for j in range(n-1):
        X[j+1] = neuron_gen(X[j])
    return X

#----------------------------------

data = mackey_glass()
y_data = data[1:]

n = 5
insize = n
outsize = 1
ressize = 100
init_len = 100
train_length = 1100
test_length = 10
a=0.3

np.random.seed(42)
W_in = np.random.rand(ressize,insize+1) - 0.5
W = np.random.rand(ressize,ressize) - 0.5

spectral_radius = np.max(np.abs(linalg.eigvals(W)))
target_radius = 0.95
W = W*target_radius/spectral_radius

x = np.zeros(ressize).reshape(-1,1)
x_history = np.zeros((1+1+ressize,train_length-init_len))

for i in range(train_length):
    u = data[i]
    ss = neuron_iterator(u,n).reshape(-1,1)
    x = x*(1-a)+a*np.tanh(np.dot(W,x)+np.dot(W_in,np.vstack((1,ss))))
    if i>=init_len:
        x_history[:,i-init_len] = np.vstack((1,u,x.reshape(-1,1)))[:,0]

print(f'shape of x_history is {x_history.shape}')
p_xhistory = pd.DataFrame(x_history)
print(p_xhistory.iloc[:10,:10])

yt = y_data[init_len:train_length].reshape(-1,1).T
reg = 1e-8
w_out = np.dot(np.dot(yt,x_history.T),linalg.inv(np.dot(x_history,x_history.T)+np.eye(ressize+1+1)*reg))
Y = np.zeros((outsize,test_length))

print(f'shape of w_out is {w_out.shape}')

u = data[train_length]
for i in range(test_length):
    ss = neuron_iterator(u,n).reshape(-1,1)
    x = x*(1-a)+a*np.tanh(np.dot(W,x)+np.dot(W_in,np.vstack((1,ss))))
    y = np.dot(w_out,np.vstack((1,u,x.reshape(-1,1 ))))
    Y[:,i] = y
    u = y

sum=0
print(f'shape of Y is {Y.shape}')
for i in range(test_length):
    sum += (y_data[train_length+i]-Y[0,i])**2
print(sum/test_length)
