import numpy as np
from scipy import linalg
import pandas as pd

dataa = pd.read_excel('/home/vishnu/Downloads/Mackey-Glass Time Series(taw17).xlsx')
dataa = dataa.drop('Unnamed: 0',axis = 1)
dataa = dataa[100:1050]

data = dataa['t'].to_numpy().reshape(-1,1)
data = np.hstack((data,np.vstack((np.zeros((1,1)),data[:-1])),np.vstack((np.zeros((2,1)),data[:-2]))))
y_data = dataa['t+1'].to_numpy()

p_data = pd.DataFrame(data)
print(p_data[:10])

insize = 3
outsize = 1
ressize = 100
init_len = 50
train_length = 800
test_length = 100
a=0.3

np.random.seed(42)
W_in = np.random.rand(ressize,insize+1) - 0.5
W = np.random.rand(ressize,ressize) - 0.5

spectral_radius = np.max(np.abs(linalg.eigvals(W)))
target_radius = 0.95
W = W*target_radius/spectral_radius

x = np.zeros(ressize).reshape(-1,1)
x_history = np.zeros((1+insize+ressize,train_length-init_len))

for i in range(train_length):
    u = data[i].reshape(-1,1)
    x = x*(1-a)+a*np.tanh(np.dot(W,x)+np.dot(W_in,np.vstack((1,u))))
    if i>=init_len:
        x_history[:,i-init_len] = np.vstack((1,u,x.reshape(-1,1)))[:,0]

yt = y_data[init_len:train_length].reshape(-1,1).T
reg = 1e-8
w_out = np.dot(np.dot(yt,x_history.T),linalg.inv(np.dot(x_history,x_history.T)+np.eye(ressize+insize+1)*reg))
Y = np.zeros((outsize,test_length))

u = data[train_length].reshape(-1,1)
for i in range(test_length):
    x = x*(1-a)+a*np.tanh(np.dot(W,x)+np.dot(W_in,np.vstack((1,u))))
    y = np.dot(w_out,np.vstack((1,u,x.reshape(-1,1 ))))
    Y[:,i] = y
    u = np.vstack((y,u[:-1]))

sum=0
for i in range(test_length):
    sum += (y_data[train_length+i]-Y[0,i])**2
print(sum/test_length)



