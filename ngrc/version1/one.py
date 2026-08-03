import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import math

dataa = pd.read_excel('/home/vishnu/Downloads/Mackey-Glass Time Series(taw17).xlsx')
dataa = dataa.drop('Unnamed: 0',axis = 1)
dataa = dataa[100:1050]

data = dataa['t'].to_numpy().reshape(-1,1)
y_data = dataa['t+1'].to_numpy()

k = 10
s = 1

train_length = 700
test_length = 10

total_features = np.zeros((math.comb(k+2,2),train_length-k))
for t in range(k,train_length):
    O_lin = np.zeros((k,1))
    non_size = math.comb(k,2)+k #or (k+1)C2 for mula
    O_nonlin = np.zeros((non_size,1))
    for i in range(k):
        O_lin[i,0] = (data[t-(i*s),0])
    
    p=0
    for i in range(len(O_lin)):
        O_nonlin[p,0] = O_lin[i,0]*O_lin[i,0]
        p+=1

    for i in range(len(O_lin)):
        for j in range(i+1,len(O_lin)):
            O_nonlin[p,0] = O_lin[i,0]*O_lin[j,0]
            p+=1
    total_features[:,t-k] = np.vstack((1,O_lin,O_nonlin))[:,0] #vstacking the functions and getting it as vector

#show casing how all went 
data_feature_df = pd.DataFrame(total_features)
#print(data_feature_df)
print(f'feature_shape is {total_features.shape}')

#calculation of the w_out
feature_len = total_features.shape[0]
yt = y_data[k:train_length]
yt = yt.reshape(-1,1).T

print(f'reshaped y which used to find w_out {yt.shape}')
reg = 1e-8
w_out = np.dot(np.dot(yt,total_features.T),linalg.inv(np.dot(total_features,total_features.T)+np.eye(feature_len)*reg))
print(f'shape of w_out {w_out.shape}')

Y = np.zeros((1,test_length))

ui = 800
test_data = np.array(data[ui-k*s:ui])
for t in range(test_length):
    O_lin = np.zeros((k,1))
    non_size = math.comb(k,2)+k #non size can also write as (k+1)C2
    O_nonlin = np.zeros((non_size,1))
    for i in range(k):
        O_lin[i,0] = (test_data[(k-1)-(i*s),0]) #assume it as 2d, as we did the reshapping initially
    
    p=0
    for i in range(len(O_lin)):
        O_nonlin[p,0] = O_lin[i,0]*O_lin[i,0]
        p+=1

    for i in range(len(O_lin)):
        for j in range(i+1,len(O_lin)):
            O_nonlin[p,0] = O_lin[i,0]*O_lin[j,0]
            p+=1
    
    x = np.vstack((1,O_lin,O_nonlin))[:,0]
    y = np.dot(w_out,x).reshape(-1,1)
    Y[:,t] = y
    test_data = np.vstack((test_data[1:],y))

sum=0
print(f'shape of Y is {Y.shape}')
for i in range(test_length):
    sum += (y_data[ui+i]-Y[0,i])**2
print(f'mean_squared_error is {sum/test_length}')