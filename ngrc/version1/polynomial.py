import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.preprocessing import PolynomialFeatures
import math

dataa = pd.read_excel('/home/vishnu/Downloads/Mackey-Glass Time Series(taw17).xlsx')
dataa = dataa.drop('Unnamed: 0',axis = 1)
dataa = dataa[100:1050]


data = dataa['t'].to_numpy().reshape(-1,1)
y_data = dataa['t+1'].to_numpy()

mean = np.mean(data)
std = np.std(data)
y_mean = np.mean(y_data)
y_std = np.std(y_data)

data = (data - mean) / std
y_data = (y_data - y_mean) / y_std

k = 8
s = 1
deg = 2
reg = 1e-4

train_length = 700
test_length = 10

sample_data = np.zeros((1,k))
poly = PolynomialFeatures(degree=deg)
poly = poly.fit(sample_data) #we use the fit initaially so that can use only tranform latter on
poly_features = poly.transform(sample_data) #tranforming to sampple data to get the shape of the resuting one
poly_size = poly_features.shape[1] #get the shape 
'''
poly size is actually (k+deg)Cdeg
no need to before step and directly assign also
'''
total_features = np.zeros((poly_size,train_length-k)) #preset the shape

for t in range(k,train_length):
    O_lin = np.zeros((1,k))
    for i in range(k):
        O_lin[0,i] = (data[t-(i*s),0])
    
    poly_features = poly.transform(O_lin.reshape(1,-1)) #we use reshape(1,-1)... it does oppsite job of reshape(-1,1)
    total_features[:,t-k] = poly_features.flatten() #flattening it to make it as a vector
'''
there why are we using reshape(1,-1) it will make it into --> full columns and one row
we do that saying the poly that these are different features of the same data that u need to do comninations with 
using reshape(-1,1) like always, then it becomes all rows and one column, this will mean there is only one feature,
so the combinations won't work
'''

total_features_df = pd.DataFrame(total_features)

print(f'feature_shape is {total_features.shape}')
feature_len = total_features.shape[0]
yt = y_data[k:train_length]
yt = yt.reshape(-1,1).T

print(f'reshaped y which used to find w_out {yt.shape}')
w_out = np.dot(np.dot(yt,total_features.T),linalg.inv(np.dot(total_features,total_features.T)+np.eye(feature_len)*reg))
print(f'shape of w_out {w_out.shape}')

Y = np.zeros((1,test_length))

ui = 700
test_data = np.array(data[ui-k*s:ui])
for t in range(test_length):
    O_lin = np.zeros((1,k))
    for i in range(k):
        O_lin[0,i] = (test_data[(k-1)-(i*s),0])
    
    poly_features = poly.transform(O_lin.reshape(1,-1))
    
    x = poly_features.flatten()
    y = np.dot(w_out,x).reshape(-1,1)
    Y[:,t] = y
    test_data = np.vstack((test_data[1:],y))

y_data_orig = dataa['t+1'].to_numpy()
y_pred_unnormalized = Y * y_std + y_mean
sum=0
print(f'shape of Y is {Y.shape}')
for i in range(test_length):
    sum += (y_data_orig[ui+i]-y_pred_unnormalized[0,i])**2
print(f'MSE is {sum/test_length}')