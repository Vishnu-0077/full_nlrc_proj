import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.preprocessing import PolynomialFeatures
import math

dataa = pd.read_csv('/home/vishnu/Downloads/traindata.csv')
dataa = dataa[1000:65000]


data = dataa.to_numpy()
y_data = dataa.iloc[1: ,0].to_numpy() #here we use iloc because.....we get only the first column

mean = np.mean(data,axis = 0) #----> this  given [mean_x,mean_y,mean_z]
std = np.std(data,axis = 0)

data = (data - mean) / std
y_data = (y_data - mean[0]) / std[0] #we only take mean_x

dk = 8 #delay k
k = dk*3
s = 1
deg = 3

train_length = 2000
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
total_features = np.zeros((poly_size,train_length-dk)) #preset the shape

for t in range(dk,train_length):
    O_lin = np.zeros((1,k))
    c=0
    for i in range(dk):
        for j in range(3):
            O_lin[0,c] = (data[t-(i*s),j]) #used the get the values properly and to fit it
            c+=1
    
    poly_features = poly.transform(O_lin.reshape(1,-1)) #we use reshape(1,-1)... it does oppsite job of reshape(-1,1)
    total_features[:,t-dk] = poly_features.flatten() #flattening it to make it as a vector
'''
there why are we using reshape(1,-1) it will make it into --> full columns and one row
we do that saying the poly that these are different features of the same data that u need to do comninations with 
using reshape(-1,1) like always, then it becomes all rows and one column, this will mean there is only one feature,
so the combinations won't work
'''

total_features_df = pd.DataFrame(total_features)
#----------------------------------------------------------------
print(f'feature_shape is {total_features.shape}')
feature_len = total_features.shape[0]
yt = y_data[dk:train_length] #changed what we get as it may affect the future
yt = yt.reshape(-1,1).T

print(f'reshaped y which used to find w_out {yt.shape}')
reg = 1e-4
w_out = np.dot(np.dot(yt,total_features.T),linalg.inv(np.dot(total_features,total_features.T)+np.eye(feature_len)*reg))
print(f'shape of w_out {w_out.shape}')
#-----------------------------------------------------------------
Y = np.zeros((1,test_length))

ui = 2001
test_data = np.array(data[ui-dk*s:ui,:])
for t in range(test_length):
    O_lin = np.zeros((1,k))
    c=0
    for i in range(dk):
        for j in range(3):
            O_lin[0,c] = (test_data[(dk-1)-(i*s),j])
            c+=1
    
    poly_features = poly.transform(O_lin.reshape(1,-1))
    
    x = poly_features.flatten()
    y = np.dot(w_out,x).reshape(-1,1)
    Y[:,t] = y
    nxt_row = np.array([y[0,0],data[ui+t,1],data[ui+t,2]]) #we used y[0,0] because y was array[[value]]
    test_data = np.vstack((test_data[1:],nxt_row))

y_data_orig = dataa.iloc[1: ,0].to_numpy()
y_pred_unnormalized = Y * std[0] + mean[0]
sum=0
print(f'shape of Y is {Y.shape}')
for i in range(test_length):
    sum += (y_data_orig[ui+i]-y_pred_unnormalized[0,i])**2
print(f'MSE is {sum/test_length}')