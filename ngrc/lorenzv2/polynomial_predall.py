import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.preprocessing import PolynomialFeatures
import math

dataa = pd.read_csv('/home/vishnu/Downloads/traindata.csv')
dataa = dataa[1000:65000]


data = dataa.to_numpy()
y_data_x = dataa.iloc[1: ,0].to_numpy() #here we use iloc because.....we get only the first column
y_data_y = dataa.iloc[1: ,1].to_numpy()
y_data_z = dataa.iloc[1: ,2].to_numpy()

mean = np.mean(data,axis = 0) #----> this  given [mean_x,mean_y,mean_z]
std = np.std(data,axis = 0)

data = (data - mean) / std
y_data_x = (y_data_x - mean[0]) / std[0] #we only take mean_x
y_data_y = (y_data_y - mean[1]) / std[1]
y_data_z = (y_data_z - mean[2]) / std[2]

dk = 8 #delay k
k = dk*3
s = 1
deg = 4

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

#w_out section down come here

#----------------------------------------------------------------------------
print(f'feature_shape is {total_features.shape}')
feature_len = total_features.shape[0]
yt_x = y_data_x[dk:train_length]
yt_x = yt_x.reshape(-1,1).T

yt_y = y_data_y[dk:train_length]
yt_y = yt_y.reshape(-1,1).T

yt_z = y_data_z[dk:train_length]
yt_z = yt_z.reshape(-1,1).T

print(f'reshaped y which used to find w_out of x {yt_x.shape}')
print(f'reshaped y which used to find w_out of y {yt_y.shape}')
print(f'reshaped y which used to find w_out of z {yt_z.shape}')

reg = 1e-4
w_out_x = np.dot(np.dot(yt_x,total_features.T),linalg.inv(np.dot(total_features,total_features.T)+np.eye(feature_len)*reg))
print(f'shape of w_out of x {w_out_x.shape}')

w_out_y = np.dot(np.dot(yt_y,total_features.T),linalg.inv(np.dot(total_features,total_features.T)+np.eye(feature_len)*reg))
print(f'shape of w_out of y {w_out_y.shape}')

w_out_z = np.dot(np.dot(yt_z,total_features.T),linalg.inv(np.dot(total_features,total_features.T)+np.eye(feature_len)*reg))
print(f'shape of w_out of z {w_out_z.shape}')

#--------------------------------------------------------------------------------------
Y_x = np.zeros((1,test_length))
Y_y = np.zeros((1,test_length))
Y_z = np.zeros((1,test_length))

ui = 10001
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
    y_x = np.dot(w_out_x,x).reshape(-1,1)
    y_y = np.dot(w_out_y,x).reshape(-1,1)
    y_z = np.dot(w_out_z,x).reshape(-1,1)
    Y_x[:,t] = y_x
    Y_y[:,t] = y_y
    Y_z[:,t] = y_z
    nxt_row = np.array([y_x[0,0],y_y[0,0],y_z[0,0]])
    test_data = np.vstack((test_data[1:],nxt_row))
    

y_data_orig_x = dataa.iloc[1: ,0].to_numpy()
y_data_orig_y = dataa.iloc[1: ,1].to_numpy()
y_data_orig_z = dataa.iloc[1: ,2].to_numpy()

y_pred_unnormalized_x = Y_x * std[0] + mean[0]
y_pred_unnormalized_y = Y_y * std[1] + mean[1]
y_pred_unnormalized_z = Y_z * std[2] + mean[2]

mse_x = 0
mse_y = 0
mse_z = 0

for i in range(test_length):
    mse_x += (y_data_orig_x[ui+i] - y_pred_unnormalized_x[0,i])**2
    mse_y += (y_data_orig_y[ui+i] - y_pred_unnormalized_y[0,i])**2
    mse_z += (y_data_orig_z[ui+i] - y_pred_unnormalized_z[0,i])**2

mse_x /= test_length
mse_y /= test_length
mse_z /= test_length

print("MSE_x:", mse_x)
print("MSE_y:", mse_y)
print("MSE_z:", mse_z)

print(f'total avg mse is {(mse_x+mse_y+mse_z)/3}')