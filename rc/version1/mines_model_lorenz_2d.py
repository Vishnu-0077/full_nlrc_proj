import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error
from scipy import linalg

data = pd.read_csv('/home/vishnu/Downloads/traindata.csv')
data_test = pd.read_csv('/home/vishnu/Downloads/testdata.csv')
data = data[1000:65000]

X = data[:-1]
X_train = X.to_numpy()
print(X.shape)
y = data.iloc[1: ,0]
y_train = y.to_numpy()
print(y.shape)

for i in range(10):
    print(f'{X_train[i]} is mapped to {y_train[i]}')

X_test = data_test[:-1]
X_test = X_test.to_numpy()
y_test = data_test.iloc[1: ,0]
y_test = y_test.to_numpy()

X_total = np.vstack((X_train,X_test))

def reservoir(u,n): #n = size of the reservoir
    (a,b) = u.shape
    x = np.zeros(n)
    W_in = np.random.rand(n,b) - 0.5
    W = np.random.rand(n,n) - 0.5 # '-0.5' will center the values in the matrix
    res_states = np.zeros((a,n))

    '''
    we used echo state property, *time goes memory fades*
    spectral radius is the largest eigenvalue of W
    we use that and normalize it to 0.95
    '''

    spectral_radius = np.max(np.abs(linalg.eigvals(W)))
    target_radius = 0.95
    W = W * target_radius / spectral_radius

    for i in range(a):
        x = np.tanh(np.dot(W, x) + np.dot(W_in, u[i,:]))
        res_states[i] = x
    return res_states


def train_W_out(X_train,y_train):
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('ridge', Ridge())])
    pipe.fit(X_train, y_train)
    return pipe

def metric_check(y_test,y_pred):
    return root_mean_squared_error(y_test,y_pred)


n=500
np.random.seed(42) #by setting this the random values generated is same even after we run it many times
res_states = reservoir(X_total,n) #can make a graph of how the prediction changes with respect to the res column
print(f'shape of the input {X_total.shape}')
print(f'shape of the reservoir that is made {res_states.shape}')
print(f'shape of the every single reservoir state {res_states[0].shape}')
X_train_states = res_states[:63999]
print(f'shape of the reservoir that is trained is: {X_train_states.shape}')
model = train_W_out(res_states[:63999],y_train)

X_test_states = res_states[63999:]
print(f'shape of the reservoir that is tested is: {X_test_states.shape}')
y_pred = model.predict(X_test_states)
print(f'root mean square error, {metric_check(y_test,y_pred)}')
