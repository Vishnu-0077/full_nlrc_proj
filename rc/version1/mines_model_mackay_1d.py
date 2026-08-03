import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error
from scipy import linalg

data = pd.read_excel('/home/vishnu/Downloads/Mackey-Glass Time Series(taw17).xlsx')
data = data.drop('Unnamed: 0',axis = 1)

sub_data = data.iloc[200:1050]

X = sub_data[['t']].to_numpy()
y = sub_data['t+1'].to_numpy()
split = int(len(X)*0.8)

X_train = X[:split]
y_train = y[:split]
X_test = X[split:]
y_test = X[split:]

X.reshape(-1,1)

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
res_states = reservoir(X,n) #can make a graph of how the prediction changes with respect to the res column
print(f'shape of the input {X.shape}')
print(f'shape of the reservoir list put for training {res_states.shape}')
print(f'shape of the every single reservoir state {res_states[0].shape}')
X_train_states = res_states[:split]
model = train_W_out(X_train_states,y_train)

X_test_states = res_states[split:]
y_pred = model.predict(X_test_states)
print(f'root mean square error, {metric_check(y_test,y_pred)}')

def plot_checking(X,y,a,b): #this is for checking how does it behave with different values of n
    n = []
    per = []
    for i in range(100,5001,500):
        res_states = reservoir(X,a,b,i)
        X_train, X_test, y_train, y_test = train_test_split(res_states, y, test_size=0.2, random_state=42)
        models = train_W_out(X_train,y_train)
        y_pred = models.predict(X_test)
        per.append(metric_check(y_test,y_pred))
        n.append(i)
    plt.plot(n,per)
    plt.show()




    





