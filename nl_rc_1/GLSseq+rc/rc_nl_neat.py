import numpy as np
from scipy import linalg
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

class rc_nl:

    def __init__(self,n=5,ressize=100,spectral_radius=0.95,leak_rate=0.3,reg=1e-8,test_len=10,init_len=100,seed=42):
        self.n = n
        self.ressize = ressize
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.reg = reg
        self.test_len = test_len
        self.init_len = init_len
        self.seed = seed
        np.random.seed(seed)

        self.W_in = np.random.rand(ressize,n+1) - 0.5
        self.W = np.random.rand(ressize,ressize) - 0.5
        self.x = np.zeros(ressize).reshape(-1,1)

        radius = np.max(np.abs(linalg.eigvals(self.W)))
        self.W = self.W * spectral_radius / radius

    def neuron_gen(self, x):
        b = 0.5
        neigh = 1e-8
        x = np.clip(x, neigh, 1 - neigh)
        if x >= b:
            return (1-x)/(1-b)
        return x/b

    def neuron_iterator(self,u):
        X = np.zeros(self.n)
        X[0] = u
        for i in range(self.n-1):
            X[i+1] = self.neuron_gen(X[i])
        return X

    def build_features(self,data):
        train_length = len(data)
        init_len = self.init_len
        self.x_history = np.zeros((self.ressize+2,train_length-init_len))
        a = self.leak_rate
        W = self.W
        W_in = self.W_in
        for i in range(train_length):
            u = data[i].item()
            ss = self.neuron_iterator(u).reshape(-1, 1)
            self.x = self.x*(1-a)+a*np.tanh(np.dot(W,self.x)+np.dot(W_in,np.vstack((1,ss))))
            if i>=init_len:
                self.x_history[:,i-init_len] = np.vstack((1,u,self.x.reshape(-1,1)))[:,0]

        return self.x_history

    def fit(self,data,y_data):
        yt = y_data[self.init_len:].reshape(-1,1)
        yt = yt.T
        reg = self.reg
        history_x = self.build_features(data)
        self.w_out = np.dot(np.dot(yt,history_x.T),linalg.inv(np.dot(history_x,history_x.T)+np.eye(self.ressize+2)*reg))
        return self

    def predict(self,u):
        Y = np.zeros((1,self.test_len))
        x = self.x.copy()
        a = self.leak_rate
        W = self.W
        W_in = self.W_in
        for i in range(self.test_len):
            ss = self.neuron_iterator(u).reshape(-1, 1)
            x = x*(1-a)+a*np.tanh(np.dot(W,x)+np.dot(W_in,np.vstack((1,ss))))
            y = np.dot(self.w_out,np.vstack((1,u,x.reshape(-1,1 ))))
            Y[:,i] = y
            u = y.item()
        return Y.flatten()
    
def logistic_data(a = 3.95, i = 0.5422,length = 2000):
    x = np.zeros(length)
    x[0] = i
    for t in range(length-1):
        x[t+1] = a*x[t]*(1-x[t])
    return x

def mackey_glass(length=3000, tau=17, beta=0.2, gamma=0.1, n=10, dt=1.0):
    x = np.zeros(length)
    
    # initial condition (important!)
    x[:tau+1] = 1.2  
    
    for t in range(tau, length-1):
        x_tau = x[t - tau]
        x[t+1] = x[t] + dt * (beta * x_tau / (1 + x_tau**n) - gamma * x[t])
    
    return x

def tent_map(mu=1.9999, x0=0.5422, length=2000):
    x = np.zeros(length)
    x[0] = x0

    for t in range(length-1):
        if x[t] < 0.5:
            x[t+1] = mu*x[t]
        else:
            x[t+1] = mu*(1-x[t])

    return x

def chebyshev_map(length=1000,k=4,x0=0.123456):
    x = np.zeros(length)
    x[0] = x0

    for t in range(1,length):
        x[t] = np.cos(k*np.arccos(x[t-1]))
    return x

data = logistic_data()
train_len = 1100
test_len = 30

p = 0 #distance between and of training data and start of test data
push = train_len+p #for easy indexing

train_X = data[:train_len] #getting the train len
test_X = data[push:push+test_len]


train_y = data[1:train_len+1]
test_y = data[push+1:push+test_len+1]

scalar = MinMaxScaler(feature_range=(0,1)) #scaling between the 0-1
train_X = scalar.fit_transform(train_X.reshape(-1,1))
train_y = scalar.transform(train_y.reshape(-1,1))
test_X = scalar.transform(test_X.reshape(-1,1))
test_y = test_y.flatten()


model = rc_nl(test_len=test_len,n=4,ressize=500,reg=1e-8)
model.fit(train_X,train_y)
y_pred = model.predict(test_X[0].item())
y_pred = scalar.inverse_transform(y_pred.reshape(-1,1)).flatten()
test_X = scalar.inverse_transform(test_X.reshape(-1,1)).flatten()

mse = mean_squared_error(test_y,y_pred)
print(f'mse is {mse}')
plt.plot(test_y,label='original')
plt.plot(y_pred,label='predicted')
plt.legend(['original','predicted'])
plt.show()