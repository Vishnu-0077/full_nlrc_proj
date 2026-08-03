import numpy as np
import random
from scipy import linalg
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

'''
what does this code does??

1. it uses the gls neuron binary trace values as the feature extractor of the time series input of
of the previous data.
2. we do not keep a neighbour hood region instead we have a specific number of iterations that the
gls neuron can take
3. the made feature training matrix is used for the calculation of the w_out
4. next the name w_out is used for the calcualtion of the recursive value of the y,
(recusrive mean, the previous output is taken as the next input)
5. done. the remaining features are then quotedd in the code'''

class nlrc:
    def __init__(self,a=0.3,b=0.5,reg=1e-8,n=10,test_len = 10,start = 0.2,eps = 0.05):
        self.b = b
        self.reg = reg
        self.n = n
        self.test_len = test_len
        self.a = a
        self.start = start
        self.eps = eps

    def gls_neuron_gen(self,x):
        """
        the gls neuron binary trace values as the feature extractor of the time series input of
        of the previous data.
        """
        b = self.b
        neigh = 1e-8
        x = np.clip(x,neigh,1-neigh)

        if x>=b:
            return (1-x)/(1-b)
        return x/b 
    
    def build_features(self,data):
        '''
        building the features of the specific time point using the gls neuron binary trace values
        '''
        n = self.n
        start = self.start
        eps = self.eps
        history_x = np.zeros((1+n,len(data)))
        max_iter = 10000
        for i in range(len(data)):
            u = data[i] #setting the first value
            X = []
            X.append(start) #item becoz, u might be ---> array[u] instead of float(u)
            while (X[-1] > u+eps or X[-1] < u-eps) and len(X)<max_iter:
                X.append(self.gls_neuron_gen(X[-1]))
            if len(X) > n:
                X = X[-n:]
            else:
                while len(X) < n:
                    X.insert(0,0)
            X = np.array(X)
            history_x[:,i] = np.vstack((1,X.reshape(-1,1))).flatten()
        
        return history_x #printing the it with shape(n+1,train_len)
    
    def memory_constructor(self,history_x):
        n = self.n
        x = np.zeros((1+n,len(history_x[0])))
        x[:,0] = history_x[:,0]
        a = self.a

        for i in range(1,history_x.shape[1]):
            x[:,i] = a*x[:,i-1]+(1-a)*history_x[:,i]
        self.final_x = x[1:,-1].flatten()
        return x
    
    def memory_constructor_pred(self,X):
        a = self.a
        X = (1-a)*X.flatten() + a*self.final_x
        self.final_x = X.flatten()
        return X
    
    def fit(self,data,y_data):
        yt = y_data.reshape(-1,1)
        yt = yt.T
        reg = self.reg
        history_x = self.build_features(data)
        history_x = self.memory_constructor(history_x)
        self.w_out = np.dot(np.dot(yt,history_x.T),linalg.inv(np.dot(history_x,history_x.T)+np.eye(self.n+1)*reg))
        return self
    
    def predict(self,test_u):
        test_len = self.test_len
        n = self.n
        start = self.start
        eps = self.eps
        Y = np.zeros((1,test_len))
        u = test_u
        for i in range(test_len):
            X = []
            X.append(start)
            while (X[-1] > u+eps or X[-1] < u-eps) and len(X)<1000:
                X.append(self.gls_neuron_gen(X[-1]))
            if len(X) > n:
                X = X[-n:]
            else:
                while len(X) < n:
                    X.insert(0,0)
            X = np.array(X)
            X = self.memory_constructor_pred(X)
            y = np.dot(self.w_out,np.vstack((1,X.reshape(-1,1))).flatten())
            Y[:,i] = y
            u = y.item() #y is an array, so we send only the float values further
        return Y.flatten()

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



data = logistic_data()
train_len = 1000
test_len = 10

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


model = nlrc(test_len=test_len,a=0.3)
model.fit(train_X,train_y)
y_pred = model.predict(test_X[0].item())
y_pred = scalar.inverse_transform(y_pred.reshape(-1,1)).flatten()

mse = mean_squared_error(test_y,y_pred)
print(f'mse is {mse}')

plt.plot(test_y)
plt.plot(y_pred)
plt.legend(['original','predicted'])
print(model.build_features(train_X).shape)
print(model.memory_constructor(model.build_features(train_X)).shape)
