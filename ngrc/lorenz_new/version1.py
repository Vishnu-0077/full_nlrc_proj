import numpy as np
import random
from scipy import linalg
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class nlfea:
    def __init__(self,b=0.5,n=10,test_len=10,reg=1e-8,k=3,degree=3):
        self.b = b
        self.n = n
        self.test_len = test_len
        self.reg = reg
        self.k = k
        self.degree = degree
        self.feature_scalar = StandardScaler()
        self.insize = 3
    
    def build_features(self,data,delay_X_train):
        k = self.k
        deg = self.degree
        insize = self.insize
        delay_buffer = delay_X_train.copy()
        self.poly = PolynomialFeatures(degree=deg)
        dummy_lin = np.zeros(insize*(k+1))
        dummy_poly = self.poly.fit_transform(dummy_lin.reshape(1,-1))
        self.feat_size = dummy_poly.shape[1]

        history_x = np.zeros((self.feat_size,len(data)))
        for i in range(len(data)):
            lin = np.concatenate((delay_buffer.flatten(),data[i]))
            delay_buffer = np.vstack((delay_buffer[1:],data[i]))
            poly_feat = self.poly.transform(lin.reshape(1,-1))
            history_x[:,i] = poly_feat.flatten()
        return history_x    
        
    
    def fit(self,data,y_data,delay_X_train):
        yt = y_data
        yt = yt.T
        reg = self.reg
        history_x = self.build_features(data,delay_X_train)
        self.w_out = np.dot(np.dot(yt,history_x.T),linalg.inv(np.dot(history_x,history_x.T)+np.eye(history_x.shape[0])*reg))
        return self
    
    def predict(self,u,delay_X_test):
        test_len = self.test_len
        k = self.k
        insize = self.insize
        Y = np.zeros((3,test_len))
        delay_buffer = delay_X_test.copy()
        for i in range(test_len):
            lin = np.concatenate((delay_buffer.flatten(),u))
            poly_feat = self.poly.transform(lin.reshape(1,-1))
            y = np.dot(self.w_out,poly_feat.flatten())
            y = np.clip(y,-3,3)
            Y[:, i] = y
            delay_buffer = np.vstack((delay_buffer[1:],u))
            u = y

        return Y



def logistic_data(a = 3.95, i = 0.5422,length = 10000):
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

def mackey_glass(length=2000, tau=17, beta=0.2, gamma=0.1, n=10, dt=1.0):
    x = np.zeros(length)
    
    # initial condition (important!)
    x[:tau+1] = 1.2  
    
    for t in range(tau, length-1):
        x_tau = x[t - tau]
        x[t+1] = x[t] + dt * (beta * x_tau / (1 + x_tau**n) - gamma * x[t])
    
    return x[100:]

def chebyshev_map(length=1500,k=4,x0=0.123456):
    x = np.zeros(length)
    x[0] = x0

    for t in range(1,length):
        x[t] = np.cos(k*np.arccos(x[t-1]))
    return x

from scipy.integrate import solve_ivp
def generate_lorenz(
    n_steps=15000,
    dt=0.01,
    sigma=10.0,
    rho=28.0,
    beta=8/3,
    initial_state=(1.0, 1.0, 1.0)
):
    def lorenz(t, state):
        x, y, z = state

        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z

        return [dx, dy, dz]

    t_span = (0, n_steps * dt)
    t_eval = np.arange(0, n_steps * dt, dt)

    sol = solve_ivp(
        lorenz,
        t_span,
        initial_state,
        t_eval=t_eval,
        method='RK45'
    )

    return sol.y.T

def generate_rossler(
    t_max=500,
    dt=0.01,
    initial_state=(1.0, 1.0, 1.0),
    a=0.2,
    b=0.2,
    c=5.7,
    discard=5000
):
    def rossler(t, state):
        x, y, z = state

        dxdt = -y - z
        dydt = x + a * y
        dzdt = b + z * (x - c)

        return [dxdt, dydt, dzdt]

    t_eval = np.arange(0, t_max, dt)

    sol = solve_ivp(
        rossler,
        (0, t_max),
        initial_state,
        t_eval=t_eval,
        method="RK45"
    )

    data = sol.y.T

    if discard > 0:
        data = data[discard:]

    return data

def generate_rossler(
    t_max=500,
    dt=0.01,
    initial_state=(1.0, 1.0, 1.0),
    a=0.2,
    b=0.2,
    c=5.7,
    discard=5000
):
    def rossler(t, state):
        x, y, z = state

        dxdt = -y - z
        dydt = x + a * y
        dzdt = b + z * (x - c)

        return [dxdt, dydt, dzdt]

    t_eval = np.arange(0, t_max, dt)

    sol = solve_ivp(
        rossler,
        (0, t_max),
        initial_state,
        t_eval=t_eval,
        method="RK45"
    )

    data = sol.y.T

    if discard > 0:
        data = data[discard:]

    return data

scalar = StandardScaler()
data = generate_rossler()
data = data[0:]
train_len = 10000
test_len = 1000
k = 9

p = 0
push = train_len+p+k

X_train = data[k:k+train_len]
delay_X_train = data[:k]
X_test = data[push:push+test_len]
delay_X_test = data[push-k:push]

y_train = data[k+1:k+train_len+1]
y_test = data[1+push:test_len+1+push]

X_train = scalar.fit_transform(X_train)
delay_X_train = scalar.transform(delay_X_train)
y_train = scalar.transform(y_train)
X_test = scalar.transform(X_test)
delay_X_test = scalar.transform(delay_X_test)

model = nlfea(test_len=test_len,degree=2,k=k,n=4,reg=1e-4)
model.fit(X_train,y_train,delay_X_train)
y_pred = model.predict(X_test[0],delay_X_test).T
y_pred = scalar.inverse_transform(y_pred)

print(f'shape of Y is {y_pred.shape}')
print(f'mse of x is {mean_squared_error(y_test[:,0],y_pred[:,0])}')
print(f'mse of y is {mean_squared_error(y_test[:,1],y_pred[:,1])}')
print(f'mse of z is {mean_squared_error(y_test[:,2],y_pred[:,2])}')
print(f'mse of all is {mean_squared_error(y_test,y_pred)}')

plt.plot(np.arange(len(y_test)),y_test,c='r',label='real')
plt.plot(np.arange(len(y_pred)),y_pred,c='b',label='predicted')
plt.legend()
plt.show()


