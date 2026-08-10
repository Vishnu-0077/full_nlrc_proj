import numpy as np
from scipy import linalg
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

class rc_nl:

    def __init__(self,ressize=100,spectral_radius=0.95,insize=3,outsize=3,leak_rate=0.3,reg=1e-8,test_len=10,init_len=100,seed=42):
        self.ressize = ressize
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.reg = reg
        self.insize = 3
        self.outsize = 3
        self.test_len = test_len
        self.init_len = init_len
        self.seed = seed
        np.random.seed(seed)

        self.W_in = np.random.rand(ressize,insize+1) - 0.5
        self.W = np.random.rand(ressize,ressize) - 0.5
        self.x = np.zeros(ressize).reshape(-1,1)

        radius = np.max(np.abs(linalg.eigvals(self.W)))
        self.W = self.W * spectral_radius / radius

    def build_features(self,data):
        train_length = len(data)
        init_len = self.init_len
        self.x_history = np.zeros((self.ressize+self.insize+1,train_length-init_len))
        a = self.leak_rate
        W = self.W
        W_in = self.W_in
        for i in range(train_length):
            u = data[i].reshape(-1,1)
            self.x = self.x*(1-a)+a*np.tanh(np.dot(W,self.x)+np.dot(W_in,np.vstack((1,u))))
            if i>=init_len:
                self.x_history[:,i-init_len] = np.vstack((1,u,self.x.reshape(-1,1)))[:,0]

        return self.x_history

    def fit(self,data,y_data):
        yt = y_data[self.init_len:]
        yt = yt.T
        reg = self.reg
        history_x = self.build_features(data)
        self.w_out = np.dot(np.dot(yt,history_x.T),linalg.inv(np.dot(history_x,history_x.T)+np.eye(self.ressize+self.insize+1)*reg))
        return self

    def predict(self,u):
        Y = np.zeros((self.outsize,self.test_len))
        x = self.x.copy()
        a = self.leak_rate
        W = self.W
        W_in = self.W_in
        for i in range(self.test_len):
            u = u.reshape(-1,1)
            x = x*(1-a)+a*np.tanh(np.dot(W,x)+np.dot(W_in,np.vstack((1,u))))
            y = np.dot(self.w_out,np.vstack((1,u,x.reshape(-1,1 ))))
            Y[:,i] = y[:,0]
            u = y
        return Y
    
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

data = generate_rossler()
train_len = 3100
test_len = 1000


X_train = data[:train_len]
y_train = data[1:train_len+1]
X_test = data[train_len:train_len+test_len]
y_test = data[train_len+1:train_len+test_len+1]

scalar = StandardScaler()

X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)
y_train = scalar.transform(y_train)

model = rc_nl(test_len=test_len,reg=1e-7,ressize=500)
model.fit(X_train,y_train)
y_pred = model.predict(X_test[0])
y_pred = scalar.inverse_transform(y_pred.T)

print(f'shape of Y is {y_pred.shape}')
print(f'mse of x is {mean_squared_error(y_test[:,0],y_pred[:,0])}')
print(f'mse of y is {mean_squared_error(y_test[:,1],y_pred[:,1])}')
print(f'mse of z is {mean_squared_error(y_test[:,2],y_pred[:,2])}')
print(f'mse of all is {mean_squared_error(y_test,y_pred)}')

import matplotlib.pyplot as plt
plt.plot(np.arange(len(y_test)),y_test,c='r',label='real')
plt.plot(np.arange(len(y_pred)),y_pred,c='b',label='predicted')
plt.legend()
plt.show()