import numpy as np
from scipy import linalg
import pandas as pd
from scipy.integrate import solve_ivp
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
class NLRC:
    def __init__(self,n=5,ressize=100,spectral_radius=0.95,leak_rate=0.3,reg=1e-8,train_len=1100,test_len=10,init_len=100,seed=42):
        self.n = n
        self.ressize = ressize
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.reg = reg
        self.train_len = train_len
        self.test_len = test_len
        self.init_len = init_len
        self.seed = seed
        np.random.seed(seed)

        self.W_in = np.random.rand(ressize,9+1) - 0.5
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
        X[0] = u.item()
        for i in range(self.n-1):
            X[i+1] = self.neuron_gen(X[i])
        return X
    
    def energy(self,x):
        if len(x) ==0:
            return 0
        return np.mean(np.square(x))
    
    def variance(self,x):
        if len(x) ==0:
            return 0
        return np.var(x)
    
    def build_features(self,data):
        W = self.W
        W_in = self.W_in
        a = self.leak_rate
        train_len = self.train_len
        init_len = self.init_len
        self.x_history = np.zeros((self.ressize+1,train_len-init_len))
        for i in range(train_len):
            lin = np.zeros(9)
            u = data[i]
            for j in range(len(u)):
                ss = self.neuron_iterator(u[j])
                ene = self.energy(ss)
                var = self.variance(ss)
                lin[3*j+0] = u[j]
                lin[3*j+1] = ene
                lin[3*j+2] = var
            lin = lin.reshape(-1,1)
            self.x = self.x*(1-a)+a*np.tanh(np.dot(W,self.x)+np.dot(W_in,np.vstack((1,lin))))
            if i>init_len:
                self.x_history[:,i-init_len] = np.vstack((1,self.x.reshape(-1,1))).flatten()
        return self.x_history

    def fit(self,data,y_data):
        yt = y_data[self.init_len:]
        yt = yt.T
        reg = self.reg
        history_x = self.build_features(data)
        self.w_out = np.dot(np.dot(yt,history_x.T),linalg.inv(np.dot(history_x,history_x.T)+np.eye(history_x.shape[0])*reg))
        return self

    def predict(self,data):
        x = self.x.copy()
        W = self.W
        W_in = self.W_in
        a = self.leak_rate
        Y = np.zeros((3,self.test_len))
        u = data
        for i in range(self.test_len):
            lin = np.zeros(9)
            for j in range(len(u)):
                ss = self.neuron_iterator(u[j])
                ene = self.energy(ss)
                var = self.variance(ss)
                lin[3*j+0] = u[j].item()
                lin[3*j+1] = ene
                lin[3*j+2] = var
            lin = lin.reshape(-1,1)
            x = x*(1-a)+a*np.tanh(np.dot(W,x)+np.dot(W_in,np.vstack((1,lin))))
            y = np.dot(self.w_out,np.vstack((1,x.reshape(-1,1 ))))
            Y[:,i] = y.flatten()
            u = y
        return Y.T

def generate_lorenz(
    n_steps=10000,
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
test_len = 500


X_train = data[:train_len]
y_train = data[1:train_len+1]
X_test = data[train_len:train_len+test_len]
y_test = data[train_len+1:train_len+test_len+1]

scalar = MinMaxScaler(feature_range=(0,1))

X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)
y_train = scalar.transform(y_train)

n_lst = [3,4,5,6,8,10]
reg_lst = [1e-3,1e-4,1e-5,1e-6,1e-7,1e-7]
res_lst = [500]

best_results = {}
for n in n_lst:
    for reg in reg_lst:
        for res in res_lst:
            model = NLRC(n=n,test_len=test_len,train_len=train_len,reg=reg,ressize=res)
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test[0])
            y_pred = scalar.inverse_transform(y_pred)
            mse = mean_squared_error(y_test,y_pred)
            print(f'n = {n}, reg = {reg}, res = {res}, mse = {mse}')
            best_results[(n,reg,res)] = mse
print('top results are')
print(sorted(best_results.items(),key=lambda x:x[1])[:5])