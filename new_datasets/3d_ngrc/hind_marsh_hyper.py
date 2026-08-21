import numpy as np
import random
from scipy import linalg
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from dtaidistance import dtw,dtw_ndim

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
            y=np.clip(y,-3,3)
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
import numpy as np
from scipy.integrate import solve_ivp

def generate_lorenz(
    n_steps=15000,
    discard_steps=2000,  # <-- Add a parameter for transient steps
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

    # Calculate time milestones
    transient_time = discard_steps * dt
    total_time = (n_steps + discard_steps) * dt

    # The solver must run from t=0 to the very end
    t_span = (0, total_time)
    
    # But it will ONLY store the points starting from transient_time
    t_eval = np.arange(transient_time, total_time, dt)

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


def generate_hindmarsh_rose(
    t_max=500,
    dt=0.01,
    initial_state=(-1.6, -10.0, 3.0),
    a=1.0,
    b=3.0,
    c=1.0,
    d=5.0,
    r=0.006,
    s=4.0,
    x0=-1.6,
    discard=1000
):
    from scipy.integrate import solve_ivp
    import numpy as np

    def hr_equations(t, state):
        x, y, z = state

        dx = y - a*x**3 + b*x**2 - z
        dy = c - d*x**2 - y
        dz = r * (s*(x - x0) - z)

        return [dx, dy, dz]

    # Total number of points including discarded transient
    n_total = int(t_max / dt) + discard

    t_full = np.arange(n_total) * dt

    sol = solve_ivp(
        hr_equations,
        (t_full[0], t_full[-1]),
        initial_state,
        t_eval=t_full,
        method="RK45"
    )

    data_full = sol.y.T

    # Remove transient
    data = data_full[discard:]
    t = t_full[discard:]

    return data

def weight_mse(means_values,weights):
    summ = 0
    for i in range(len(means_values)):
        summ += weights[i]*(means_values[i])
    return summ


def gaussian_weights(mean_values):
    weights = np.zeros(len(mean_values))
    med = np.median(mean_values)
    mad = np.median(np.abs(mean_values-med))
    sigma = 1.4826*mad + 1e-12
    for i in range(len(mean_values)):
        weights[i] = np.exp(-((mean_values[i]-med)**2)/(2*sigma**2))
    weights = weights/np.sum(weights)
    return weights

def inverse_distance_weights(mean_values):
    weights = np.zeros(len(mean_values))
    med = np.median(mean_values)
    for i in range(len(mean_values)):
        weights[i] = 1/(1+(np.abs(mean_values[i]-med)))
    weights = weights/np.sum(weights)
    return weights

scalar = StandardScaler()
data = generate_hindmarsh_rose()
n_splits = 5
train_len = 1000
test_len = 500
skip_len = 500
k=3
mse_lst = []
reg_lst = [1e-6,1e-5,1e-4,1e-3]

big_mse_lst = {}
for reg in reg_lst:
    small_mse_lst = []
    for i in range(0,n_splits*(train_len+1000),skip_len):
        X_train = data[i+k:i+k+train_len]
        X_train_delay = data[i:i+k]
        X_test = data[i+k+train_len:i+k+train_len+test_len]
        X_test_delay = data[i+train_len:i+k+train_len]

        y_train = data[i+k+1:i+k+train_len+1]
        y_test = data[i+1+k+train_len:i+1+k+train_len+test_len]

        X_train = scalar.fit_transform(X_train)
        X_train_delay = scalar.transform(X_train_delay)
        X_test = scalar.transform(X_test)
        X_test_delay = scalar.transform(X_test_delay)
        y_train = scalar.transform(y_train)

        model = nlfea(test_len=test_len,degree=2,k=k,reg=reg)
        model.fit(X_train,y_train,X_train_delay)
        y_pred = model.predict(X_test[0],X_test_delay).T
        y_pred = scalar.inverse_transform(y_pred)
        mse = mean_absolute_error(y_test,y_pred)
        small_mse_lst.append(mse)

    weights = inverse_distance_weights(small_mse_lst)
    cal_mse = float(weight_mse(small_mse_lst,weights))
    print(f'for {reg} cal mse is {cal_mse}')
    big_mse_lst[(reg)] = cal_mse


print(sorted(big_mse_lst.items(),key=lambda x:x[1]))



