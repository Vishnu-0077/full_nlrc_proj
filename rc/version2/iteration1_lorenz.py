import numpy as np
from scipy import linalg
import pandas as pd
from scipy.integrate import solve_ivp
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

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
    return x[100:]

def tent_map(mu=1.9999, x0=0.5422, length=2000):
    x = np.zeros(length)
    x[0] = x0

    for t in range(length-1):
        if x[t] < 0.5:
            x[t+1] = mu*x[t]
        else:
            x[t+1] = mu*(1-x[t])

    return x

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

import numpy as np
from scipy.integrate import solve_ivp

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
scalar = StandardScaler()
data = scalar.fit_transform(data)
y_data = data[1:]


insize = 3
outsize = 3
ressize = 500
init_len = 100
train_length = 3100
test_length = 500
a=0.3

np.random.seed(42)
W_in = np.random.rand(ressize,insize+1) - 0.5
W = np.random.rand(ressize,ressize) - 0.5

spectral_radius = np.max(np.abs(linalg.eigvals(W)))
target_radius = 0.95
W = W*target_radius/spectral_radius

x = np.zeros(ressize).reshape(-1,1)
x_history = np.zeros((1+insize+ressize,train_length-init_len))

for i in range(train_length):
    u = data[i].reshape(-1,1)
    x = x*(1-a)+a*np.tanh(np.dot(W,x)+np.dot(W_in,np.vstack((1,u))))
    if i>=init_len:
        x_history[:,i-init_len] = np.vstack((1,u,x.reshape(-1,1)))[:,0]

print(f'shape of x_history is {x_history.shape}')
p_xhistory = pd.DataFrame(x_history)

yt = y_data[init_len:train_length].T
reg = 1e-4
w_out = np.dot(np.dot(yt,x_history.T),linalg.inv(np.dot(x_history,x_history.T)+np.eye(ressize+insize+1)*reg))
Y = np.zeros((outsize,test_length))

print(f'shape of w_out is {w_out.shape}')

u = data[train_length]
for i in range(test_length):
    u = u.reshape(-1,1)
    x = x*(1-a)+a*np.tanh(np.dot(W,x)+np.dot(W_in,np.vstack((1,u))))
    y = np.dot(w_out,np.vstack((1,u,x.reshape(-1,1 ))))
    Y[:,i] = y[:,0]
    u = y


y_test = y_data[train_length:train_length+test_length]
Y = Y.T
Y = scalar.inverse_transform(Y)
y_test = scalar.inverse_transform(y_test)
print(f'shape of Y is {Y.shape}')
print(f'mse of x is {mean_squared_error(y_test[:,0],Y[:,0])}')
print(f'mse of y is {mean_squared_error(y_test[:,1],Y[:,1])}')
print(f'mse of z is {mean_squared_error(y_test[:,2],Y[:,2])}')
print(f'mse of all is {mean_squared_error(y_test,Y)}')

import matplotlib.pyplot as plt
plt.plot(y_test)
plt.plot(Y)
plt.show()