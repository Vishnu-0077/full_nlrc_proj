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
import numpy as np
import os

k=3
train_len = 3000+k
test_len = 100
data = generate_lorenz()
data = data[train_len-k:]
os.makedirs('nl_rc_2/train_test_seperate/datasets/test',exist_ok=True)
np.save('nl_rc_2/train_test_seperate/datasets/test/lorenz.npy', data)
print('test data generated')
