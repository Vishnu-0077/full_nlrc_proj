import numpy as np

def tent_map(mu=1.9999, x0=0.5422, length=2000):
    x = np.zeros(length)
    x[0] = x0

    for t in range(length-1):
        if x[t] < 0.5:
            x[t+1] = mu*x[t]
        else:
            x[t+1] = mu*(1-x[t])

    return x[100:]
import matplotlib.pyplot as plt

data = tent_map()
data=data[:10]
plt.plot(data)
plt.show()