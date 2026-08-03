import numpy as np
def logistic_data(a = 3.95, i = 0.5422,length = 2000):
    x = np.zeros(length)
    x[0] = i
    for t in range(length-1):
        x[t+1] = a*x[t]*(1-x[t])
    return x[100:]

import matplotlib.pyplot as plt
data = logistic_data()
data=data[:10]
plt.plot(data)
plt.show()
