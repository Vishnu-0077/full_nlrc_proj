import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

data = pd.read_csv('/home/vishnu/Downloads/traindata.csv')
data = data[1000:65000]
X = data['xs']
y = data['ys']
z = data['zs']

print(data.describe())

ax.plot(X,y,z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

