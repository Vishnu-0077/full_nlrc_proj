import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel('/home/vishnu/Downloads/Mackey-Glass Time Series(taw17).xlsx')
data = data.drop('Unnamed: 0',axis = 1)

y = data['t+1']
X = data.drop('t+1',axis = 1)

print(y.to_numpy().reshape(-1,1).shape)
print(data.columns)

plt.subplot(1,2,1)
plt.scatter(X['t'][200:1050],y[200:1050])

plt.subplot(1,2,2)
plt.plot(X[200:500])

plt.show()
