import numpy as np
from sklearn.datasets import load_iris

X,y = load_iris(return_X_y=True)
print(type(set(y)))