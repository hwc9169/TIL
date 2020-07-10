import numpy as np
def CEE(x,y):
    delta =1e-7
    return -np.sum(y*np.log(x+delta))

y = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, ])
x = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
cost = CEE(x, y)
print(cost)

x = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
cost = CEE(x, y)
print(cost)